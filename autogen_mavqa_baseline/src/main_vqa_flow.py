import asyncio
import re
import json
import os
import time
import sys
from autogen_core import CancellationToken

try:
    from config_loader import app_config
    from image_utils import process_image_for_vlm_agent
    from agents import (
        vqa_orchestrator, initial_vlm_agent, failure_analysis_agent,
        object_attribute_agent, reattempt_vlm_agent, llm_client_vllm,
        vlm_client_vllm
    )
    from prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_VQA_V2, INITIAL_VLM_SYSTEM_PROMPT_DEFAULT,
        FAILURE_ANALYSIS_SYSTEM_PROMPT, OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS, REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS,
        GRADING_SYSTEM_PROMPT_TEMPLATE
    )
    from autogen_agentchat.agents import AssistantAgent
except ImportError as e:
    print(f"ERROR: Error importing modules in main_vqa_flow.py: {e}.")
    print("Ensure all required files (config_loader, image_utils, agents, prompts, vllm_clients) exist and are accessible.")
    sys.exit(1)

def get_message_content(message: dict, default_if_none: str = "") -> str:
    if not message or not isinstance(message, dict): return default_if_none
    content = message.get("content", default_if_none)
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", default_if_none).strip()
    return default_if_none if content is None else str(content).strip()

def write_response_to_jsonl(response_data, filename):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create output directory {output_dir}, saving disabled for this item. Error: {e}")
            return

    try:
        sanitized_data = json.loads(json.dumps(response_data, default=str))
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sanitized_data) + '\n')
    except TypeError as e_type:
        print(f"Warning: TypeError serializing data for JSONL file {filename}: {e_type}. Skipping save for this item.")
    except Exception as e:
        print(f"Warning: Error writing to JSONL file {filename}: {e}. Skipping save for this item.")

async def run_vqa_pipeline(image_path: str, question: str, question_id: str ="unknown_qid", target_answer: str = None) -> dict:
    start_time = time.time()
    current_config = app_config
    inference_settings = current_config.get("inference_settings", {})
    dataset_s_config = current_config.get("datasets", {})
    verbose = inference_settings.get("verbose", True)

    response_data = {
        "question_id": str(question_id), "image_path": image_path, "question": question,
        "target_answer": target_answer, "initial_answer": "", "final_answer": "",
        "match_baseline_failed": False, "is_numeric_reattempt": False,
        "analysis_output": "", "object_attributes_queried": "", "reattempt_answer": "",
        "grades": [], "processing_time_seconds": 0.0, "error": None
    }

    agent_list = [vqa_orchestrator, initial_vlm_agent, failure_analysis_agent, object_attribute_agent, reattempt_vlm_agent]
    client_list = [vlm_client_vllm, llm_client_vllm]
    if any(agent is None for agent in agent_list) or any(client is None for client in client_list):
        error_msg = "One or more core agents or clients are not initialized."
        print(f"ERROR for QID {question_id}: {error_msg}")
        response_data["error"] = "InitializationError"
        response_data["final_answer"] = "[Answer Failed]"
        return response_data

    try:
        base64_image_url = process_image_for_vlm_agent(image_path, current_config)
        if not base64_image_url: raise ValueError("Image processing returned empty result.")
    except Exception as e_img:
        if verbose: print(f"ERROR processing image {os.path.basename(image_path)} for QID {question_id}: {e_img}")
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        response_data["final_answer"] = "[Answer Failed]"
        return response_data

    initial_answer_text = "[Agent Communication Error]"
    try:
        active_dataset_name = dataset_s_config.get("dataset_name", "default")
        if active_dataset_name == "vqa-v2":
            initial_vlm_agent.system_message = INITIAL_VLM_SYSTEM_PROMPT_VQA_V2
        else:
            initial_vlm_agent.system_message = INITIAL_VLM_SYSTEM_PROMPT_DEFAULT

        initial_vlm_message_content = [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
        initial_user_message_to_agent = {"role": "user", "content": initial_vlm_message_content}

        if verbose: print(f"   Orchestrator: Calling Initial_VLM_Agent...")
        await initial_vlm_agent.on_reset(cancellation_token=CancellationToken())

        chat_result_initial = await vqa_orchestrator.a_initiate_chat(
            recipient=initial_vlm_agent, message=initial_user_message_to_agent,
            max_turns=1, summary_method="last_msg", silent=(not verbose)
        )
        initial_answer_text = get_message_content(chat_result_initial.chat_history[-1] if chat_result_initial else {})
        if not initial_answer_text: initial_answer_text = "[Initial VLM Agent returned empty message]"

    except Exception as e_agent_call:
        print(f"ERROR during Initial_VLM_Agent call for QID {question_id}: {e_agent_call}")
        response_data["error"] = f"InitialAgentCallError: {str(e_agent_call)}"
        initial_answer_text = "[Agent Call Failed]"

    response_data["initial_answer"] = initial_answer_text
    final_answer_text = initial_answer_text
    if verbose: print(f"   Initial_VLM_Agent response: <<< {initial_answer_text} >>>")

    is_direct_failure_marker = (
        "[Answer Failed]" in initial_answer_text or
        "sorry" in initial_answer_text.lower() or
        "unable to answer" in initial_answer_text.lower() or
        "cannot answer" in initial_answer_text.lower() or
        not initial_answer_text.strip() or
        "[Agent Communication Error]" in initial_answer_text or
        "[Agent Call Failed]" in initial_answer_text or
        "[Initial VLM Agent returned empty message]" in initial_answer_text
    )
    is_problematic_numeric_answer = (
        ("[Non-zero Numeric Answer]" in initial_answer_text and "[Answer Failed]" in initial_answer_text) or
        ("[Zero Numeric Answer]" in initial_answer_text and "[Answer Failed]" in initial_answer_text)
    )
    is_any_failed_numeric = "[Numeric Answer]" in initial_answer_text and is_direct_failure_marker

    match_baseline_failed = is_direct_failure_marker or is_problematic_numeric_answer or is_any_failed_numeric
    if inference_settings.get("force_multi_agents", False) and not response_data.get("error"):
        if not match_baseline_failed and verbose:
            print("   Note: force_multi_agents=True, forcing reattempt.")
        match_baseline_failed = True

    response_data["match_baseline_failed"] = match_baseline_failed
    reattempt_performed = False

    if match_baseline_failed and response_data["error"] is None:
        reattempt_performed = True
        if verbose: print("   Orchestrator: Analyzing failure...")

        analysis_output = "[Analysis Agent Communication Error]"
        try:
            failure_analysis_agent.system_message = FAILURE_ANALYSIS_SYSTEM_PROMPT
            failure_analysis_user_message = (
                f"Original Question: '{question}'\n"
                f"Initial VLM Response: '{initial_answer_text}'\n"
                f"Analyze failure. Suggest strategy: 'numeric reattempt needed for: [item]' OR 'general reattempt, focus on: [items]'.")
            if verbose: print(f"   Orchestrator: Calling Failure_Analysis_Agent...")
            await failure_analysis_agent.on_reset(cancellation_token=CancellationToken())

            chat_result_analysis = await vqa_orchestrator.a_initiate_chat(
                recipient=failure_analysis_agent, message=failure_analysis_user_message,
                max_turns=1, summary_method="last_msg", silent=(not verbose)
            )
            analysis_output = get_message_content(chat_result_analysis.chat_history[-1] if chat_result_analysis else {})
            if not analysis_output: analysis_output = "[Analysis Agent returned empty message]"

        except Exception as e_agent_call:
            print(f"ERROR during Failure_Analysis_Agent call for QID {question_id}: {e_agent_call}")
            response_data["error"] = f"FailureAnalysisAgentError: {str(e_agent_call)}"
            analysis_output = "[Analysis Failed]"

        response_data["analysis_output"] = analysis_output
        if verbose: print(f"   Failure_Analysis_Agent response: <<< {analysis_output} >>>")

        is_numeric_reattempt = "numeric reattempt needed" in analysis_output.lower() and "[Analysis Failed]" not in analysis_output
        response_data["is_numeric_reattempt"] = is_numeric_reattempt
        reattempt_answer_text = "[Reattempt Not Performed or Failed]"

        if is_numeric_reattempt:
            if verbose: print("   Orchestrator: Numeric reattempt. Re-prompting VLM for careful counting...")
            item_to_count = "the relevant objects"
            item_match = re.search(r"numeric reattempt needed for:\s*(.+)", analysis_output, re.IGNORECASE)
            if item_match: item_to_count = item_match.group(1).strip()
            
            try:
                active_dataset_name = dataset_s_config.get("dataset_name", "default")
                if active_dataset_name == "vqa-v2": reattempt_vlm_agent.system_message = REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS
                else: reattempt_vlm_agent.system_message = REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS
            except Exception as e_sys_msg: print(f"Warning: Error updating system message for reattempt_vlm_agent: {e_sys_msg}")

            numeric_reattempt_user_prompt = (
                f"Original question: '{question}'\n"
                f"Previous Answer: '{initial_answer_text}'\n"
                f"Analysis suggests issues with counting: '{item_to_count}'.\n"
                f"INSTRUCTION: Look carefully. Count step-by-step how many '{item_to_count}' are present. List instances if possible. Respond starting with '[Reattempted Answer]' followed ONLY by the final number/answer."
            )
            numeric_reattempt_message = {"role": "user", "content": [{"type": "text", "text": numeric_reattempt_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]}

            if verbose: print(f"   Orchestrator: Calling Reattempt_VLM_Agent for counting...")
            try:
                await reattempt_vlm_agent.on_reset(cancellation_token=CancellationToken())
                chat_result_num_reattempt = await vqa_orchestrator.a_initiate_chat(
                    recipient=reattempt_vlm_agent, message=numeric_reattempt_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                reattempt_answer_text = get_message_content(chat_result_num_reattempt.chat_history[-1] if chat_result_num_reattempt else {})
                if not reattempt_answer_text: reattempt_answer_text = "[Numeric Reattempt Agent returned empty message]"
            except Exception as e_agent_call:
                print(f"ERROR during numeric reattempt agent call for QID {question_id}: {e_agent_call}")
                response_data["error"] = f"{response_data.get('error', '')}; NumericReattemptAgentError: {str(e_agent_call)}".strip("; ")
                reattempt_answer_text = "[Numeric Reattempt Agent Call Failed]"

        elif "[Analysis Failed]" not in analysis_output:
            if verbose: print("   Orchestrator: General reattempt. Step 1: Getting textual object descriptions...")
            textually_specified_objects = analysis_output
            focus_match = re.search(r"general reattempt, focus on:\s*(.+)", analysis_output, re.IGNORECASE)
            if focus_match: textually_specified_objects = focus_match.group(1).strip()
            
            object_attributes_descriptions = "[Attribute Query Agent Error]"
            try:
                object_attribute_agent.system_message = OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS
                attribute_query_user_prompt = (
                    f"Original VQA Question: '{question}'.\n"
                    f"Analysis suggests focusing on: '{textually_specified_objects}'.\n"
                    f"Describe these items visually based on the image, focusing on relevant details."
                )
                attribute_query_content = [{"type": "text", "text": attribute_query_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
                attribute_query_message = {"role": "user", "content": attribute_query_content}

                if verbose: print(f"   Orchestrator: Calling Object_Attribute_Agent for descriptions...")
                await object_attribute_agent.on_reset(cancellation_token=CancellationToken())
                chat_result_attr = await vqa_orchestrator.a_initiate_chat(
                    recipient=object_attribute_agent, message=attribute_query_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                object_attributes_descriptions = get_message_content(chat_result_attr.chat_history[-1] if chat_result_attr else {})
                if not object_attributes_descriptions: object_attributes_descriptions = "[Attribute Agent returned empty message]"

            except Exception as e_agent_call:
                print(f"ERROR during object attribute agent call for QID {question_id}: {e_agent_call}")
                response_data["error"] = f"{response_data.get('error', '')}; AttributeQueryAgentError: {str(e_agent_call)}".strip("; ")
                object_attributes_descriptions = "[Attribute Query Agent Call Failed]"

            response_data["object_attributes_queried"] = object_attributes_descriptions
            if verbose: print(f"   Object_Attribute_Agent response: <<< {object_attributes_descriptions} >>>")

            if "[Attribute Query Agent Error]" not in object_attributes_descriptions and "[Attribute Query Agent Call Failed]" not in object_attributes_descriptions:
                try:
                    active_dataset_name = dataset_s_config.get("dataset_name", "default")
                    if active_dataset_name == "vqa-v2": reattempt_vlm_agent.system_message = REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS
                    else: reattempt_vlm_agent.system_message = REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS
                except Exception as e_sys_msg: print(f"Warning: Error updating system message for reattempt_vlm_agent: {e_sys_msg}")

                general_reattempt_user_prompt = (
                    f"Original question: '{question}'.\n"
                    f"Previous Answer: '{initial_answer_text}'.\n"
                    f"Analysis suggested focus on: '{textually_specified_objects}'.\n"
                    f"Descriptions of focus items: '{object_attributes_descriptions}'.\n"
                    f"INSTRUCTION: Using image and ALL text context, provide an improved answer. Reason first, then answer starting ONLY with '[Reattempted Answer]'."
                )
                general_reattempt_content = [{"type": "text", "text": general_reattempt_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
                general_reattempt_message = {"role": "user", "content": general_reattempt_content}

                if verbose: print(f"   Orchestrator: Calling Reattempt_VLM_Agent with new context...")
                try:
                    await reattempt_vlm_agent.on_reset(cancellation_token=CancellationToken())
                    chat_result_gen_reattempt = await vqa_orchestrator.a_initiate_chat(
                        recipient=reattempt_vlm_agent, message=general_reattempt_message,
                        max_turns=1, summary_method="last_msg", silent=(not verbose)
                    )
                    reattempt_answer_text = get_message_content(chat_result_gen_reattempt.chat_history[-1] if chat_result_gen_reattempt else {})
                    if not reattempt_answer_text: reattempt_answer_text = "[General Reattempt Agent returned empty message]"
                except Exception as e_agent_call:
                    print(f"ERROR during general reattempt agent call for QID {question_id}: {e_agent_call}")
                    response_data["error"] = f"{response_data.get('error', '')}; GeneralReattemptAgentError: {str(e_agent_call)}".strip("; ")
                    reattempt_answer_text = "[General Reattempt Agent Call Failed]"
            else:
                if verbose: print("   Orchestrator: Skipping final reattempt due to attribute query failure.")
                reattempt_answer_text = "[Reattempt Skipped Due to Attribute Query Failure]"
        
        response_data["reattempt_answer"] = reattempt_answer_text
        possible_failure_markers = ["[Agent Communication Error]", "[Reattempt Not Performed or Failed]",
                                    "[Reattempt Skipped", "[Agent Call Failed]", "Agent returned empty message"]
        if not any(marker in reattempt_answer_text for marker in possible_failure_markers) and reattempt_answer_text.strip():
            final_answer_text = reattempt_answer_text
        else:
            final_answer_text = initial_answer_text
            if verbose: print(f"   Note: Reattempt failed or produced invalid response, using initial answer as final.")

    response_data["final_answer"] = final_answer_text
    if verbose: print(f"   Final Answer selected for QID {question_id}: <<< {final_answer_text} >>>")

    if target_answer and str(target_answer).strip() and response_data["error"] is None:
        if verbose: print(f"   Orchestrator: Performing grading against target: '{target_answer}'")
        if llm_client_vllm:
            temp_grader_agent = AssistantAgent( name=f"Temp_Grader_{question_id}", model_client=llm_client_vllm )
            grades = []
            for i in range(3):
                grader_id = i
                try:
                    temp_grader_agent.system_message = GRADING_SYSTEM_PROMPT_TEMPLATE.format(grader_id=grader_id)
                except Exception as e_sys_msg: print(f"Warning: Error updating system message for temp_grader_agent: {e_sys_msg}")

                grading_user_message = (
                    f"Question: '{question}'\n"
                    f"Target Answer: '{target_answer}'\n"
                    f"Model's Final Answer: '{final_answer_text}'\n"
                    f"Provide grade."
                )
                grade_text = f"[Grader {grader_id}] [ErrorInGrading]"
                try:
                    if verbose: print(f"     Orchestrator: Calling Grader {grader_id}...")
                    await temp_grader_agent.on_reset(cancellation_token=CancellationToken())
                    grading_chat_result = await vqa_orchestrator.a_initiate_chat(
                        recipient=temp_grader_agent, message=grading_user_message,
                        max_turns=1, summary_method="last_msg", silent=(not verbose)
                    )
                    grade_text = get_message_content(grading_chat_result.chat_history[-1] if grading_chat_result else {}, grade_text)
                    if not grade_text.strip().startswith(f"[Grader {grader_id}]"):
                        grade_text = f"[Grader {grader_id}] [Malformed Response] " + grade_text
                except Exception as e_agent_call:
                    print(f"ERROR during grading agent call (Grader {grader_id}) for QID {question_id}: {e_agent_call}")
                    response_data["error"] = f"{response_data.get('error', '')}; GradingAgentError{grader_id}: {str(e_agent_call)}".strip("; ")
                grades.append(grade_text)
            response_data["grades"] = grades
            if verbose: print(f"   Grading results: {grades}")
        else:
            if verbose: print("   Grading skipped: LLM client not available.")
            response_data["grades"] = ["Grading Skipped - LLM Client Error"] * 3
    elif response_data["error"]:
        if verbose: print("   Grading skipped due to earlier pipeline error.")
        response_data["grades"] = ["Grading Skipped - Pipeline Error"] * 3
    else:
        if verbose: print("   Grading skipped: No valid target_answer provided.")

    end_time = time.time()
    response_data["processing_time_seconds"] = round(end_time - start_time, 2)
    if verbose: print(f"   Total processing time for QID {question_id}: {response_data['processing_time_seconds']:.2f}s")

    if verbose: print(f"--- VQA Pipeline End for QID: {question_id} ---")
    return response_data