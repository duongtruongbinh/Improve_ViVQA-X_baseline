# main_vqa_flow.py
from autogen_core import CancellationToken
import os
import time
import json
import re
import sys
import traceback

try:
    from config_loader import app_config
    from image_utils import process_image_for_vlm_agent
    from agents import (
        vqa_orchestrator, initial_vlm_agent, failure_analysis_agent,
        object_attribute_agent, reattempt_vlm_agent
    )
    from vllm_clients import llm_client_vllm, vlm_client_vllm 
    from prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_VQA_V2, INITIAL_VLM_SYSTEM_PROMPT_DEFAULT,
        FAILURE_ANALYSIS_SYSTEM_PROMPT, OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS, REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS,
        GRADING_SYSTEM_PROMPT_TEMPLATE
    )
    # Change from AssistantAgent to ConversableAgent
    from autogen.agentchat.conversable_agent import ConversableAgent
except ImportError as e:
    print(f"ERROR: Error importing modules in main_vqa_flow.py: {e}.")
    print("Ensure all required files (config_loader, image_utils, agents, prompts, vllm_clients) exist and are accessible, and AutoGen packages are installed.")
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
    token = CancellationToken()

    response_data = {
        "question_id": str(question_id), "image_path": image_path, "question": question,
        "target_answer": target_answer, "initial_answer": "", "final_answer": "",
        "match_baseline_failed": False, "is_numeric_reattempt": False,
        "analysis_output": "", "object_attributes_queried": "", "reattempt_answer": "",
        "grades": [], "processing_time_seconds": 0.0, "error": None
    }

    agent_instances_list = [vqa_orchestrator, initial_vlm_agent, failure_analysis_agent, object_attribute_agent, reattempt_vlm_agent]
    client_instances_list = [vlm_client_vllm, llm_client_vllm] 
    if any(agent is None for agent in agent_instances_list) or any(client is None for client in client_instances_list):
        error_msg = "One or more core agents or clients are not initialized."
        print(f"ERROR for QID {question_id}: {error_msg}")
        response_data["error"] = "InitializationError"
        response_data["final_answer"] = "[Answer Failed]"
        return response_data

    if question is None:
        question = ""

    valid_image_url_for_message = None
    try:
        processed_image_url = process_image_for_vlm_agent(image_path, current_config)
        if isinstance(processed_image_url, str):
            is_valid_data_uri = False
            if processed_image_url.startswith("data:image/") and ";base64," in processed_image_url:
                parts = processed_image_url.split("base64,")
                if len(parts) == 2 and parts[1].strip(): 
                    is_valid_data_uri = True
            
            is_valid_http_url = processed_image_url.startswith("http://") or processed_image_url.startswith("https://")

            if is_valid_data_uri or is_valid_http_url:
                valid_image_url_for_message = processed_image_url
            elif processed_image_url: 
                raise ValueError(f"Image processing returned a malformed or unsupported URL string: '{processed_image_url[:100]}...'")
            else: 
                 raise ValueError("Image processing returned an empty URL string.")
        elif processed_image_url is not None: 
            raise ValueError(f"Image processing returned an unexpected type for URL: {type(processed_image_url)}")
        else: 
            raise ValueError("Image processing returned None, and an image is required for VQA.")

    except Exception as e_img:
        if verbose: print(f"ERROR processing image {os.path.basename(image_path)} for QID {question_id}: {e_img}")
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        response_data["final_answer"] = "[Answer Failed]"
        return response_data

    initial_answer_text = "[Agent Communication Error]"
    try:
        active_dataset_name = dataset_s_config.get("dataset_name", "default")
        if active_dataset_name == "vqa-v2":
            initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_VQA_V2)
        else:
            initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_DEFAULT)

        initial_vlm_message_parts = [{"type": "text", "text": question}]
        if valid_image_url_for_message:
            initial_vlm_message_parts.append({"type": "image_url", "image_url": {"url": valid_image_url_for_message}})
        else:
            error_msg = "MissingImageError: Valid image URL is required for VQA but was not available for the initial call."
            if verbose: print(f"CRITICAL ERROR QID {question_id}: {error_msg}")
            response_data["error"] = error_msg
            response_data["final_answer"] = "[Answer Failed]"
            return response_data

        initial_user_message = {
            "role": "user",
            "content": initial_vlm_message_parts
        }
        
        if verbose: print(f"   Orchestrator: Calling Initial_VLM_Agent...")
        initial_vlm_agent.reset()

        chat_result_initial = await vqa_orchestrator.a_initiate_chat(
            recipient=initial_vlm_agent,
            message=initial_user_message,
            max_turns=20,
            summary_method="last_msg",
            silent=(not verbose)
        )
        initial_answer_text = get_message_content(chat_result_initial.chat_history[-1] if chat_result_initial and chat_result_initial.chat_history else {})
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

    if match_baseline_failed and response_data["error"] is None:
        if verbose: print("   Orchestrator: Analyzing failure...")

        analysis_output = "[Analysis Agent Communication Error]"
        try:
            failure_analysis_agent.update_system_message(FAILURE_ANALYSIS_SYSTEM_PROMPT)
            failure_analysis_user_message_str = ( 
                f"Original Question: '{question}'\n"
                f"Initial VLM Response: '{initial_answer_text}'\n"
                f"Analyze failure. Suggest strategy: 'numeric reattempt needed for: [item]' OR 'general reattempt, focus on: [items]'.")
            if verbose: print(f"   Orchestrator: Calling Failure_Analysis_Agent...")
            failure_analysis_agent.reset()

            chat_result_analysis = await vqa_orchestrator.a_initiate_chat(
                recipient=failure_analysis_agent, message=failure_analysis_user_message_str,
                max_turns=1, summary_method="last_msg", silent=(not verbose)
            )
            analysis_output = get_message_content(chat_result_analysis.chat_history[-1] if chat_result_analysis and chat_result_analysis.chat_history else {})
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
                if active_dataset_name == "vqa-v2": 
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS)
                else: 
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS)
            except Exception as e_sys_msg: print(f"Warning: Error updating system message for reattempt_vlm_agent: {e_sys_msg}")

            numeric_reattempt_user_prompt = (
                f"Original question: '{question}'\n"
                f"Previous Answer: '{initial_answer_text}'\n"
                f"Analysis suggests issues with counting: '{item_to_count}'.\n"
                f"INSTRUCTION: Look carefully. Count step-by-step how many '{item_to_count}' are present. List instances if possible. Respond starting with '[Reattempted Answer]' followed ONLY by the final number/answer."
            )
            numeric_reattempt_message_parts = [{"type": "text", "text": numeric_reattempt_user_prompt}]
            if valid_image_url_for_message:
                numeric_reattempt_message_parts.append({"type": "image_url", "image_url": {"url": valid_image_url_for_message}})
            else: 
                if verbose: print("   WARNING: Valid image URL not available for reattempt.")

            numeric_reattempt_user_message = {
                "role": "user",
                "content": numeric_reattempt_message_parts
            }

            if verbose: print(f"   Orchestrator: Calling Reattempt_VLM_Agent for counting...")
            try:
                reattempt_vlm_agent.reset()
                chat_result_num_reattempt = await vqa_orchestrator.a_initiate_chat(
                    recipient=reattempt_vlm_agent, message=numeric_reattempt_user_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                reattempt_answer_text = get_message_content(chat_result_num_reattempt.chat_history[-1] if chat_result_num_reattempt and chat_result_num_reattempt.chat_history else {})
                if not reattempt_answer_text: reattempt_answer_text = "[Reattempt VLM Agent returned empty message]"
            except Exception as e_agent_call:
                print(f"ERROR during Reattempt_VLM_Agent call for QID {question_id}: {e_agent_call}")
                response_data["error"] = f"{response_data.get('error', '')}; ReattemptAgentCallError: {str(e_agent_call)}".strip("; ")
                reattempt_answer_text = "[Reattempt Agent Call Failed]"

        elif "[Analysis Failed]" not in analysis_output:
            if verbose: print("   Orchestrator: General reattempt. Querying for specific object attributes...")
            textually_specified_objects = analysis_output 
            focus_match = re.search(r"general reattempt, focus on:\s*(.+)", analysis_output, re.IGNORECASE)
            if focus_match: textually_specified_objects = focus_match.group(1).strip()
            
            object_attributes_descriptions = "[Attribute Query Agent Error]"
            try:
                object_attribute_agent.update_system_message(OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS)
                object_attribute_user_prompt = (
                    f"Question: '{question}'\n"
                    f"Focus on: '{textually_specified_objects}'\n"
                    f"Describe these objects and their relevant attributes in detail. If not visible, state so clearly."
                )
                object_attribute_message_parts = [{"type": "text", "text": object_attribute_user_prompt}]
                if valid_image_url_for_message:
                    object_attribute_message_parts.append({"type": "image_url", "image_url": {"url": valid_image_url_for_message}})
                else:
                    if verbose: print("   WARNING: Valid image URL not available for object attribute query.")
                
                object_attribute_user_message = {
                    "role": "user", 
                    "content": object_attribute_message_parts
                }
                
                if verbose: print(f"   Orchestrator: Calling Object_Attribute_Agent...")
                object_attribute_agent.reset()
                chat_result_obj_attr = await vqa_orchestrator.a_initiate_chat(
                    recipient=object_attribute_agent, message=object_attribute_user_message, 
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                object_attributes_descriptions = get_message_content(chat_result_obj_attr.chat_history[-1] if chat_result_obj_attr and chat_result_obj_attr.chat_history else {})
                if not object_attributes_descriptions: object_attributes_descriptions = "[Attribute Query Agent returned empty message]"
            except Exception as e_agent_call:
                print(f"ERROR during Object_Attribute_Agent call for QID {question_id}: {e_agent_call}")
                response_data["error"] = f"{response_data.get('error', '')}; AttributeAgentCallError: {str(e_agent_call)}".strip("; ")
                object_attributes_descriptions = "[Attribute Query Agent Call Failed]"

            response_data["object_attributes_queried"] = object_attributes_descriptions
            if verbose: print(f"   Object_Attribute_Agent response: <<< {object_attributes_descriptions[:150]}... >>>")

            if "[Attribute Query Agent Error]" not in object_attributes_descriptions and "[Attribute Query Agent Call Failed]" not in object_attributes_descriptions:
                if verbose: print("   Orchestrator: Calling Reattempt_VLM_Agent with new context...")
                try:
                    active_dataset_name = dataset_s_config.get("dataset_name", "default")
                    if active_dataset_name == "vqa-v2": 
                        reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS)
                    else: 
                        reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS)
                except Exception as e_sys_msg: print(f"Warning: Error updating system message for reattempt_vlm_agent: {e_sys_msg}")
                
                final_reattempt_user_prompt = (
                    f"Original question: '{question}'\n"
                    f"Previous answer: '{initial_answer_text}'\n"
                    f"Additional context about the image:\n{object_attributes_descriptions}\n"
                    f"INSTRUCTION: Using this new information, answer the question. Start with '[Reattempted Answer]' followed by your answer."
                )
                final_reattempt_message_parts = [{"type": "text", "text": final_reattempt_user_prompt}]
                if valid_image_url_for_message:
                    final_reattempt_message_parts.append({"type": "image_url", "image_url": {"url": valid_image_url_for_message}})
                else:
                    if verbose: print("   WARNING: Valid image URL not available for final reattempt.")
                
                final_reattempt_user_message = {
                    "role": "user",
                    "content": final_reattempt_message_parts
                }
                
                try:
                    reattempt_vlm_agent.reset()
                    chat_result_final_reattempt = await vqa_orchestrator.a_initiate_chat(
                        recipient=reattempt_vlm_agent, message=final_reattempt_user_message,
                        max_turns=1, summary_method="last_msg", silent=(not verbose)
                    )
                    reattempt_answer_text = get_message_content(chat_result_final_reattempt.chat_history[-1] if chat_result_final_reattempt and chat_result_final_reattempt.chat_history else {})
                    if not reattempt_answer_text: reattempt_answer_text = "[Final Reattempt VLM Agent returned empty message]"
                except Exception as e_agent_call:
                    print(f"ERROR during final Reattempt_VLM_Agent call for QID {question_id}: {e_agent_call}")
                    response_data["error"] = f"{response_data.get('error', '')}; FinalReattemptAgentCallError: {str(e_agent_call)}".strip("; ")
                    reattempt_answer_text = "[Final Reattempt Agent Call Failed]"
            else:
                if verbose: print("   Orchestrator: Skipping reattempt due to object attribute query failure.")
                reattempt_answer_text = "[Reattempt Skipped - Object Attribute Query Failed]"
        
        response_data["reattempt_answer"] = reattempt_answer_text
        possible_failure_markers = ["[Agent Communication Error]", "[Reattempt Not Performed or Failed]",
                                    "[Reattempt Skipped", "[Agent Call Failed]", "Agent returned empty message"]
        if not any(marker in reattempt_answer_text for marker in possible_failure_markers) and reattempt_answer_text.strip():
            final_answer_text = reattempt_answer_text
        else:
            final_answer_text = initial_answer_text 
            if verbose: print("   Orchestrator: Reattempt failed or was skipped. Using initial answer.")

    response_data["final_answer"] = final_answer_text
    if verbose: print(f"   Final Answer selected for QID {question_id}: <<< {final_answer_text} >>>")

    # Updated grading section
    if target_answer and str(target_answer).strip() and response_data["error"] is None:
        if verbose: print(f"   Orchestrator: Performing grading against target: '{target_answer}'")
        
        try:
            # Skip creating a new agent and use the existing orchestrator agent
            grades = []
            num_graders = inference_settings.get("num_grader_agents", 1)
            
            for i in range(num_graders):
                grader_id = i
                grading_user_message_str = (
                    f"You are a temporary grader identified as Grader {grader_id}.\n"
                    f"System instruction: {GRADING_SYSTEM_PROMPT_TEMPLATE.format(grader_id=grader_id)}\n\n"
                    f"Question: '{question}'\n"
                    f"Target Answer: '{target_answer}'\n"
                    f"Model's Final Answer: '{final_answer_text}'\n"
                    f"Provide grade."
                )
                
                grade_text = f"[Grader {grader_id}] [ErrorInGrading]"
                
                try:
                    if verbose: print(f"     Orchestrator: Using existing agent for Grader {grader_id}...")
                    # Use the existing failure_analysis_agent which is already properly initialized
                    failure_analysis_agent.reset()
                    grading_chat_result = await vqa_orchestrator.a_initiate_chat(
                        recipient=failure_analysis_agent,
                        message=grading_user_message_str,
                        max_turns=1,
                        summary_method="last_msg",
                        silent=(not verbose)
                    )
                    
                    grade_response = get_message_content(
                        grading_chat_result.chat_history[-1] 
                        if grading_chat_result and grading_chat_result.chat_history 
                        else {}, 
                        grade_text
                    )
                    
                    # Format the response properly
                    if not grade_response.strip().startswith(f"[Grader {grader_id}]"):
                        grade_text = f"[Grader {grader_id}] " + grade_response
                    else:
                        grade_text = grade_response
                        
                except Exception as e_agent_call:
                    print(f"ERROR during grading agent call (Grader {grader_id}) for QID {question_id}: {e_agent_call}")
                    response_data["error"] = f"{response_data.get('error', '')}; GradingAgentError{grader_id}: {str(e_agent_call)}".strip("; ")
                
                grades.append(grade_text)
            
            response_data["grades"] = grades
            if verbose: print(f"   Grades: {grades}")
            
        except Exception as e_grading:
            print(f"ERROR during grading setup for QID {question_id}: {e_grading}")
            response_data["error"] = f"{response_data.get('error', '')}; GradingSetupError: {str(e_grading)}".strip("; ")
            response_data["grades"] = ["[Grading Failed - Configuration Error]"] * inference_settings.get("num_grader_agents", 1)
    
    elif response_data["error"]:
        if verbose: print("   Grading skipped due to earlier pipeline error.")
        response_data["grades"] = ["Grading Skipped - Pipeline Error"] * inference_settings.get("num_grader_agents", 1)
    else:
        if verbose: print("   Grading skipped: No valid target_answer provided.")

    end_time = time.time()
    response_data["processing_time_seconds"] = round(end_time - start_time, 2)
    if verbose: print(f"   Total processing time for QID {question_id}: {response_data['processing_time_seconds']:.2f}s")

    if verbose: print(f"--- VQA Pipeline End for QID: {question_id} ---")
    
    filename_template = inference_settings.get("output_individual_response_filename_template")
    if inference_settings.get("save_individual_responses", False) and filename_template:
        individual_filename = filename_template.format(question_id=question_id)
        write_response_to_jsonl(response_data, individual_filename)
        
    return response_data