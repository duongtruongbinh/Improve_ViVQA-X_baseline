# main_vqa_flow.py

import asyncio
import re
import json
import os
import time # For basic timing if needed

# Use relative imports assuming a package structure or flat structure
try:
    from config_loader import app_config
    from image_utils import process_image_for_vlm_agent
    from agents import (
        vqa_orchestrator, initial_vlm_agent, failure_analysis_agent,
        object_attribute_agent, reattempt_vlm_agent, llm_client_vllm
    )
    from prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_VQA_V2, INITIAL_VLM_SYSTEM_PROMPT_DEFAULT,
        FAILURE_ANALYSIS_SYSTEM_PROMPT, OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS, REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS,
        GRADING_SYSTEM_PROMPT_TEMPLATE
    )
    # Import AssistantAgent temporarily for grading if not using a dedicated grader agent
    from autogen_agentchat.agents import AssistantAgent
except ImportError as e:
     print(f"Error importing modules in main_vqa_flow.py: {e}. Ensure all required files exist.")
     raise

# Helper to extract text content from AutoGen message more robustly
def get_message_content(message: dict, default_if_none: str = "") -> str:
    if not message or not isinstance(message, dict): return default_if_none
    content = message.get("content", default_if_none)
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", default_if_none).strip()
    # Fallback if content is unexpected type or None
    return default_if_none if content is None else str(content).strip()

# Function to write results to JSONL file
def write_response_to_jsonl(response_data, filename):
    """Appends a dictionary (JSON object) to a JSONL file."""
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create output directory {output_dir}, saving disabled for this item. Error: {e}")
            return # Skip saving if dir cannot be made

    try:
        # Basic sanitation: convert non-serializable types to string
        sanitized_data = json.loads(json.dumps(response_data, default=str))
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sanitized_data) + '\n')
    except TypeError as e_type:
         print(f"Warning: TypeError serializing data for JSONL file {filename}: {e_type}. Skipping save for this item.")
         # print(f"Problematic data structure: {response_data}") # Optional: Debug data
    except Exception as e:
        print(f"Warning: Error writing to JSONL file {filename}: {e}. Skipping save for this item.")


async def run_vqa_pipeline(image_path: str, question: str, question_id: str ="unknown_qid", target_answer: str = None):
    """
    Orchestrates the VQA pipeline for a single image-question pair using AutoGen agents.
    This version DOES NOT USE EXTERNAL TOOLS (GroundingDINO, CLIP-Count).
    Reattempts rely on re-prompting VLMs/LLMs based on user's specified strategy.
    """
    start_time = time.time()
    current_config = app_config
    inference_settings = current_config.get("inference_settings", {})
    dataset_settings = current_config.get("dataset_settings", {})
    verbose = inference_settings.get("verbose", True)

    if verbose:
        print(f"--- VQA Pipeline Start for QID: {question_id} ---")
        print(f"  Image: {os.path.basename(image_path)}")
        print(f"  Question: {question}")

    response_data = {
        "question_id": str(question_id), "image_path": image_path, "question": question,
        "target_answer": target_answer, "initial_answer": "", "final_answer": "",
        "match_baseline_failed": False, "is_numeric_reattempt": False,
        "analysis_output": "", "object_attributes_queried": "", "reattempt_answer": "",
        "grades": [], "processing_time_seconds": 0.0, "error": None
    }

    # --- Ensure Orchestrator and Clients are Ready ---
    if vqa_orchestrator is None or initial_vlm_agent is None or failure_analysis_agent is None or object_attribute_agent is None or reattempt_vlm_agent is None:
        response_data["error"] = "AgentInitializationError"
        response_data["final_answer"] = "[Answer Failed]"
        print(f"ERROR: One or more core agents are not initialized for QID {question_id}.")
        # Attempt to save error state
        if inference_settings.get("save_output_response"):
            write_response_to_jsonl(response_data, inference_settings.get("output_response_filename"))
        return response_data
    if vlm_client_vllm is None or llm_client_vllm is None:
         response_data["error"] = "ClientInitializationError"
         response_data["final_answer"] = "[Answer Failed]"
         print(f"ERROR: VLM or LLM client is not initialized for QID {question_id}.")
         if inference_settings.get("save_output_response"):
             write_response_to_jsonl(response_data, inference_settings.get("output_response_filename"))
         return response_data

    # --- Step 1: Prepare Image ---
    try:
        base64_image_url = process_image_for_vlm_agent(image_path, current_config)
        if not base64_image_url: raise ValueError("Image processing returned empty result.")
    except Exception as e_img:
        if verbose: print(f"ERROR processing image {image_path}: {e_img}")
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        response_data["final_answer"] = "[Answer Failed]"
        # Attempt to save error state
        if inference_settings.get("save_output_response"):
            write_response_to_jsonl(response_data, inference_settings.get("output_response_filename"))
        return response_data

    # --- Step 2: Initial VLM Attempt ---
    active_dataset_name = dataset_settings.get("dataset_name", "default")
    try:
        if active_dataset_name == "vqa-v2":
            initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_VQA_V2)
        else:
            initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_DEFAULT)
    except Exception as e_sys_msg:
         print(f"Error updating system message for initial_vlm_agent: {e_sys_msg}")
         # Continue with default or handle error
    
    initial_vlm_message_content = [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
    initial_user_message_to_agent = {"role": "user", "content": initial_vlm_message_content}

    if verbose: print(f"Orchestrator: Calling Initial_VLM_Agent...")
    
    initial_answer_text = "[Agent Communication Error]"
    try:
        # Clear chat history for the recipient agent before initiating for a clean slate
        initial_vlm_agent.clear_history()
        vqa_orchestrator.clear_history() # Clear orchestrator history relevant to this agent pair? Maybe not needed.

        chat_result_initial = await vqa_orchestrator.a_initiate_chat(
            recipient=initial_vlm_agent,
            message=initial_user_message_to_agent,
            max_turns=1,
            summary_method="last_msg",
            silent=(not verbose) # Suppress AutoGen's internal print statements if not verbose
        )
        initial_answer_text = get_message_content(chat_result_initial.last_message_dict if chat_result_initial else {})
    except Exception as e_agent_call:
        print(f"ERROR during Initial_VLM_Agent call for QID {question_id}: {e_agent_call}")
        response_data["error"] = f"InitialAgentCallError: {str(e_agent_call)}"
        initial_answer_text = "[Agent Call Failed]" # Ensure it reflects failure

    response_data["initial_answer"] = initial_answer_text
    final_answer_text = initial_answer_text
    if verbose: print(f"Initial_VLM_Agent response: <<< {initial_answer_text} >>>")

    # --- Step 3: Analyze Failure ---
    # Determine if reattempt is needed based on response markers
    is_direct_failure_marker = (
        "[Answer Failed]" in initial_answer_text or
        "sorry" in initial_answer_text.lower() or
        "unable to answer" in initial_answer_text.lower() or
        "cannot answer" in initial_answer_text.lower() or
        not initial_answer_text.strip() or
        initial_answer_text == "[Agent Communication Error]" or
        initial_answer_text == "[Agent Call Failed]"
    )
    # Check for problematic numeric markers from VQA-v2 prompt style
    is_problematic_numeric_answer = (
        ("[Non-zero Numeric Answer]" in initial_answer_text and "[Answer Failed]" in initial_answer_text) or
        ("[Zero Numeric Answer]" in initial_answer_text and "[Answer Failed]" in initial_answer_text)
    )

    match_baseline_failed = is_direct_failure_marker or is_problematic_numeric_answer
    if inference_settings.get("force_multi_agents", False):
        match_baseline_failed = True
        if verbose: print("Note: force_multi_agents=True, proceeding to reattempt.")

    response_data["match_baseline_failed"] = match_baseline_failed
    reattempt_performed = False # Flag to know if reattempt block was entered

    if match_baseline_failed:
        reattempt_performed = True
        if verbose: print("Orchestrator: Initial answer failed or flagged. Analyzing failure...")

        # --- Step 4: Call Failure_Analysis_Agent ---
        try:
            failure_analysis_agent.update_system_message(FAILURE_ANALYSIS_SYSTEM_PROMPT)
        except Exception as e_sys_msg:
            print(f"Error updating system message for failure_analysis_agent: {e_sys_msg}")

        failure_analysis_user_message = (
            f"Original Question: '{question}'\n"
            f"Initial VLM Response: '{initial_answer_text}'\n"
            f"Analyze this failure/response. Determine reattempt strategy (numeric or general focus)."
        )
        if verbose: print(f"Orchestrator: Calling Failure_Analysis_Agent...")

        analysis_output = "[Analysis Agent Communication Error]"
        try:
            failure_analysis_agent.clear_history()
            vqa_orchestrator.clear_history()
            chat_result_analysis = await vqa_orchestrator.a_initiate_chat(
                recipient=failure_analysis_agent, message=failure_analysis_user_message,
                max_turns=1, summary_method="last_msg", silent=(not verbose)
            )
            analysis_output = get_message_content(chat_result_analysis.last_message_dict if chat_result_analysis else {})
        except Exception as e_agent_call:
            print(f"Error during Failure_Analysis_Agent call for QID {question_id}: {e_agent_call}")
            if response_data["error"] is None: response_data["error"] = ""
            response_data["error"] += f"; FailureAnalysisAgentError: {str(e_agent_call)}"
            analysis_output = "[Analysis Failed]" # Ensure failure is noted

        response_data["analysis_output"] = analysis_output
        if verbose: print(f"Failure_Analysis_Agent response: <<< {analysis_output} >>>")

        # Determine reattempt strategy (NO TOOLS)
        is_numeric_reattempt = "numeric reattempt needed" in analysis_output.lower()
        response_data["is_numeric_reattempt"] = is_numeric_reattempt
        reattempt_answer_text = "[Reattempt Not Performed or Failed]"

        if is_numeric_reattempt:
            # --- Step 5a: Numeric Reattempt (No Tools) ---
            if verbose: print("Orchestrator: Numeric reattempt indicated. Re-prompting VLM for careful counting...")
            item_to_count = "the relevant objects" # Default
            item_match = re.search(r"numeric reattempt needed for:\s*(.+)", analysis_output, re.IGNORECASE)
            if item_match: item_to_count = item_match.group(1).strip()
            
            try: # Update system message for reattempt agent
                if active_dataset_name == "vqa-v2":
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS)
                else:
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS)
            except Exception as e_sys_msg: print(f"Error updating system message for reattempt_vlm_agent: {e_sys_msg}")

            numeric_reattempt_user_prompt = (
                f"Original question: '{question}'\n"
                f"Previous Answer: '{initial_answer_text}'\n"
                f"Analysis indicates issues with counting: '{item_to_count}'.\n"
                f"INSTRUCTION: Examine the image closely. Count the number of '{item_to_count}'. Provide step-by-step reasoning for your count. "
                f"List each instance if possible before the final number. Respond starting with '[Reattempted Answer]' followed by the final numeric answer."
            )
            numeric_reattempt_message_content = [{"type": "text", "text": numeric_reattempt_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
            numeric_reattempt_message = {"role": "user", "content": numeric_reattempt_message_content}

            try:
                reattempt_vlm_agent.clear_history()
                vqa_orchestrator.clear_history()
                chat_result_num_reattempt = await vqa_orchestrator.a_initiate_chat(
                    recipient=reattempt_vlm_agent, message=numeric_reattempt_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                reattempt_answer_text = get_message_content(chat_result_num_reattempt.last_message_dict if chat_result_num_reattempt else {})
            except Exception as e_agent_call:
                 print(f"Error during numeric reattempt agent call for QID {question_id}: {e_agent_call}")
                 if response_data["error"] is None: response_data["error"] = ""
                 response_data["error"] += f"; NumericReattemptAgentError: {str(e_agent_call)}"

        elif "[Analysis Failed]" not in analysis_output: # General reattempt only if analysis didn't fail
            # --- Step 5b: General Reattempt (No Tools) ---
            if verbose: print("Orchestrator: General reattempt indicated. Step 1: Getting textual object descriptions...")
            textually_specified_objects = analysis_output # Default to using raw analysis output if parsing fails
            focus_match = re.search(r"general reattempt, focus on:\s*(.+)", analysis_output, re.IGNORECASE)
            if focus_match: textually_specified_objects = focus_match.group(1).strip()
            
            # Step 5b-1: Get textual descriptions using Object_Attribute_Agent
            try:
                object_attribute_agent.update_system_message(OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS)
            except Exception as e_sys_msg: print(f"Error updating system message for object_attribute_agent: {e_sys_msg}")
                
            attribute_query_user_prompt = (
                f"Original VQA Question was: '{question}'.\n"
                f"Analysis suggests focusing on these items in the image: '{textually_specified_objects}'.\n"
                f"Please describe these items visually based on the image."
            )
            attribute_query_content = [{"type": "text", "text": attribute_query_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
            attribute_query_message = {"role": "user", "content": attribute_query_content}
            
            object_attributes_descriptions = "[Attribute Query Agent Error]"
            try:
                if verbose: print(f"Orchestrator: Calling Object_Attribute_Agent for descriptions...")
                object_attribute_agent.clear_history()
                vqa_orchestrator.clear_history()
                chat_result_attr = await vqa_orchestrator.a_initiate_chat(
                    recipient=object_attribute_agent, message=attribute_query_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                object_attributes_descriptions = get_message_content(chat_result_attr.last_message_dict if chat_result_attr else {})
            except Exception as e_agent_call:
                 print(f"Error during object attribute agent call for QID {question_id}: {e_agent_call}")
                 if response_data["error"] is None: response_data["error"] = ""
                 response_data["error"] += f"; AttributeQueryAgentError: {str(e_agent_call)}"

            response_data["object_attributes_queried"] = object_attributes_descriptions
            if verbose: print(f"Object_Attribute_Agent response: <<< {object_attributes_descriptions} >>>")

            # Step 5b-2: Reattempt VLM call with these textual descriptions
            try: # Update system message for reattempt agent
                if active_dataset_name == "vqa-v2":
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS)
                else:
                    reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS)
            except Exception as e_sys_msg: print(f"Error updating system message for reattempt_vlm_agent: {e_sys_msg}")

            general_reattempt_user_prompt = (
                f"Original question: '{question}'.\n"
                f"Previous Answer: '{initial_answer_text}'.\n"
                f"Analysis suggested focus on: '{textually_specified_objects}'.\n"
                f"Descriptions of these items: '{object_attributes_descriptions}'.\n"
                f"INSTRUCTION: Using the image and ALL the text context above, provide an improved answer to the original question. Reason step-by-step first, then give the final answer starting with '[Reattempted Answer]'."
            )
            general_reattempt_content = [{"type": "text", "text": general_reattempt_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image_url}}]
            general_reattempt_message = {"role": "user", "content": general_reattempt_content}

            try:
                if verbose: print(f"Orchestrator: Calling Reattempt_VLM_Agent with new textual context...")
                reattempt_vlm_agent.clear_history()
                vqa_orchestrator.clear_history()
                chat_result_gen_reattempt = await vqa_orchestrator.a_initiate_chat(
                    recipient=reattempt_vlm_agent, message=general_reattempt_message,
                    max_turns=1, summary_method="last_msg", silent=(not verbose)
                )
                reattempt_answer_text = get_message_content(chat_result_gen_reattempt.last_message_dict if chat_result_gen_reattempt else {})
            except Exception as e_agent_call:
                 print(f"Error during general reattempt agent call for QID {question_id}: {e_agent_call}")
                 if response_data["error"] is None: response_data["error"] = ""
                 response_data["error"] += f"; GeneralReattemptAgentError: {str(e_agent_call)}"
        
        # --- Update final answer based on reattempt ---
        # Only update if reattempt didn't obviously fail
        if "[Agent Communication Error]" not in reattempt_answer_text and \
           "[Reattempt Not Performed or Failed]" not in reattempt_answer_text and \
           reattempt_answer_text.strip():
            final_answer_text = reattempt_answer_text
        response_data["reattempt_answer"] = reattempt_answer_text # Log the attempt regardless

    # --- Final Answer Selection ---
    response_data["final_answer"] = final_answer_text
    if verbose: print(f"Final Answer selected for QID {question_id}: <<< {final_answer_text} >>>")

    # --- Step 7: Grading (using LLM) ---
    if target_answer and str(target_answer).strip():
        if verbose: print(f"Orchestrator: Performing grading against target: '{target_answer}'")
        
        # Use a temporary agent for grading to avoid polluting main agents' history/prompts
        # Ensure llm_client_vllm is functional before proceeding
        if llm_client_vllm:
            temp_grader_agent = AssistantAgent( name="Temp_Grader", model_client=llm_client_vllm )
            
            for i in range(3): # Simulate 3 graders
                grader_id = i
                temp_grader_agent.update_system_message(GRADING_SYSTEM_PROMPT_TEMPLATE.format(grader_id=grader_id))
                
                grading_user_message = (
                    f"Question: '{question}'\n"
                    f"Target Answer: '{target_answer}'\n"
                    f"Model's Final Answer (after potential reattempt): '{final_answer_text}'\n"
                    f"Please provide your grade."
                )
                
                grade_text = f"[Grader {grader_id}] [ErrorInGrading]"
                try:
                    if verbose: print(f"Orchestrator: Calling Temp_Grader_Agent (Grader {grader_id})...")
                    temp_grader_agent.clear_history()
                    vqa_orchestrator.clear_history()
                    grading_chat_result = await vqa_orchestrator.a_initiate_chat(
                        recipient=temp_grader_agent, message=grading_user_message,
                        max_turns=1, summary_method="last_msg", silent=(not verbose)
                    )
                    grade_text = get_message_content(grading_chat_result.last_message_dict if grading_chat_result else {}, grade_text)
                except Exception as e_agent_call:
                    print(f"Error during grading agent call (Grader {grader_id}) for QID {question_id}: {e_agent_call}")
                    if response_data["error"] is None: response_data["error"] = ""
                    response_data["error"] += f"; GradingAgentError{grader_id}: {str(e_agent_call)}"
                
                response_data["grades"].append(grade_text)
            if verbose: print(f"Grading results: {response_data['grades']}")
        else:
            if verbose: print("Grading skipped: LLM client not available.")
            response_data["grades"] = ["Grading Skipped - LLM Client Error"] * 3
    else:
        if verbose: print("Grading skipped: No target_answer provided.")

    # --- Step 8: Finalize and Save ---
    end_time = time.time()
    response_data["processing_time_seconds"] = round(end_time - start_time, 2)
    if verbose: print(f"Total processing time for QID {question_id}: {response_data['processing_time_seconds']:.2f}s")

    if inference_settings.get("save_output_response"):
        output_file = inference_settings.get("output_response_filename", "outputs/autogen_vqa_default.jsonl")
        write_response_to_jsonl(response_data, output_file)

    if verbose: print(f"--- VQA Pipeline End for QID: {question_id} ---")
    return response_data