# debate_flow.py
import json
import time
import os
import logging
from collections import Counter
from typing import List, Dict, Any
import re

from ..config_loader import app_config
from ..image_utils import process_image_for_vlm_agent
from ..evaluation import perform_direct_accuracy_check
from ..agents.debate_agents import VQADebateSolverAgent

try:
    from ..vllm_clients import llm_config_vlm
except ImportError:
    llm_config_vlm = None

def parse_solver_output_basic(raw_output: str) -> Dict[str, str | None]:
    answer = None
    raw_output_cleaned = raw_output.strip()
    
    if raw_output_cleaned.lower().startswith("answer:"):
        answer = raw_output_cleaned.split(":",1)[-1].strip()
    elif "\n" in raw_output_cleaned:
        potential_answer = raw_output_cleaned.split("\n")[0].strip()
        if potential_answer.lower().startswith("answer:"):
            answer = potential_answer.split(":",1)[-1].strip()
        else:
            answer = potential_answer
    else:
        answer = raw_output_cleaned
        
    return {"answer": answer, "reasoning": None}


async def run_simplified_debate_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str = "unknown_qid",
    target_answer: any = None,
    question_type: str = "other",
    logger_instance: logging.Logger = None
) -> dict:
    start_time = time.time()
    current_config = app_config
    inference_settings = current_config.get("inference_settings", {})
    vqa_debate_settings = current_config.get("vqa_debate_settings", {})
    
    logger = logger_instance if logger_instance else logging.getLogger("SimplifiedDebateFlowLogger")
    if not logger_instance and not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if inference_settings.get("verbose", False) else logging.INFO)

    _effective_solver_llm_config = False
    if llm_config_vlm is None:
        logger.critical(f"CRITICAL_ERROR (Debate Flow QID {question_id}): llm_config_vlm from vllm_clients is None or not imported. Solvers will lack LLM capabilities.")
    elif not isinstance(llm_config_vlm, dict) or "config_list" not in llm_config_vlm or not llm_config_vlm.get("config_list"):
        logger.critical(f"CRITICAL_ERROR (Debate Flow QID {question_id}): llm_config_vlm is not in the expected format (must be a dict with a non-empty 'config_list'). Solvers will lack LLM capabilities.")
    elif not llm_config_vlm["config_list"][0].get("base_url") and not str(llm_config_vlm["config_list"][0].get("model","")).startswith("gpt-"):
        logger.critical(f"CRITICAL_ERROR (Debate Flow QID {question_id}): VLM 'base_url' is missing in imported llm_config_vlm for a non-GPT model. Solvers will lack LLM capabilities.")
    else:
        _effective_solver_llm_config = llm_config_vlm.copy()
        _effective_solver_llm_config["cache_seed"] = inference_settings.get("random_seed", _effective_solver_llm_config.get("cache_seed"))
        _effective_solver_llm_config["temperature"] = vqa_debate_settings.get("solver_temperature", 0.0)

    verbose = inference_settings.get("verbose", False)
    num_solvers = vqa_debate_settings.get("num_solvers", 3)
    max_debate_rounds = vqa_debate_settings.get("max_rounds", 1)
    f1_threshold_setting = inference_settings.get("f1_threshold_for_other_type", 0.5)

    response_data = {
        "question_id": str(question_id),
        "image_path": image_path,
        "question": question,
        "target_answers": target_answer,
        "question_type": question_type,
        "debate_history_per_round": [],
        "final_solver_outputs": [],
        "final_aggregated_answer": "[Pipeline Incomplete]",
        "final_confidence": None,
        "processing_time_seconds": 0.0,
        "error": None,
        "direct_accuracy_check": {},
        "num_debate_rounds_completed": 0,
        "stop_reason": "Unknown"
    }

    valid_image_url_for_message = None
    try:
        processed_image_url = process_image_for_vlm_agent(image_path, current_config)
        if not isinstance(processed_image_url, str):
            raise ValueError(f"Image processing returned non-string: {type(processed_image_url)}")
        if not (processed_image_url.startswith("data:image/") or \
                processed_image_url.startswith("http://") or \
                processed_image_url.startswith("https://")):
            raise ValueError(f"Image processing returned invalid URL scheme: {str(processed_image_url)[:100]}")
        valid_image_url_for_message = processed_image_url
    except Exception as e_img:
        logger.error(f"ERROR (Debate Flow QID {question_id}) processing image '{os.path.basename(image_path)}': {e_img}", exc_info=verbose)
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        response_data["stop_reason"] = "ImageProcessingError"
        if target_answer is not None:
            current_target_for_eval = target_answer[0]['answer'] if isinstance(target_answer, list) and target_answer and isinstance(target_answer[0], dict) else target_answer
            direct_check_res, _, _ = perform_direct_accuracy_check(
                final_answer_text=response_data["final_aggregated_answer"],
                target_answer=current_target_for_eval,
                question_type=question_type,
                f1_threshold_other=f1_threshold_setting
            )
            response_data["direct_accuracy_check"] = direct_check_res
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data

    solver_agents: List[VQADebateSolverAgent] = []
    try:
        if not _effective_solver_llm_config:
            raise ValueError("Solver LLM configuration is invalid or missing.")
        for i in range(num_solvers):
            solver_name = f"VQASolver_{i+1}_QID_{question_id}"
            agent = VQADebateSolverAgent(name=solver_name, llm_config=_effective_solver_llm_config)
            solver_agents.append(agent)
    except Exception as e_agent_init:
        logger.critical(f"Failed to initialize VQADebateSolverAgents for QID {question_id}: {e_agent_init}", exc_info=verbose)
        response_data["error"] = f"AgentInitializationError: {str(e_agent_init)}"
        response_data["stop_reason"] = "AgentInitError"
        if target_answer is not None:
            current_target_for_eval = target_answer[0]['answer'] if isinstance(target_answer, list) and target_answer and isinstance(target_answer[0], dict) else target_answer
            direct_check_res, _, _ = perform_direct_accuracy_check(
                response_data["final_aggregated_answer"], current_target_for_eval, question_type,
                f1_threshold_other=f1_threshold_setting
            )
            response_data["direct_accuracy_check"] = direct_check_res
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data
        
    if not solver_agents:
        logger.error(f"No solver agents were initialized for QID {question_id}. Cannot proceed.")
        response_data["error"] = response_data.get("error", "No solver agents initialized.")
        response_data["stop_reason"] = response_data.get("stop_reason", "NoAgents")
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data
            
    logger.info(f"{len(solver_agents)} VQADebateSolverAgents initialized for QID {question_id}.")

    all_rounds_outputs: List[List[Dict[str, Any]]] = []
    previous_round_outputs_for_next_round: List[Dict[str, Any]] = [] 

    for current_round_idx in range(max_debate_rounds):
        response_data["num_debate_rounds_completed"] = current_round_idx + 1
        current_round_display = current_round_idx + 1
        logger.info(f"\n--- Debate Round {current_round_display} (QID: {question_id}) ---")
        
        current_round_solver_outputs_collected: List[Dict[str, Any]] = []
        
        for agent_idx, solver_agent in enumerate(solver_agents):
            solver_agent.reset() 
            
            system_prompt_for_turn = solver_agent.render_system_message(
                question=question,
                current_round=current_round_display,
                max_rounds=max_debate_rounds,
                neighbor_responses=previous_round_outputs_for_next_round
            )
            
            solver_input_payload = [
                {"type": "image_url", "image_url": {"url": valid_image_url_for_message}},
                {"type": "text", "text": question}
            ]
            
            messages_for_solver_turn = [
                {"role": "system", "content": system_prompt_for_turn},
                {"role": "user", "content": solver_input_payload}
            ]

            logger.debug(f" {solver_agent.name} (Round {current_round_display}) System Prompt Preview: {system_prompt_for_turn[:400]}...")
            logger.info(f" {solver_agent.name} (Round {current_round_display}): Calling VLM...")
            
            raw_solver_output_str = ""
            parsed_output = {"answer": f"[SolverError_R{current_round_display}]", "reasoning": None}
            current_solver_confidence = 0.0 
            
            try:
                if not _effective_solver_llm_config:
                    raise RuntimeError("Cannot call VLM: Solver LLM configuration is invalid.")

                chat_res_solver_msg_obj = await solver_agent.a_generate_reply(
                    messages=messages_for_solver_turn,
                    sender=None, 
                )
                
                if chat_res_solver_msg_obj and isinstance(chat_res_solver_msg_obj, dict):
                    raw_solver_output_str = chat_res_solver_msg_obj.get("content","")
                elif isinstance(chat_res_solver_msg_obj, str):
                    raw_solver_output_str = chat_res_solver_msg_obj
                else:
                     raw_solver_output_str = ""

                if not raw_solver_output_str:
                    logger.warning(f" {solver_agent.name} (Round {current_round_display}) returned empty content for QID {question_id}.")
                    parsed_output["answer"] = "[SolverEmptyOutput]"
                else:
                    parsed_output = parse_solver_output_basic(raw_solver_output_str) 
                    if not parsed_output.get("answer"):
                        parsed_output["answer"] = "[SolverEmptyAnswerAfterParsing]"
                    else:
                        current_solver_confidence = 0.95

                logger.info(f" {solver_agent.name} (R{current_round_display}): Ans='{str(parsed_output['answer'])[:70]}...', Conf={current_solver_confidence:.2f}, Rsn='{str(parsed_output.get('reasoning'))[:50] if parsed_output.get('reasoning') else 'N/A'}'")
            
            except Exception as e_solve:
                logger.error(f" ERROR (Debate Flow QID {question_id}) in {solver_agent.name} (Round {current_round_display}): {e_solve}", exc_info=verbose)
            
            current_round_solver_outputs_collected.append({
                "solver_name": solver_agent.name,
                "round_num": current_round_display, 
                "answer": parsed_output["answer"],
                "reasoning": parsed_output.get("reasoning"),
                "confidence": current_solver_confidence, 
                "raw_output": raw_solver_output_str 
            })
            
        all_rounds_outputs.append(current_round_solver_outputs_collected)
        previous_round_outputs_for_next_round = current_round_solver_outputs_collected 
        
        if not any(out.get('answer') and not str(out.get('answer', '')).startswith(("[SolverError", "[SolverEmpty", "[SolverEmptyAnswerAfterParsing")) for out in current_round_solver_outputs_collected):
            logger.error(f"All solvers failed or produced no valid answer in round {current_round_display} for QID {question_id}. Stopping debate.")
            response_data["error"] = response_data.get("error", f"AllSolversFailed_R{current_round_display}")
            response_data["stop_reason"] = f"AllSolversFailed_R{current_round_display}"
            break

    response_data["debate_history_per_round"] = all_rounds_outputs
    
    if all_rounds_outputs and all_rounds_outputs[-1]:
        final_solver_outputs_from_last_round = all_rounds_outputs[-1]
        response_data["final_solver_outputs"] = final_solver_outputs_from_last_round
        
        answers_for_vote = [
            sol_out.get("answer", "") for sol_out in final_solver_outputs_from_last_round 
            if sol_out.get("answer") and isinstance(sol_out.get("answer"), str) and \
               not str(sol_out.get("answer")).startswith(("[SolverError", "[SolverEmpty", "[SolverEmptyAnswerAfterParsing"))
        ]
        if answers_for_vote:
            vote_counts = Counter(answers_for_vote)
            if vote_counts:
                majority_answer, majority_count = vote_counts.most_common(1)[0]
                response_data["final_aggregated_answer"] = majority_answer
                
                confidences_for_majority_answer = [
                    s_out.get("confidence", 0.0) for s_out in final_solver_outputs_from_last_round
                    if s_out.get("answer") == majority_answer and isinstance(s_out.get("confidence"), (float, int))
                ]
                if confidences_for_majority_answer:
                    response_data["final_confidence"] = sum(confidences_for_majority_answer) / len(confidences_for_majority_answer)
                else:
                    response_data["final_confidence"] = 0.0

                if response_data["stop_reason"] == "Unknown":
                    response_data["stop_reason"] = "CompletedAfterRounds_MajorityVote"
                logger.info(f"Debate for QID {question_id} concluded. Majority Answer: '{str(majority_answer)[:100]}...' with {majority_count} votes.")
            else:
                response_data["final_aggregated_answer"] = "[NoValidAnswersForVote]"
                if response_data["stop_reason"] == "Unknown": response_data["stop_reason"] = "NoValidFinalAnswers_NoVoteCounts"
                if not response_data["error"]: response_data["error"] = "AggregationError: No valid answers from solvers for majority vote."
        else:
            response_data["final_aggregated_answer"] = "[NoValidAnswersForVote]"
            if response_data["stop_reason"] == "Unknown": response_data["stop_reason"] = "NoValidFinalAnswers_NoAnswersForVote"
            if not response_data["error"]: response_data["error"] = "AggregationError: No valid answers from solvers for vote."
    elif not response_data.get("error"):
        if response_data["num_debate_rounds_completed"] > 0 :
             response_data["final_aggregated_answer"] = "[NoDebateOutputsFromLastRoundOrNoRoundsCompleted]"
             if response_data["stop_reason"] == "Unknown": response_data["stop_reason"] = "NoOutputsOrNoRoundsCompleted"
             if not response_data["error"]: response_data["error"] = "PipelineError: No debate outputs from final round or no rounds completed."
        else: 
             response_data["final_aggregated_answer"] = "[NoDebateRoundsRun]"
             if response_data["stop_reason"] == "Unknown": response_data["stop_reason"] = "NoRoundsRun"
             if not response_data["error"]: response_data["error"] = "PipelineError: No debate rounds were configured or run."

    if target_answer is not None and response_data.get("stop_reason") != "ImageProcessingError":
        actual_target_for_eval = None
        if isinstance(target_answer, list) and target_answer:
            if isinstance(target_answer[0], dict) and 'answer' in target_answer[0]:
                actual_target_for_eval = target_answer[0]['answer']
            elif isinstance(target_answer[0], str):
                 actual_target_for_eval = target_answer[0]
        elif isinstance(target_answer, str):
            actual_target_for_eval = target_answer

        if actual_target_for_eval is not None:
            direct_check_results, _, _ = perform_direct_accuracy_check(
                final_answer_text=response_data["final_aggregated_answer"],
                target_answers=actual_target_for_eval,
                question_type=response_data["question_type"],
                f1_threshold_other=f1_threshold_setting
            )
            response_data["direct_accuracy_check"] = direct_check_results
        else:
            logger.warning(f"QID {question_id}: Could not determine a single target answer string for direct accuracy check from target_answers: {target_answer}")
            response_data["direct_accuracy_check"] = {"notes": "Could not determine single target answer for eval."}

    end_time = time.time()
    response_data["processing_time_seconds"] = round(end_time - start_time, 2)

    logger.info(f"--- Simplified Debate VQA Pipeline END for QID: {question_id}, Final Aggregated Answer: '{str(response_data['final_aggregated_answer'])[:100]}...', Time: {response_data['processing_time_seconds']:.2f}s, Stop Reason: {response_data['stop_reason']} ---")
    return response_data