import asyncio
import time
import os
import logging
from typing import List, Dict, Any, Union

from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.models import ChatCompletionClient

from ..agents.mixture_agents import VQAWorkerAgent, VQAOrchestratorAgent, UserVQATask, FinalVQAResult
from ..image_utils import process_image_for_vlm_agent
from ..utils import create_error_result_dict
from ..evaluation import perform_direct_accuracy_check


async def run_mixture_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str,
    target_answer: Any,
    question_type: str,
    num_layers: int,
    num_workers_per_layer: int,
    vlm_client: ChatCompletionClient,
    current_config_for_image_processing: dict,
    logger_instance: logging.Logger,
    f1_threshold_setting: float
) -> Dict[str, Any]:
    
    main_logger = logger_instance
    main_logger.info(f"Starting Mixture of Agents VQA Pipeline for QID: {question_id}")

    valid_image_url_for_message: str
    try:
        processed_url = process_image_for_vlm_agent(image_path, current_config_for_image_processing)
        if not isinstance(processed_url, str) or not \
           (processed_url.startswith("data:image/") or processed_url.startswith("http")):
            if not os.path.exists(processed_url):
                 raise ValueError(f"Invalid image URL/data URI or non-existent local path: {str(processed_url)[:100]}")
        valid_image_url_for_message = processed_url
        main_logger.debug(f"QID {question_id}: Image processed. URL/Path: {valid_image_url_for_message[:100]}...")
    except Exception as e_img:
        main_logger.error(f"QID {question_id}: Error processing image '{os.path.basename(image_path)}': {e_img}", exc_info=True)
        error_data = create_error_result_dict(question_id, image_path, question, target_answer,
                                        f"ImageProcessingError: {str(e_img)}",
                                        "ImageError", "mixture_of_agents")
        error_data["question_type"] = question_type
        return error_data

    runtime = SingleThreadedAgentRuntime()
    worker_agent_type_name = f"moa_worker_type_qid_{question_id}"
    
    await VQAWorkerAgent.register(
        runtime, 
        worker_agent_type_name, 
        lambda: VQAWorkerAgent(model_client=vlm_client, agent_name=f"Worker_{worker_agent_type_name}")
    )
    
    orchestrator_agent_type_name = f"moa_orchestrator_type_qid_{question_id}"
    orchestrator_instance_key = f"orchestrator_instance_for_qid_{question_id}"
    await VQAOrchestratorAgent.register(
        runtime,
        orchestrator_agent_type_name,
        lambda: VQAOrchestratorAgent(
            model_client=vlm_client, 
            worker_agent_types=[worker_agent_type_name] * num_workers_per_layer,
            num_layers=num_layers,
            agent_name=f"Orchestrator_{orchestrator_agent_type_name}"
        ),
    )
    main_logger.info(f"QID {question_id}: Runtime & agents registered. NumLayersConfig: {num_layers}, WorkersPerLayerConfig: {num_workers_per_layer}.")
    
    runtime.start()
    initial_user_task = UserVQATask(
        task=question, 
        image_url=valid_image_url_for_message, 
        question_id=question_id,
        target_answer=target_answer,
        question_type=question_type
    )
    
    pipeline_final_result_obj: Union[FinalVQAResult, None] = None
    pipeline_error_str: Union[str, None] = None
    start_time_pipeline = time.time()

    try:
        pipeline_final_result_obj = await runtime.send_message(
            initial_user_task, 
            AgentId(orchestrator_agent_type_name, orchestrator_instance_key)
        )
    except Exception as e_runtime:
        main_logger.error(f"QID {question_id}: Exception during runtime.send_message: {e_runtime}", exc_info=True)
        pipeline_error_str = f"RuntimeSendError: {str(e_runtime)}"
    finally:
        await runtime.stop_when_idle()
        main_logger.info(f"QID {question_id}: Runtime stopped.")
        
    processing_time_total = round(time.time() - start_time_pipeline, 2)

    response_dict: Dict[str, Any]
    if isinstance(pipeline_final_result_obj, FinalVQAResult):
        pipeline_final_result_obj.error = pipeline_final_result_obj.error or pipeline_error_str
        pipeline_final_result_obj.processing_time_seconds = processing_time_total 
        
        response_dict = {
            "question_id": pipeline_final_result_obj.question_id,
            "image_path": image_path, 
            "question": question, 
            "target_answers": target_answer, 
            "question_type": question_type, 
            "final_aggregated_answer": pipeline_final_result_obj.final_aggregated_answer,
            "mixture_all_layers_outputs": pipeline_final_result_obj.all_worker_outputs_by_layer,
            "processing_time_seconds": pipeline_final_result_obj.processing_time_seconds,
            "error": pipeline_final_result_obj.error,
            "direct_accuracy_check": {}, 
            "flow_type_used": "mixture_of_agents",
            "stop_reason": pipeline_final_result_obj.stop_reason
        }
    else:
        main_logger.error(f"QID {question_id}: Pipeline returned unexpected type: {type(pipeline_final_result_obj)}. Error: {pipeline_error_str}")
        response_dict = create_error_result_dict(
            question_id, image_path, question, target_answer,
            pipeline_error_str or "PipelineDidNotReturnFinalVQAResult",
            "PipelineError", "mixture_of_agents" 
        )
        response_dict["question_type"] = question_type 
        response_dict["processing_time_seconds"] = processing_time_total
        if "final_aggregated_answer" not in response_dict:
             response_dict["final_aggregated_answer"] = f"[{response_dict.get('error_type', 'UnknownErrorInPipeline')}]"
        if "direct_accuracy_check" not in response_dict:
            response_dict["direct_accuracy_check"] = {}

    current_final_answer_text = response_dict.get("final_aggregated_answer", "")
    if not isinstance(current_final_answer_text, str):
        current_final_answer_text = str(current_final_answer_text)

    target_for_eval = target_answer
    if isinstance(target_answer, list) and target_answer:
        if isinstance(target_answer[0], dict) and 'answer' in target_answer[0]:
            target_for_eval = target_answer[0]['answer']
        elif isinstance(target_answer[0], str):
            target_for_eval = target_answer[0]
    elif isinstance(target_answer, str):
            target_for_eval = target_answer

    direct_check_results, _, _ = perform_direct_accuracy_check(
        final_answer_text=current_final_answer_text,
        target_answers=target_for_eval,
        question_type=response_dict["question_type"],
        current_error=response_dict.get("error"),
        f1_threshold_other=f1_threshold_setting,
        verbose=current_config_for_image_processing.get("inference_settings",{}).get("verbose", False)
    )
    response_dict["direct_accuracy_check"] = direct_check_results

    main_logger.info(f"Mixture of Agents VQA Pipeline END for QID: {question_id}. Final Answer: '{str(response_dict['final_aggregated_answer'])[:100]}...'. Time: {response_dict['processing_time_seconds']:.2f}s. Error: {response_dict.get('error')}")
    return response_dict