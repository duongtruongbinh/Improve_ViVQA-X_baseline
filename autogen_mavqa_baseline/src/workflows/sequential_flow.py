# sequential_flow.py
import asyncio
import time
import os
import logging
from typing import List, Dict, Any, Tuple

from autogen_core import AgentId, SingleThreadedAgentRuntime, TopicId
from autogen_core.models import ChatCompletionClient

from ..image_utils import process_image_for_vlm_agent
from ..evaluation import perform_direct_accuracy_check
from ..agents.sequential_agents import (
    VQATaskRelayMessage,
    VQAImageContextualizerAgent,
    VQAQuestionAnswererAgent,
    VQAAnswerFormatterAgent,
    VQAResultCollectorAgent,
    IMAGE_CONTEXTUALIZER_TOPIC,
    QUESTION_ANSWERER_TOPIC,
    ANSWER_FORMATTER_TOPIC,
    RESULT_COLLECTOR_TOPIC
)
from ..utils import create_error_result_dict # For fallback


async def run_vqa_sequential_workflow(
    tasks: List[Dict[str, Any]],
    vlm_model_client: ChatCompletionClient,
    llm_model_client: ChatCompletionClient,
    current_config: dict, # For image processing and eval settings
    logger_instance: logging.Logger,
    f1_threshold_setting: float
) -> List[Dict[str, Any]]:

    main_logger = logger_instance
    main_logger.info(f"Starting Sequential VQA Workflow for {len(tasks)} tasks.")

    runtime = SingleThreadedAgentRuntime()

    # Shared dictionaries for results and completion events
    completion_events: Dict[str, asyncio.Event] = {}
    final_results_store: Dict[str, Dict[str, Any]] = {}

    # Register agents
    # Agent types are unique strings that identify the agent class
    # contextualizer_agent_type = "VQAImageContextualizerAgent_Type"
    # qa_agent_type = "VQAQuestionAnswererAgent_Type"
    # formatter_agent_type = "VQAAnswerFormatterAgent_Type"
    # collector_agent_type = "VQAResultCollectorAgent_Type"

    # await VQAImageContextualizerAgent.register(
    #     runtime, type=contextualizer_agent_type,
    #     factory=lambda: VQAImageContextualizerAgent(model_client=vlm_model_client)
    # )
    # await VQAQuestionAnswererAgent.register(
    #     runtime, type=qa_agent_type,
    #     factory=lambda: VQAQuestionAnswererAgent(model_client=llm_model_client)
    # )
    # await VQAAnswerFormatterAgent.register(
    #     runtime, type=formatter_agent_type,
    #     factory=lambda: VQAAnswerFormatterAgent(model_client=llm_model_client)
    # )
    # await VQAResultCollectorAgent.register(
    #     runtime, type=collector_agent_type,
    #     factory=lambda: VQAResultCollectorAgent(
    #         completion_events=completion_events,
    #         final_results_store=final_results_store
    #     )
    # )
    # main_logger.info("Sequential VQA agents registered with runtime.")

    # Subscribe agents to their respective topics (using the topic *they listen to*)
    # The agent instances are created dynamically by the runtime when a message is published to their type.
    # Here, we need to ensure the runtime knows an agent of a certain *type* should listen to a certain *topic*.
    # The @type_subscription decorator handles this if topic_type matches the agent's registered type.
    # Let's ensure our topic constants match the agent types used for subscription.
    # The factory creates an instance, and @type_subscription on the class makes it listen.
    # So, when we publish to IMAGE_CONTEXTUALIZER_TOPIC, an instance of VQAImageContextualizerAgent
    # (which is registered with type=contextualizer_agent_type and subscribes via decorator to IMAGE_CONTEXTUALIZER_TOPIC)
    # should pick it up, *if* IMAGE_CONTEXTUALIZER_TOPIC is used as the registration type.
    # Or, more simply, the source of the AgentId during registration needs to match the subscription.

    # Let's re-think registration and topics.
    # When an agent is registered, it gets an AgentId(type, key).
    # The @type_subscription makes an agent *instance* listen to a topic *if its own registered type* matches the topic_type in the decorator.
    # So, the `type` parameter in `register` is key.

    # Revised agent registration matching topic names for clarity with @type_subscription
    # await runtime.delete_agent(AgentId(contextualizer_agent_type, "default")) # clean up if re-running cell
    # await runtime.delete_agent(AgentId(qa_agent_type, "default"))
    # await runtime.delete_agent(AgentId(formatter_agent_type, "default"))
    # await runtime.delete_agent(AgentId(collector_agent_type, "default"))


    await VQAImageContextualizerAgent.register(
        runtime, type=IMAGE_CONTEXTUALIZER_TOPIC, # Agent of this type will listen to this topic
        factory=lambda: VQAImageContextualizerAgent(model_client=vlm_model_client)
    )
    await VQAQuestionAnswererAgent.register(
        runtime, type=QUESTION_ANSWERER_TOPIC,
        factory=lambda: VQAQuestionAnswererAgent(model_client=llm_model_client)
    )
    await VQAAnswerFormatterAgent.register(
        runtime, type=ANSWER_FORMATTER_TOPIC,
        factory=lambda: VQAAnswerFormatterAgent(model_client=llm_model_client)
    )
    await VQAResultCollectorAgent.register(
        runtime, type=RESULT_COLLECTOR_TOPIC,
        factory=lambda: VQAResultCollectorAgent(
            completion_events=completion_events,
            final_results_store=final_results_store
        )
    )
    main_logger.info("Sequential VQA agents re-registered with runtime using topic names as types.")


    runtime.start()
    main_logger.info("Runtime started.")

    all_pipeline_results: List[Dict[str, Any]] = []
    verbose_logging = current_config.get("inference_settings", {}).get("verbose", False)

    for task_idx, task_data in enumerate(tasks):
        start_time_task = time.time()
        qid = task_data["question_id"]
        image_path = task_data["image_path"] # Original path for the final dict
        question = task_data["question"]
        target_answers = task_data.get("target_answer") # Allow None
        question_type = task_data.get("question_type", "other") # Allow None

        main_logger.info(f"Processing task {task_idx + 1}/{len(tasks)} (QID: {qid})")

        completion_events[qid] = asyncio.Event()

        valid_image_url_for_message: Union[str, None] = None
        task_error: Union[str, None] = None
        try:
            processed_url = process_image_for_vlm_agent(task_data["image_path"], current_config)
            if not isinstance(processed_url, str) or not \
               (processed_url.startswith("data:image/") or processed_url.startswith("http")):
                raise ValueError(f"Invalid image URL/data URI from process_image_for_vlm_agent: {str(processed_url)[:100]}")
            valid_image_url_for_message = processed_url
        except Exception as e_img:
            main_logger.error(f"QID {qid}: Error processing image '{os.path.basename(task_data['image_path'])}': {e_img}", exc_info=verbose_logging)
            task_error = f"ImageProcessingError: {str(e_img)}"

        if task_error or not valid_image_url_for_message:
            err_result = create_error_result_dict(
                qid, image_path, question, target_answers,
                task_error or "Failed to get valid image URL",
                "ImageError", "sequential", question_type
            )
            err_result["processing_time_seconds"] = round(time.time() - start_time_task, 2)
            all_pipeline_results.append(err_result)
            completion_events[qid].set() # Unblock if anyone was waiting (though not strictly necessary here)
            continue

        initial_message = VQATaskRelayMessage(
            question_id=qid,
            image_url=valid_image_url_for_message,
            question=question,
            target_answers=target_answers,
            question_type=question_type,
            current_step_output="",  # Add this field with default value
            error=None   
        )

        # Publish to the first agent in the sequence
        # The `source` of TopicId can be any unique string for this publication instance.
        await runtime.publish_message(
            initial_message,
            topic_id=TopicId(IMAGE_CONTEXTUALIZER_TOPIC, source=f"workflow_start_{qid}")
        )
        main_logger.debug(f"QID {qid}: Initial message published to {IMAGE_CONTEXTUALIZER_TOPIC}.")

        try:
            await asyncio.wait_for(completion_events[qid].wait(), timeout=current_config.get("inference_settings",{}).get("sequential_task_timeout", 120.0))
            main_logger.info(f"QID {qid}: Task processing completed.")
            # Result is now in final_results_store[qid]
            task_result_data = final_results_store.get(qid)
            if task_result_data:
                task_result_data["image_path"] = image_path # Add original image path
                task_result_data["question_type"] = question_type # Ensure it's there
                if target_answers is not None and not task_result_data.get("error"):
                    eval_target = target_answers
                    if isinstance(target_answers, list) and target_answers and isinstance(target_answers[0], dict) and 'answer' in target_answers[0]:
                        eval_target = target_answers[0]['answer']
                    elif isinstance(target_answers, list) and target_answers and isinstance(target_answers[0], str):
                        eval_target = target_answers[0]


                    direct_check, _, _ = perform_direct_accuracy_check(
                        final_answer_text=task_result_data["final_answer"],
                        target_answers=eval_target, # Use potentially processed target
                        question_type=task_result_data["question_type"],
                        f1_threshold_other=f1_threshold_setting,
                        verbose=verbose_logging
                    )
                    task_result_data["direct_accuracy_check"] = direct_check
                else:
                     task_result_data["direct_accuracy_check"] = {"notes": "Evaluation skipped due to missing target or error."}

                task_result_data["processing_time_seconds"] = round(time.time() - start_time_task, 2)
                all_pipeline_results.append(task_result_data)
            else:
                main_logger.error(f"QID {qid}: Completion event set, but no result found in store.")
                err_result = create_error_result_dict(
                    qid, image_path, question, target_answers,
                    "InternalError: Result missing after completion event",
                    "WorkflowError"
                )
                err_result["flow_type_used"] = "sequential"
                err_result["question_type"] = question_type
                err_result["processing_time_seconds"] = round(time.time() - start_time_task, 2)
                all_pipeline_results.append(err_result)

        except asyncio.TimeoutError:
            main_logger.error(f"QID {qid}: Completion event set, but no result found in store.")
            err_result = create_error_result_dict(
                qid, image_path, question, target_answers,
                "InternalError: Result missing after completion event"
            )
            err_result["flow_type_used"] = "sequential"
            err_result["question_type"] = question_type
            err_result["processing_time_seconds"] = round(time.time() - start_time_task, 2)
            all_pipeline_results.append(err_result)
        except Exception as e_flow:
            main_logger.error(f"QID {qid}: Error during sequential flow execution: {e_flow}", exc_info=verbose_logging)
            err_result = create_error_result_dict(
                qid, image_path, question, target_answers,
                f"WorkflowExecutionError: {str(e_flow)}"
            )
            err_result["flow_type_used"] = "sequential"
            err_result["question_type"] = question_type
            err_result["processing_time_seconds"] = round(time.time() - start_time_task, 2)
            all_pipeline_results.append(err_result)
        finally:
            if qid in completion_events: # Clean up event
                del completion_events[qid]
            # final_results_store entry for qid is kept for now

    await runtime.stop_when_idle()
    main_logger.info("Runtime stopped.")
    return all_pipeline_results