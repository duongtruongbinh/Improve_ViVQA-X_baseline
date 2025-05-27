import json
import time
import os
import logging

from ..config_loader import app_config
from ..image_utils import process_image_for_vlm_agent
from ..evaluation import perform_direct_accuracy_check
from ..agents.reflection_agents import VQAGeneratorAgent, VQAReflectorAgent
from ..vllm_clients import llm_config_vlm

async def run_simple_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str = "unknown_qid",
    target_answer: any = None,
    question_type: str = "other",
    vqa_agent_instance: VQAGeneratorAgent = None,
    critique_text: str = None,
    logger_instance: logging.Logger = None
) -> dict:
    start_time = time.time()
    current_config = app_config
    inference_settings = current_config.get("inference_settings", {})
    verbose = inference_settings.get("verbose", False)

    logger = logger_instance
    if not logger:
        logger = logging.getLogger("SimpleVQAFlowDefaultLogger")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    response_data = {
        "question_id": str(question_id),
        "image_path": image_path,
        "question": question,
        "target_answers": target_answer,
        "question_type": question_type,
        "final_answer": "[Pipeline Incomplete]",
        "confidence_score": 0.0,
        "processing_time_seconds": 0.0,
        "error": None,
        "direct_accuracy_check": {}
    }

    valid_image_url_for_message = None
    try:
        processed_image_url = process_image_for_vlm_agent(image_path, current_config)
        if not isinstance(processed_image_url, str) or not \
            (processed_image_url.startswith("data:image/") or processed_image_url.startswith("http")):
            raise ValueError(f"Image processing returned invalid or empty URL: {str(processed_image_url)[:100]}")
        valid_image_url_for_message = processed_image_url
    except Exception as e_img:
        logger.error(f"ERROR (Simple VQA Flow QID {question_id}) processing image '{os.path.basename(image_path)}': {e_img}", exc_info=verbose)
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        if target_answer is not None:
            direct_check_res, _, _ = perform_direct_accuracy_check(
                response_data["final_answer"], target_answer, question_type, response_data["error"], verbose
            )
            response_data["direct_accuracy_check"] = direct_check_res
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data

    agent = vqa_agent_instance
    if agent is None:
        try:
            if llm_config_vlm is None:
                raise ValueError("llm_config_vlm is not available or not imported correctly.")
            agent = VQAGeneratorAgent(
                name=f"ReflectionAgent_QID_{question_id}",
                llm_config=llm_config_vlm 
            )
        except Exception as e_agent_init:
            logger.critical(f"Failed to initialize VQAGeneratorAgent for QID {question_id}: {e_agent_init}", exc_info=verbose)
            response_data["error"] = f"AgentInitializationError: {str(e_agent_init)}"
            if target_answer is not None:
                direct_check_res, _, _ = perform_direct_accuracy_check(
                    response_data["final_answer"], target_answer, question_type, response_data["error"], verbose
                )
                response_data["direct_accuracy_check"] = direct_check_res
            response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
            return response_data
            
    agent.reset()
    agent.prepare_system_message(question_text=question, question_type=question_type, critique_text=critique_text)
        
    user_prompt_text = "Based on the image and your instructions, provide the answer to the question."

    user_message_content_payload = [
        {"type": "image_url", "image_url": {"url": valid_image_url_for_message}},
        {"type": "text", "text": user_prompt_text}
    ]

    messages_for_agent_turn = []
    if agent.system_message and agent.system_message.strip():
        messages_for_agent_turn.append({"role": "system", "content": agent.system_message})
    
    messages_for_agent_turn.append({"role": "user", "content": user_message_content_payload})
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"VQA Agent (QID: {question_id}) System Message (rendered): {agent.system_message}")
        logger.debug(f"VQA Agent (QID: {question_id}) User Message Text: {user_prompt_text}")

    logger.info(f"VQA Agent (QID: {question_id}): Calling VLM...")

    generated_answer = "[VLMCallFailed]"
    confidence_score = 0.0
    try:
        chat_res = await agent.a_generate_reply(
            messages=messages_for_agent_turn,
            sender=None,
        )
        generated_content_str = chat_res.get("content") if chat_res else None

        if not generated_content_str:
            logger.warning(f"VQA Agent returned empty content for QID {question_id}.")
            response_data["error"] = "VQAAgentError: Agent returned empty content."
            generated_answer = "[VQAAgentEmptyContent]"
        else:
            generated_answer, confidence_score = agent.parse_model_output(generated_content_str, question_type)
            logger.info(f"VQA Agent Output (QID: {question_id}): Answer='{generated_answer[:100]}...', Confidence={confidence_score:.2f}")

    except Exception as e_agent_call:
        logger.error(f"ERROR (Simple VQA Flow QID {question_id}) in VQA Agent call: {e_agent_call}", exc_info=verbose)
        response_data["error"] = f"VQAAgentCallError: {str(e_agent_call)}"
        generated_answer = "[VQAAgentCallError]"

    response_data["final_answer"] = generated_answer
    response_data["confidence_score"] = confidence_score

    if target_answer is not None:
        direct_check_results, _, _ = perform_direct_accuracy_check(
            final_answer_text=response_data["final_answer"],
            target_answers=target_answer,
            question_type=response_data["question_type"],
            current_error=response_data.get("error"),
            verbose=verbose,
            f1_threshold_other=inference_settings.get("f1_threshold_for_other_type", 0.5)
        )
        response_data["direct_accuracy_check"] = direct_check_results

    end_time = time.time()
    response_data["processing_time_seconds"] = round(end_time - start_time, 2)

    logger.info(f"--- Simple VQA Pipeline END for QID: {question_id}, Final Answer: '{str(response_data['final_answer'])[:100]}...', Confidence: {confidence_score:.2f}, Time: {response_data['processing_time_seconds']:.2f}s ---")
    return response_data

async def run_reflection_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str,
    target_answer: any,
    question_type: str,
    vqa_agent: VQAGeneratorAgent = None,
    reflector_agent: VQAReflectorAgent = None,
    max_reflections: int = 2,
    confidence_threshold: float = 0.8,
    logger_instance: logging.Logger = None
) -> dict:
    logger = logger_instance
    if not logger:
        logger = logging.getLogger("ReflectionVQAFlowDefaultLogger")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    # Initialize agents if not provided
    if vqa_agent is None:
        vqa_agent = VQAGeneratorAgent(
            name=f"VQAGenerator_QID_{question_id}",
            llm_config=llm_config_vlm
        )
    
    if reflector_agent is None:
        reflector_agent = VQAReflectorAgent(
            name=f"VQAReflector_QID_{question_id}",
            llm_config=llm_config_vlm
        )

    # Initial answer
    response = await run_simple_vqa_pipeline(
        image_path=image_path,
        question=question,
        question_id=question_id,
        target_answer=target_answer,
        question_type=question_type,
        vqa_agent_instance=vqa_agent,
        logger_instance=logger
    )

    # Check if initial answer is correct
    if response.get("direct_accuracy_check", {}).get("is_correct", False):
        logger.info(f"Initial answer is correct, skipping reflection.")
        return response

    # Reflection loop
    best_response = response
    best_confidence = response.get("confidence_score", 0.0)
    
    for i in range(max_reflections):
        current_confidence = response.get("confidence_score", 0.0)
        
        # Stop if we've reached confidence threshold
        if current_confidence >= confidence_threshold:
            logger.info(f"Confidence threshold met ({current_confidence:.2f} >= {confidence_threshold}), stopping reflection.")
            break

        # Get reflection
        reflection = await reflector_agent.evaluate(
            question=question,
            answer=response["final_answer"],
            question_type=question_type
        )
        
        if reflection == "[APPROVED]":
            logger.info("Answer approved by reflector, stopping reflection.")
            break
            
        logger.info(f"Reflection {i+1}/{max_reflections}: {reflection}")
        
        # Improve answer based on reflection
        response = await run_simple_vqa_pipeline(
            image_path=image_path,
            question=question,
            question_id=question_id,
            target_answer=target_answer,
            question_type=question_type,
            vqa_agent_instance=vqa_agent,
            critique_text=reflection,
            logger_instance=logger
        )
        
        # Update best response if current is better
        if response.get("direct_accuracy_check", {}).get("is_correct", False):
            logger.info("Found correct answer, stopping reflection.")
            return response
            
        current_confidence = response.get("confidence_score", 0.0)
        if current_confidence > best_confidence:
            best_response = response
            best_confidence = current_confidence

    # Return best response found
    return best_response