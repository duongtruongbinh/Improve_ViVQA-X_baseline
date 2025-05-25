import torch
import time
import os
import logging
import json
import aiohttp

from ..config_loader import app_config
from ..evaluation import perform_direct_accuracy_check
from ..image_utils import process_image_for_vlm_agent

SIMPLE_DIRECT_SYSTEM_PROMPT = (
    "You are a Visual Question Answering (VQA) system. "
    "Use only the information visible in the image. "
    "Answer each question with one word or short phrase whenever possible. "
    "Always use exactly this output format, with no extra text:\n"
    "Answer: <your concise answer>"
)

async def run_simple_direct_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str = "unknown_qid",
    target_answer: any = None,
    question_type: str = "other",
    logger_instance: logging.Logger = None,
    config_settings=None
) -> dict:
    start_time = time.time()
    
    logger = logger_instance if logger_instance else logging.getLogger("SimpleDirectVQAFlow")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        is_verbose_cfg = False
        if config_settings and "inference_settings" in config_settings:
            is_verbose_cfg = config_settings["inference_settings"].get("verbose", False)
        elif hasattr(app_config, "inference_settings"): 
            is_verbose_cfg = app_config.get("inference_settings", {}).get("verbose", False)
        logger.setLevel(logging.DEBUG if is_verbose_cfg else logging.INFO)

    if not config_settings:
        config_settings = app_config
        logger.debug(f"QID {question_id} [SimpleDirectFlow]: config_settings not passed, using global app_config.")

    response_data = {
        "question_id": str(question_id),
        "image_path": image_path,
        "question": question,
        "target_answers": target_answer,
        "question_type": question_type,
        "final_answer": "[Pipeline Incomplete]",
        "processing_time_seconds": 0.0,
        "error": None,
        "direct_accuracy_check": {}
    }

    logger.debug(f"QID {question_id} [SimpleDirectFlow]: Type of config_settings object: {type(config_settings)}")
    if isinstance(config_settings, dict):
        logger.debug(f"QID {question_id} [SimpleDirectFlow]: Top-level keys in config_settings: {list(config_settings.keys())}")
        key_to_check = "vllm_details"
        if key_to_check in config_settings:
            logger.debug(f"QID {question_id} [SimpleDirectFlow]: Key '{key_to_check}' IS PRESENT in config_settings.")
            vlm_details_direct_access = config_settings[key_to_check]
            logger.debug(f"QID {question_id} [SimpleDirectFlow]: Value from config_settings['{key_to_check}']: {json.dumps(vlm_details_direct_access, default=str)[:500]}...")
            vlm_details_via_get = config_settings.get(key_to_check)
            logger.debug(f"QID {question_id} [SimpleDirectFlow]: Value from config_settings.get('{key_to_check}'): {json.dumps(vlm_details_via_get, default=str)[:500]}...")
            vlm_details = vlm_details_via_get if vlm_details_via_get is not None else {} # Ưu tiên .get() nếu nó hoạt động
            if not vlm_details and vlm_details_direct_access: # Nếu .get() trả về rỗng nhưng truy cập trực tiếp có giá trị
                 logger.warning(f"QID {question_id} [SimpleDirectFlow]: config_settings.get('{key_to_check}') returned empty/None, but direct access config_settings['{key_to_check}'] had data. Using direct access.")
                 vlm_details = vlm_details_direct_access
        else:
            logger.error(f"QID {question_id} [SimpleDirectFlow]: Key '{key_to_check}' IS NOT FOUND in config_settings keys. vlm_details will be empty.")
            vlm_details = {}
    else:
        logger.error(f"QID {question_id} [SimpleDirectFlow]: config_settings is NOT a dict (type: {type(config_settings)}). Cannot reliably get 'vllm_details'.")
        vlm_details = {}
        
    logger.debug(f"QID {question_id} [SimpleDirectFlow]: Final vlm_details to be used: {json.dumps(vlm_details, default=str)[:500]}...")


    api_provider = vlm_details.get("api_provider", "vllm") 
    logger.debug(f"QID {question_id} [SimpleDirectFlow]: Determined api_provider: {api_provider}")

    api_base_url = None
    model_identifier = None
    api_key = "EMPTY" 
    temperature = vlm_details.get("temperature", 0.2) 

    if api_provider == "openai":
        api_base_url = vlm_details.get("openai_base_url")
        model_identifier = vlm_details.get("openai_vlm_model_name") 
        api_key_config = vlm_details.get("openai_api_key", "EMPTY")
        if isinstance(api_key_config, str) and api_key_config.startswith("${") and api_key_config.endswith("}"):
             env_var_name = api_key_config[2:-1]
             api_key = os.getenv(env_var_name, "EMPTY")
        else:
            api_key = api_key_config if isinstance(api_key_config, str) else "EMPTY"
        logger.debug(f"QID {question_id} [SimpleDirectFlow]: OpenAI provider - URL='{api_base_url}', Model='{model_identifier}'")
    elif api_provider == "vllm":
        api_base_url = vlm_details.get("vlm_url")
        model_identifier = vlm_details.get("vlm_model_name")
        api_key = vlm_details.get("api_key", "EMPTY")
        logger.debug(f"QID {question_id} [SimpleDirectFlow]: vLLM provider - URL='{api_base_url}', Model='{model_identifier}'")
    else:
        error_msg = f"Unsupported api_provider '{api_provider}' in vlm_details. Must be 'vllm' or 'openai'."
        logger.error(error_msg)
        response_data["error"] = f"ConfigurationError: {error_msg}"
        response_data["final_answer"] = "[Config Error - Provider]"
    
    if not response_data["error"] and (not api_base_url or not model_identifier):
        missing_details_list = []
        actual_url_key_tried = ""
        actual_model_key_tried = ""

        if api_provider == "vllm":
            actual_url_key_tried = "vlm_url"
            actual_model_key_tried = "vlm_model_name"
        elif api_provider == "openai":
            actual_url_key_tried = "openai_base_url"
            actual_model_key_tried = "openai_vlm_model_name"
        
        if not api_base_url:
            missing_details_list.append(f"URL (expected key: 'vlm_details.{actual_url_key_tried}')")
        if not model_identifier:
            missing_details_list.append(f"Model Identifier (expected key: 'vlm_details.{actual_model_key_tried}')")
        
        error_msg_detail = f"For api_provider '{api_provider}', the following are missing or empty: {'; '.join(missing_details_list)}."
        logger.error(error_msg_detail)
        response_data["error"] = f"ConfigurationError: {error_msg_detail}"
        response_data["final_answer"] = "[Config Error - URL/Model]"
        
    if response_data["error"]: 
        if target_answer is not None:
            direct_check_res, _, _ = perform_direct_accuracy_check(
                response_data["final_answer"], target_answer, question_type, response_data["error"],
                verbose=logger.isEnabledFor(logging.DEBUG)
            )
            response_data["direct_accuracy_check"] = direct_check_res
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data
    
    processed_image_url = None
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        processed_image_url = process_image_for_vlm_agent(image_path, config_settings)
        if not isinstance(processed_image_url, str) or not processed_image_url.startswith("data:image/"):
            raise ValueError(f"Image processing returned invalid data URL format: {str(processed_image_url)[:100]}...")
    except Exception as e_img:
        logger.error(f"ERROR (SimpleDirectFlow QID {question_id}) processing image '{os.path.basename(image_path)}': {e_img}", exc_info=logger.isEnabledFor(logging.DEBUG))
        response_data["error"] = f"ImageProcessingError: {str(e_img)}"
        response_data["final_answer"] = "[Image Error]"
        if target_answer is not None:
            direct_check_res, _, _ = perform_direct_accuracy_check(
                response_data["final_answer"], target_answer, question_type, response_data["error"], 
                verbose=logger.isEnabledFor(logging.DEBUG)
            )
            response_data["direct_accuracy_check"] = direct_check_res
        response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
        return response_data

    messages_for_api = []
    if SIMPLE_DIRECT_SYSTEM_PROMPT and SIMPLE_DIRECT_SYSTEM_PROMPT.strip():
         messages_for_api.append({"role": "system", "content": SIMPLE_DIRECT_SYSTEM_PROMPT}) 
    
    user_content_list = []
    user_content_list.append({"type": "image_url", "image_url": {"url": processed_image_url}})
    user_content_list.append({"type": "text", "text": f"Question: {question}"})
    messages_for_api.append({"role": "user", "content": user_content_list})
    
    request_payload = {
        "model": model_identifier,
        "messages": messages_for_api,
        "max_tokens": 30, 
        "temperature": temperature 
    }
    if isinstance(temperature, (float, int)) and temperature > 0.0:
        request_payload["do_sample"] = True 
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_key and isinstance(api_key, str) and api_key.upper() != "EMPTY" and api_key.upper() != "NONE":
        headers["Authorization"] = f"Bearer {api_key}"

    generated_answer_parsed = "[VLMCallFailed]"
    chat_completions_url = f"{api_base_url.rstrip('/')}/chat/completions"

    try:
        logger.info(f"SimpleDirectFlow (QID: {question_id}): Calling endpoint {chat_completions_url} for model {model_identifier}...")
        logger.debug(f"SimpleDirectFlow (QID: {question_id}): Request Payload: {json.dumps(request_payload, default=str)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(chat_completions_url, headers=headers, json=request_payload) as resp:
                response_text_for_logging = await resp.text()
                logger.debug(f"SimpleDirectFlow (QID: {question_id}): API Response Status: {resp.status}, Response Text: {response_text_for_logging[:500]}")
                if resp.status == 200:
                    api_response = json.loads(response_text_for_logging) 
                    choices = api_response.get("choices")
                    raw_model_output = ""
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        message_content = choices[0].get("message", {}).get("content")
                        if message_content is not None:
                             raw_model_output = str(message_content).strip()
                        else: 
                            raw_model_output = str(choices[0].get("text","")).strip() if isinstance(choices[0], dict) else ""
                    else: 
                        raw_model_output = str(api_response.get("text", api_response.get("message",{}).get("content",""))).strip()

                    logger.info(f"SimpleDirectFlow (QID: {question_id}): Raw API Output Parsed='{raw_model_output[:200]}...'")

                    if "Answer: " in raw_model_output:
                        generated_answer_parsed = raw_model_output.split("Answer:", 1)[-1].strip()
                    elif raw_model_output.startswith("Answer "):
                        generated_answer_parsed = raw_model_output.split("Answer ", 1)[-1].strip()
                    else:
                        generated_answer_parsed = raw_model_output
                        if raw_model_output: 
                            logger.warning(f"SimpleDirectFlow (QID: {question_id}): Model output did not strictly follow 'Answer: <text>' format. Using: '{raw_model_output}'")
                    
                    if not generated_answer_parsed and raw_model_output: 
                        generated_answer_parsed = raw_model_output 
                    elif not raw_model_output: 
                        generated_answer_parsed = "[VLMEmptyContent]"
                else:
                    response_data["error"] = f"APIError: Status {resp.status} - {response_text_for_logging[:200]}"
                    generated_answer_parsed = f"[APIError_{resp.status}]"

    except aiohttp.ClientConnectorError as e_conn:
        logger.error(f"ERROR (SimpleDirectFlow QID {question_id}) connecting to API: {e_conn}", exc_info=logger.isEnabledFor(logging.DEBUG))
        response_data["error"] = f"APIConnectionError: {str(e_conn)}"
        generated_answer_parsed = "[APIConnectionError]"
    except Exception as e_call:
        logger.error(f"ERROR (SimpleDirectFlow QID {question_id}) during API call: {e_call}", exc_info=logger.isEnabledFor(logging.DEBUG))
        response_data["error"] = f"APICallError: {str(e_call)}"
        generated_answer_parsed = "[APICallError]"

    response_data["final_answer"] = generated_answer_parsed

    if target_answer is not None:
        f1_threshold = config_settings.get("inference_settings", {}).get("f1_threshold_for_other_type", 0.5)
        direct_check_results, _, _ = perform_direct_accuracy_check(
            final_answer_text=response_data["final_answer"],
            target_answers=target_answer,
            question_type=response_data["question_type"],
            current_error=response_data.get("error"),
            verbose=logger.isEnabledFor(logging.DEBUG),
            f1_threshold_other=f1_threshold
        )
        response_data["direct_accuracy_check"] = direct_check_results

    response_data["processing_time_seconds"] = round(time.time() - start_time, 2)
    logger.info(f"--- Simple Direct VQA Pipeline END for QID: {question_id}, Final Answer: '{str(response_data['final_answer'])[:100]}...', Time: {response_data['processing_time_seconds']:.2f}s ---")
    return response_data