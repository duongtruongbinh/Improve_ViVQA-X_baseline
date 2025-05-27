import os
import time
import json
import re
import sys
import traceback
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

_yes_no_starters_specialized = [
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "am ",
    "can ", "could ", "will ", "would ", "should ", 
    "has ", "have ", "had ", "may ", "might ", "must ",
    "is there ", "are there ", "was there ", "were there ",
    "can there ", "will there ", "is it ", "are they "
]
_number_starters_specialized = [
    "how many", "what is the number of", "count the number of", "what number"
]

FAILURE_MARKERS = [
    "[Answer Failed]", "sorry", "unable to answer", "cannot answer"
]

def get_question_type_specialized(question_text: str) -> str:
    """Determine the type of question based on its text.
    
    Args:
        question_text: The question text to analyze
        
    Returns:
        str: Question type - one of "yes/no", "number", or "other"
    """
    if not isinstance(question_text, str):
        return "other" 
    q_lower = question_text.lower().strip()
    
    if any(q_lower.startswith(s) for s in _yes_no_starters_specialized):
        return "yes/no"
    if any(q_lower.startswith(s) for s in _number_starters_specialized):
        return "number"
    return "other"

try:
    from ..evaluation import perform_direct_accuracy_check
except ImportError as e_check:
    print(f"ERROR: Could not import from evaluation.py: {e_check}. Ensure evaluation.py is in the src directory.")
    sys.exit(1)

try:
    from ..config_loader import app_config
    from ..image_utils import process_image_for_vlm_agent
    from ..agents.specialized_agents import (
        vqa_orchestrator, initial_vlm_agent, failure_analysis_agent,
        object_attribute_agent, reattempt_vlm_agent
    )
    from ..vllm_clients import llm_client_vllm, vlm_client_vllm
    from ..prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_VQA_V2, INITIAL_VLM_SYSTEM_PROMPT_DEFAULT,
        FAILURE_ANALYSIS_SYSTEM_PROMPT, OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS, REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS
    )
    from autogen.agentchat.conversable_agent import ConversableAgent
except ImportError as e:
    print(f"ERROR: Error importing modules in main_vqa_flow.py: {e}.")
    print("Ensure all required files (config_loader, image_utils, agents, prompts, vllm_clients, check) exist and are accessible, and AutoGen packages are installed.")
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

class SpecializedVQAWorkflow:
    """VQA workflow that uses specialized agents for different question types."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        f1_threshold: float = 0.5,
        force_multi_agents: bool = False
    ):
        self.config = config
        self.f1_threshold = f1_threshold
        self.force_multi_agents = force_multi_agents
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def _check_agent_initialization(self) -> Optional[str]:
        """Check if all required agents are initialized."""
        agent_instances = [vqa_orchestrator, initial_vlm_agent, failure_analysis_agent, 
                         object_attribute_agent, reattempt_vlm_agent]
        client_instances = [vlm_client_vllm, llm_client_vllm]
        
        if any(agent is None for agent in agent_instances) or any(client is None for client in client_instances):
            return "One or more core agents or clients are not initialized"
        return None
        
    async def _get_initial_answer(
        self,
        processed_url: str,
        question: str,
        question_type: str
    ) -> Tuple[str, Optional[str]]:
        """Get initial answer from VLM agent."""
        try:
            # Set system prompt based on dataset
            active_dataset = self.config.get("datasets", {}).get("dataset_name", "default")
            if active_dataset == "vqa-v2":
                initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_VQA_V2)
            else:
                initial_vlm_agent.update_system_message(INITIAL_VLM_SYSTEM_PROMPT_DEFAULT)
                
            # Prepare message
            message_parts = [{"type": "text", "text": question}]
            if processed_url:
                message_parts.append({"type": "image_url", "image_url": {"url": processed_url}})
            else:
                return "[MissingImageError]", "Valid image URL is required for VQA but was not available"
                
            initial_user_message = {"role": "user", "content": message_parts}
            
            # Get response
            initial_vlm_agent.reset()
            chat_result = await vqa_orchestrator.a_initiate_chat(
                recipient=initial_vlm_agent,
                message=initial_user_message,
                max_turns=1,
                summary_method="last_msg",
                silent=(not self.logger.isEnabledFor(logging.DEBUG))
            )
            
            answer = get_message_content(chat_result.chat_history[-1] if chat_result and chat_result.chat_history else {})
            if not answer:
                return "[Initial VLM Agent returned empty message]", None
                
            return answer, None
            
        except Exception as e:
            error_msg = f"Error getting initial answer: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return "[Agent Call Failed]", error_msg
            
    async def _analyze_failure(
        self,
        question: str,
        initial_answer: str
    ) -> Tuple[str, Optional[str]]:
        """Analyze failure of initial answer."""
        try:
            failure_analysis_agent.update_system_message(FAILURE_ANALYSIS_SYSTEM_PROMPT)
            analysis_message = (
                f"Original Question: '{question}'\nInitial VLM Response: '{initial_answer}'\n"
                "Analyze failure. Suggest strategy: 'numeric reattempt needed for: [item]' OR 'general reattempt, focus on: [items]'."
            )
            
            failure_analysis_agent.reset()
            chat_result = await vqa_orchestrator.a_initiate_chat(
                recipient=failure_analysis_agent,
                message=analysis_message,
                max_turns=1,
                summary_method="last_msg",
                silent=(not self.logger.isEnabledFor(logging.DEBUG))
            )
            
            analysis = get_message_content(chat_result.chat_history[-1] if chat_result and chat_result.chat_history else {})
            if not analysis:
                return "[Analysis Agent returned empty message]", None
                
            return analysis, None
            
        except Exception as e:
            error_msg = f"Error analyzing failure: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return "[Analysis Failed]", error_msg
            
    async def _get_reattempt_answer(
        self,
        processed_url: str,
        question: str,
        analysis: str,
        question_type: str
    ) -> Tuple[str, Optional[str]]:
        """Get reattempt answer based on analysis."""
        try:
            # Set system prompt based on dataset
            active_dataset = self.config.get("datasets", {}).get("dataset_name", "default")
            if active_dataset == "vqa-v2":
                reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS)
            else:
                reattempt_vlm_agent.update_system_message(REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS)
                
            # Prepare message
            message_parts = [{"type": "text", "text": question}]
            if processed_url:
                message_parts.append({"type": "image_url", "image_url": {"url": processed_url}})
            else:
                return "[MissingImageError]", "Valid image URL is required for VQA but was not available"
                
            reattempt_user_message = {"role": "user", "content": message_parts}
            
            # Get response
            reattempt_vlm_agent.reset()
            chat_result = await vqa_orchestrator.a_initiate_chat(
                recipient=reattempt_vlm_agent,
                message=reattempt_user_message,
                max_turns=1,
                summary_method="last_msg",
                silent=(not self.logger.isEnabledFor(logging.DEBUG))
            )
            
            answer = get_message_content(chat_result.chat_history[-1] if chat_result and chat_result.chat_history else {})
            if not answer:
                return "[Reattempt VLM Agent returned empty message]", None
                
            return answer, None
            
        except Exception as e:
            error_msg = f"Error getting reattempt answer: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return "[Reattempt Agent Call Failed]", error_msg
            
    def _is_failure_marker(self, answer: str) -> bool:
        """Check if answer contains failure markers."""
        return any(marker in answer.lower() for marker in FAILURE_MARKERS)
        
    async def run_workflow(
        self,
        image_path: str,
        question: str,
        question_id: str = "unknown_qid",
        target_answer: Any = None,
        question_type: str = "other"
    ) -> Dict[str, Any]:
        """Run the VQA workflow."""
        start_time = time.time()
        response_data = {
            "question_id": str(question_id),
            "image_path": image_path,
            "question": question,
            "target_answers": target_answer,
            "question_type": question_type,
            "initial_answer": "",
            "final_answer": "",
            "match_baseline_failed": False,
            "is_numeric_reattempt": False,
            "analysis_output": "",
            "object_attributes_queried": "",
            "reattempt_answer": "",
            "grades": [],
            "processing_time_seconds": 0.0,
            "error": None,
            "direct_accuracy_check": {},
            "majority_vote": ""
        }
        
        # Check agent initialization
        if error := self._check_agent_initialization():
            response_data["error"] = error
            response_data["final_answer"] = "[Answer Failed]"
            return response_data
            
        # Process image
        try:
            processed_url = process_image_for_vlm_agent(image_path, self.config)
            if not processed_url:
                response_data["error"] = "Image processing failed"
                response_data["final_answer"] = "[Answer Failed]"
                return response_data
        except Exception as e:
            response_data["error"] = f"Image processing error: {str(e)}"
            response_data["final_answer"] = "[Answer Failed]"
            return response_data
            
        # Get initial answer
        initial_answer, error = await self._get_initial_answer(processed_url, question, question_type)
        response_data["initial_answer"] = initial_answer
        if error:
            response_data["error"] = error
            response_data["final_answer"] = "[Answer Failed]"
            return response_data
            
        # Check if reattempt is needed
        if self._is_failure_marker(initial_answer) or self.force_multi_agents:
            response_data["match_baseline_failed"] = True
            
            # Analyze failure
            analysis, error = await self._analyze_failure(question, initial_answer)
            response_data["analysis_output"] = analysis
            if error:
                response_data["error"] = error
                response_data["final_answer"] = "[Answer Failed]"
                return response_data
                
            # Get reattempt answer
            reattempt_answer, error = await self._get_reattempt_answer(
                processed_url, question, analysis, question_type
            )
            response_data["reattempt_answer"] = reattempt_answer
            if error:
                response_data["error"] = error
                response_data["final_answer"] = "[Answer Failed]"
                return response_data
                
            # Use reattempt answer if valid
            if not self._is_failure_marker(reattempt_answer):
                response_data["final_answer"] = reattempt_answer
            else:
                response_data["final_answer"] = initial_answer
        else:
            response_data["final_answer"] = initial_answer
            
        # Evaluate result
        direct_check_results, grades_list, mj_vote_str = perform_direct_accuracy_check(
            final_answer_text=response_data["final_answer"],
            target_answers=target_answer,
            question_type=question_type,
            current_error=response_data.get("error"),
            verbose=self.logger.isEnabledFor(logging.DEBUG),
            f1_threshold_other=self.f1_threshold
        )
        
        response_data["direct_accuracy_check"] = direct_check_results
        response_data["grades"] = grades_list
        response_data["majority_vote"] = mj_vote_str
        
        # Calculate processing time
        end_time = time.time()
        response_data["processing_time_seconds"] = round(end_time - start_time, 2)
        
        # Save response if configured
        if self.config.get("inference_settings", {}).get("save_individual_responses", False):
            filename_template = self.config["inference_settings"].get("output_individual_response_filename_template")
            if filename_template:
                individual_filename = filename_template.format(question_id=question_id)
                write_response_to_jsonl(response_data, individual_filename)
                
        return response_data

async def run_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str = "unknown_qid",
    target_answer: Any = None,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Run the VQA pipeline with the specialized workflow."""
    workflow = SpecializedVQAWorkflow(
        config=app_config,
        logger=logger_instance,
        f1_threshold=app_config.get("inference_settings", {}).get("f1_threshold_for_other_type", 0.5),
        force_multi_agents=app_config.get("inference_settings", {}).get("force_multi_agents", False)
    )
    return await workflow.run_workflow(
        image_path=image_path,
        question=question,
        question_id=question_id,
        target_answer=target_answer,
        question_type=get_question_type_specialized(question)
    )