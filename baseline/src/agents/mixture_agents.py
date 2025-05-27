import logging
import os
import time
import asyncio
from typing import List, Dict, Any, Union, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core.models import ChatCompletionClient

PROMPT_DIR_MIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts', 'mixture')
jinja_env_mixture_agents = None
if os.path.isdir(PROMPT_DIR_MIXTURE):
    jinja_env_mixture_agents = Environment(
        loader=FileSystemLoader(PROMPT_DIR_MIXTURE),
        autoescape=select_autoescape(['j2']),
        trim_blocks=True,
        lstrip_blocks=True
    )
else:
    print(f"WARNING (mixture_agents.py): Prompt directory for Mixture of Agents not found at: {PROMPT_DIR_MIXTURE}.")


def load_mixture_prompt_template(template_filename: str, **kwargs) -> str:
    if jinja_env_mixture_agents is None:
        error_msg = f"ERROR (mixture_agents.py): Jinja environment not initialized. Cannot load '{template_filename}' from {PROMPT_DIR_MIXTURE}."
        logging.getLogger("MixturePromptLoader").error(error_msg)
        return f"Critical Error: Prompt template '{template_filename}' could not be loaded."
    try:
        template = jinja_env_mixture_agents.get_template(template_filename)
        return template.render(**kwargs)
    except Exception as e:
        logging.getLogger("MixturePromptLoader").error(f"ERROR loading/rendering prompt '{template_filename}': {e}", exc_info=True)
        return f"Critical Error: Prompt template '{template_filename}' rendering failed. Details: {str(e)}"

class UserVQATask(BaseModel):
    task: str
    image_url: str
    question_id: str
    target_answer: Optional[Any] = None
    question_type: str = "other"

class WorkerVQATask(BaseModel):
    task: str
    question_id: str 
    image_url: Optional[str] = None
    previous_results: List[str] = Field(default_factory=list)

class WorkerVQAResult(BaseModel):
    result: str
    question_id: str

class FinalVQAResult(BaseModel):
    question_id: str
    image_path: str
    question: str
    target_answers: Optional[Any] = None
    question_type: str = "other"
    final_aggregated_answer: str
    all_worker_outputs_by_layer: List[List[Dict[str, Any]]] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    direct_accuracy_check: Dict[str, Any] = Field(default_factory=dict)
    flow_type_used: str = "mixture_of_agents"
    stop_reason: str = "Unknown"

class VQAWorkerAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        agent_name: str = "VQAWorkerAgent"
    ) -> None:
        super().__init__(description="A VQA Worker Agent that answers questions based on an image and prior results.")
        self._model_client = model_client
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")

    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key}")


    @message_handler
    async def handle_task(self, message: WorkerVQATask, ctx: MessageContext) -> WorkerVQAResult:
        self.logger.debug(f"Received task for QID: {message.question_id} with {len(message.previous_results)} previous results. Image URL provided: {message.image_url is not None}")

        vlm_content_list_for_user_message = []
        if message.image_url:
            vlm_content_list_for_user_message.append({"type": "image_url", "image_url": {"url": message.image_url}})

        user_question_text = f"Question: {message.task}"
        vlm_content_list_for_user_message.append({"type": "text", "text": user_question_text})

        system_prompt_text: str
        if message.previous_results:
            valid_previous_results = [r for r in message.previous_results if r and not r.startswith("[")]
            template_to_load = "worker_synthesis.j2" if valid_previous_results else "worker_initial.j2"
            system_prompt_text = load_mixture_prompt_template(template_to_load, previous_results=valid_previous_results)
        else:
            system_prompt_text = load_mixture_prompt_template("worker_initial.j2")
        
        if "Critical Error" in system_prompt_text:
            self.logger.error(f"QID {message.question_id} ({self.id}) - Failed to load system prompt.")
            return WorkerVQAResult(result=f"[Error_PromptLoadingFailure_{self.id.key if self.id else 'unbound'}]", question_id=message.question_id)

        messages_for_client_call = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": vlm_content_list_for_user_message}
        ]
        
        self.logger.debug(f"QID {message.question_id} ({self.id}) - System Prompt: {system_prompt_text[:300]}...")
        self.logger.debug(f"QID {message.question_id} ({self.id}) - User Content for client: {vlm_content_list_for_user_message}")

        final_answer = f"[VLMError_{self.id.key if self.id else 'unbound'}]"
        try:
            model_result = await self._model_client.create(messages=messages_for_client_call)
            if isinstance(model_result.content, str) and model_result.content.strip():
                final_answer = model_result.content.strip()
            else:
                self.logger.warning(f"QID {message.question_id} ({self.id}) - VLM returned empty or non-string content: {type(model_result.content)}. Defaulting to '[EmptyVLMOutput]'.")
                final_answer = "[EmptyVLMOutput]"
        except Exception as e:
            self.logger.error(f"QID {message.question_id} ({self.id}) - Error during VLM call: {e}", exc_info=True)
            final_answer = f"[VLMCallException_{str(e)[:50]}]" 
            
        self.logger.info(f"QID {message.question_id} ({self.id}) - Output: '{final_answer[:100]}...'")
        return WorkerVQAResult(result=final_answer, question_id=message.question_id)

class VQAOrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
        agent_name: str = "VQAOrchestratorAgent"
    ) -> None:
        super().__init__(description="Orchestrates VQA tasks across multiple worker layers and aggregates the final answer.")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")

    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key}")

    @message_handler
    async def handle_task(self, message: UserVQATask, ctx: MessageContext) -> FinalVQAResult:
        start_time_orchestrator = time.time()
        self.logger.info(f"Received task for QID: {message.question_id} - '{message.task[:100]}...'")

        current_worker_task = WorkerVQATask(
            task=message.task,
            question_id=message.question_id,
            image_url=message.image_url,
            previous_results=[]
        )
        
        all_layers_outputs_for_final_result: List[List[Dict[str, Any]]] = []
        last_layer_string_results: List[str] = []
        orchestrator_error: Union[str, None] = None
        orchestrator_stop_reason: str = "Unknown"


        for i in range(self._num_layers):
            current_layer_idx_display = i + 1
            self.logger.info(f"QID {message.question_id}: Starting Layer {current_layer_idx_display}/{self._num_layers}")
            
            worker_ids_for_layer = [
                AgentId(worker_type, f"qid_{message.question_id}_layer_{i}_worker_{j}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            
            self.logger.debug(f"QID {message.question_id} Layer {current_layer_idx_display}: Dispatching to {len(worker_ids_for_layer)} workers: {worker_ids_for_layer}")
            
            layer_raw_results = await asyncio.gather(
                *[self.send_message(current_worker_task, worker_id) for worker_id in worker_ids_for_layer],
                return_exceptions=True
            )
            
            self.logger.info(f"QID {message.question_id} Layer {current_layer_idx_display}: Received {len(layer_raw_results)} results.")

            current_layer_processed_results_for_log: List[Dict[str, Any]] = []
            current_layer_string_results_for_next_task: List[str] = []
            has_valid_result_this_layer = False

            for res_idx, raw_res in enumerate(layer_raw_results):
                worker_id_str = str(worker_ids_for_layer[res_idx])
                if isinstance(raw_res, WorkerVQAResult):
                    current_layer_string_results_for_next_task.append(raw_res.result)
                    current_layer_processed_results_for_log.append({
                        "worker_id": worker_id_str, "output": raw_res.result, "error": None
                    })
                    if not raw_res.result.startswith(("[VLMError", "[EmptyVLMOutput", "[Error_PromptLoadingFailure", "[VLMCallException")):
                        has_valid_result_this_layer = True
                elif isinstance(raw_res, Exception):
                    self.logger.error(f"QID {message.question_id} Layer {current_layer_idx_display}: Worker {worker_id_str} raised: {raw_res}")
                    error_msg = f"[WorkerException_{worker_id_str}:{str(raw_res)[:100]}]"
                    current_layer_string_results_for_next_task.append(error_msg)
                    current_layer_processed_results_for_log.append({
                        "worker_id": worker_id_str, "output": None, "error": error_msg
                    })
                else:
                    self.logger.warning(f"QID {message.question_id} Layer {current_layer_idx_display}: Worker {worker_id_str} returned unexpected type: {type(raw_res)}. Content: {str(raw_res)[:100]}")
                    warn_msg = "[Error:UnexpectedWorkerResultType]"
                    current_layer_string_results_for_next_task.append(warn_msg)
                    current_layer_processed_results_for_log.append({
                        "worker_id": worker_id_str, "output": warn_msg, "error": "UnexpectedWorkerResultType"
                    })
            
            all_layers_outputs_for_final_result.append(current_layer_processed_results_for_log)

            if not has_valid_result_this_layer and i < self._num_layers - 1 :
                self.logger.error(f"QID {message.question_id}: Layer {current_layer_idx_display} produced no valid results. Cannot proceed.")
                orchestrator_error="LayerProducedNoValidResults"
                orchestrator_stop_reason=f"Layer_{current_layer_idx_display}_NoValidResults"
                break 
            
            last_layer_string_results = current_layer_string_results_for_next_task
            
            if i < self._num_layers - 1:
                current_worker_task = WorkerVQATask(
                    task=message.task, 
                    question_id=message.question_id,
                    image_url=None,
                    previous_results=current_layer_string_results_for_next_task
                )
        
        final_aggregated_answer = "[Error_AggregationNotReached]"
        if orchestrator_error:
            final_aggregated_answer = f"[Error_PipelineStoppedEarly:{orchestrator_error}]"
        else:
            self.logger.info(f"QID {message.question_id}: Performing final aggregation based on {len(last_layer_string_results)} results from last worker layer.")
            valid_last_layer_results = [r for r in last_layer_string_results if r and not r.startswith("[")]

            if not valid_last_layer_results:
                self.logger.error(f"QID {message.question_id}: No valid results from the last worker layer for final aggregation.")
                orchestrator_error = "NoValidResultsForFinalAggregation"
                orchestrator_stop_reason = "FinalAggregation_NoValidInputs"
                final_aggregated_answer = "[Error:NoValidResultsForFinalAggregation]"
            else:
                final_system_prompt_text = load_mixture_prompt_template("orchestrator_final_aggregation.j2", valid_last_layer_results=valid_last_layer_results)
                if "Critical Error" in final_system_prompt_text:
                     self.logger.error(f"QID {message.question_id} - Failed to load orchestrator final aggregation prompt.")
                     orchestrator_error = "PromptLoadingError_FinalAggregation"
                     orchestrator_stop_reason = "FinalAggregation_PromptError"
                     final_aggregated_answer = "[Error_PromptLoadingFailure_FinalAggregation]"
                else:
                    final_user_prompt_text = f"Original Question: {message.task}"
                    final_vlm_content_list_agg = [{"type": "text", "text": final_user_prompt_text}]
                    
                    messages_for_client_agg = [
                        {"role": "system", "content": final_system_prompt_text},
                        {"role": "user", "content": final_vlm_content_list_agg}
                    ]
                    client_params_agg = {
                        "messages": messages_for_client_agg
                    }

                    self.logger.debug(f"QID {message.question_id} - Final Aggregation System Prompt: {final_system_prompt_text[:300]}...")
                    self.logger.debug(f"QID {message.question_id} - Final Aggregation User Content: {final_user_prompt_text[:100]}...")
                    
                    try:
                        final_model_result = await self._model_client.create(**client_params_agg)
                        if isinstance(final_model_result.content, str) and final_model_result.content.strip():
                            final_aggregated_answer = final_model_result.content.strip()
                            orchestrator_stop_reason = "Completed"
                        else:
                            final_aggregated_answer = "[EmptyFinalAggregationOutput]"
                            orchestrator_error = "EmptyFinalAggregationOutput"
                            orchestrator_stop_reason = "FinalAggregation_EmptyOutput"
                    except Exception as e_agg:
                        self.logger.error(f"QID {message.question_id} - Error during final VLM aggregation call: {e_agg}", exc_info=True)
                        orchestrator_error = f"FinalAggregationException: {str(e_agg)[:100]}"
                        final_aggregated_answer = "[Error_FinalAggregationFailed]"
                        orchestrator_stop_reason = "ErrorInFinalAggregation"
            
        self.logger.info(f"QID {message.question_id} - Final Aggregated Answer: '{final_aggregated_answer[:100]}...'")
        
        end_time_orchestrator = time.time()
        return FinalVQAResult(
            question_id=message.question_id, image_path=message.image_url, question=message.task, target_answers=message.target_answer, question_type=message.question_type,
            final_aggregated_answer=final_aggregated_answer,
            all_worker_outputs_by_layer=all_layers_outputs_for_final_result,
            processing_time_seconds=round(end_time_orchestrator - start_time_orchestrator, 2),
            error=orchestrator_error,
            stop_reason=orchestrator_stop_reason
        )