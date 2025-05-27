import asyncio
import os
import sys
import json
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, AsyncGenerator

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage

# Import local modules
try:
    from .config_loader import app_config
    from .vllm_clients import vlm_client_vllm
    from .image_utils import process_image_for_vlm_agent
    from .evaluation import perform_direct_accuracy_check, clean_answer_for_comparison, preprocess_model_answer_for_eval
    from .dataloader import VQAv2Dataset, GQADataset
except ImportError as e:
    sys.stderr.write(f"CRITICAL_ERROR_MIXTURE: Failed to import required modules: {e}\n")
    sys.exit(1)

# Constants from config
VQA_CONFIG = app_config.get("datasets", {})
MIXTURE_CONFIG = app_config.get("vqa_mixture_settings", {})
INFERENCE_CONFIG = app_config.get("inference_settings", {})

# VQA Dataset Configuration
APP_CONFIG_DATASET_NAME = VQA_CONFIG.get("dataset_name", "vqa-v2")
APP_CONFIG_DATASET_SPLIT = VQA_CONFIG.get("vqa_v2_dataset_split" if APP_CONFIG_DATASET_NAME == "vqa-v2" else "gqa_dataset_split", "rest-val")

# Mixture of Agents Configuration
APP_CONFIG_MIXTURE_NUM_LAYERS = MIXTURE_CONFIG.get("num_layers", 2)
APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER = MIXTURE_CONFIG.get("num_workers_per_layer", 3)

# Inference Settings
APP_CONFIG_MAX_QUESTIONS_TO_LOAD = VQA_CONFIG.get("num_test_data") if VQA_CONFIG.get("use_num_test_data", False) else None
APP_CONFIG_EVAL_F1_THRESHOLD = INFERENCE_CONFIG.get("f1_threshold_for_other_type", 0.5)

VQA_V2_ANSWER_TYPE = "vqa_standard"

# Data classes
@dataclass
class VQAUserTask:
    task: str
    image_path: str
    question_id: Any = None
    ground_truth_answers_struct: List[Dict[str, str]] = field(default_factory=list)
    answer_type: str = "other"

@dataclass
class VQAWorkerTask:
    task: str
    image_path: str
    previous_results: List[str] = field(default_factory=list)

@dataclass
class WorkerTaskResult:
    result: str
    worker_id: AgentId

@dataclass
class VQAFinalResult:
    result: str
    question_id: Any = None

# Agent Classes
class VQAAgentLogicBase:
    async def _prepare_and_call_vlm_client(
        self, model_client: Any, question_text: str,
        image_path: str, system_prompt_parts: List[str] = None
    ) -> AssistantMessage:
        autogen_messages_for_client = []
        if system_prompt_parts and any(sp.strip() for sp in system_prompt_parts):
            autogen_messages_for_client.append(SystemMessage(content="\n".join(system_prompt_parts), source="system"))
        autogen_messages_for_client.append(UserMessage(content=question_text, source="user"))
        return await model_client.create(messages=autogen_messages_for_client, image_path_for_create=image_path)

class VQAWorkerAgent(RoutedAgent, VQAAgentLogicBase):
    def __init__(self, model_client: Any) -> None:
        super().__init__(description="VQA Worker Agent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: VQAWorkerTask, ctx: MessageContext) -> WorkerTaskResult:
        system_prompts = []
        if not message.previous_results:
            try:
                worker_id_suffix = ctx.sender.name.split('/')[-1]
                worker_index = int(worker_id_suffix.split('_')[-1])
            except:
                worker_index = 0
            
            base_instruction = self._model_client.vqa_system_instruction_template
            
            if worker_index == 0:
                system_prompts.append(base_instruction +
                                    "\nFocus on identifying the primary objects and their immediate visible properties. Be very direct and literal in your answer.")
            elif worker_index == 1:
                system_prompts.append(base_instruction +
                                    "\nConsider the actions, interactions, and the overall scene context. What is generally happening or depicted? Provide a concise interpretation.")
            else:
                system_prompts.append(base_instruction +
                                    "\nPay close attention to details, text (if any), and less obvious elements. What specific details can you extract that might be relevant?")
        
        elif message.previous_results:
            synthesis_prompt = (
                "You are an expert VQA synthesis agent. Your task is to refine an answer based on previous attempts. "
                "Original question: '{orig_q}'.\n"
                "Previous answers from other agents:\n{prev_ans_formatted}\n\n"
                "Critically evaluate these previous answers against the image (which is the primary source of truth). "
                "If they agree and are correct, confirm the answer. If they differ or are incorrect, "
                "provide a new, more accurate single-word or short-phrase answer. "
                "Format your response as: Answer: <your refined answer>"
            ).format(
                orig_q=message.task,
                prev_ans_formatted="\n".join([f"- \"{r}\"" for r in message.previous_results])
            )
            system_prompts.append(synthesis_prompt)

        assistant_response_msg = await self._prepare_and_call_vlm_client(
            model_client=self._model_client, question_text=message.task,
            image_path=message.image_path, system_prompt_parts=system_prompts
        )
        processed_answer = assistant_response_msg.content
        return WorkerTaskResult(result=processed_answer, worker_id=self.id)

class VQAOrchestratorAgent(RoutedAgent, VQAAgentLogicBase):
    def __init__(self, model_client: Any, worker_agent_types: List[str], num_layers: int) -> None:
        super().__init__(description="VQA Orchestrator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers

    @message_handler
    async def handle_task(self, message: VQAUserTask, ctx: MessageContext) -> VQAFinalResult:
        sys.stdout.write(f"\n{'-'*20} Orchestrator-{self.id}: New VQA Task {'-'*20}\n")
        sys.stdout.write(f"Orchestrator: QID: {message.question_id}, Image: '{message.image_path}', Question: '{message.task}'\n")
        
        current_task_for_workers = VQAWorkerTask(
            task=message.task, image_path=message.image_path, previous_results=[]
        )
        
        for i in range(self._num_layers):
            layer_worker_ids = [AgentId(wt, f"{self.id.key}/layer_{i}/worker_{j}") for j, wt in enumerate(self._worker_agent_types)]
            sys.stdout.write(f"Orchestrator: Dispatching to Layer {i} Workers: {[wid.key for wid in layer_worker_ids]}\n")
            layer_results_futures = [self.send_message(current_task_for_workers, wid) for wid in layer_worker_ids]
            
            raw_worker_outputs = []
            try:
                raw_worker_outputs = await asyncio.gather(*[asyncio.wait_for(f, timeout=120.0) for f in layer_results_futures])
            except asyncio.TimeoutError:
                sys.stderr.write(f"ERROR_MIXTURE_ORCH: Timeout Layer {i} for QID {message.question_id}.\n")
            except Exception as e:
                sys.stderr.write(f"ERROR_MIXTURE_ORCH: Gather Layer {i} for QID {message.question_id}: {e}\n")

            layer_actual_results = []
            for res_idx, res_obj in enumerate(raw_worker_outputs):
                wid_str = layer_worker_ids[res_idx].key if res_idx < len(layer_worker_ids) else f"unk_w_{res_idx}"
                if isinstance(res_obj, WorkerTaskResult) and res_obj.result and res_obj.result.strip():
                    layer_actual_results.append(res_obj.result)
                    sys.stdout.write(f"  L{i}-Worker '{res_obj.worker_id.key}': '{res_obj.result}'\n")
                else:
                    sys.stderr.write(f"  L{i}-Worker '{wid_str}': (bad result type: {type(res_obj)} or empty for QID {message.question_id})\n")
            
            current_task_for_workers = VQAWorkerTask(
                task=message.task, image_path=message.image_path,
                previous_results=layer_actual_results if layer_actual_results else []
            )

        sys.stdout.write(f"Orchestrator: Final aggregation with {len(current_task_for_workers.previous_results)} results for QID {message.question_id}.\n")
        final_agg_system_prompts = []
        
        if not current_task_for_workers.previous_results:
            sys.stdout.write(f"Orchestrator: No previous worker results for QID {message.question_id}. Orchestrator generating answer directly.\n")
        else:
            final_synthesis_prompt = (
                "You are the final decision-making Visual Question Answering expert. "
                "You have received several answers from different worker agents for the question: '{orig_q}'.\n"
                "Worker Agent Responses to Synthesize:\n{prev_ans_formatted}\n\n"
                "Your task is to critically analyze these worker responses in conjunction with the image (the ultimate source of truth). "
                "Determine the single best, most accurate, and concise answer. "
                "If all worker answers appear flawed, derive your own answer directly from the image and question. "
                "Format your final answer as: Answer: <your definitive answer>"
            ).format(
                orig_q=message.task,
                prev_ans_formatted="\n".join([f"- \"{r}\"" for r in current_task_for_workers.previous_results])
            )
            final_agg_system_prompts.append(final_synthesis_prompt)

        final_model_response_msg = await self._prepare_and_call_vlm_client(
            model_client=self._model_client, question_text=message.task,
            image_path=message.image_path, system_prompt_parts=final_agg_system_prompts if final_agg_system_prompts else None
        )
        final_answer = final_model_response_msg.content
        sys.stdout.write(f"Orchestrator: Final Aggregated Answer (QID {message.question_id}): '{final_answer}'\n{'-'*50}\n")
        return VQAFinalResult(result=final_answer, question_id=message.question_id)

# Helper Functions
def load_vqa_data_from_config():
    try:
        dataset_class = GQADataset if APP_CONFIG_DATASET_NAME == 'gqa' else VQAv2Dataset
        dataset = dataset_class(app_config, transform=None)
        return dataset
    except Exception as e:
        sys.stderr.write(f"ERROR_MIXTURE_LOAD_DATA: Failed to load dataset: {e}\n")
        traceback.print_exc()
        return None

async def run_single_vqa_moa_task(runtime: SingleThreadedAgentRuntime, task_info: Dict, orch_id_key: str) -> Union[VQAFinalResult, None]:
    if not os.path.exists(task_info["image_path"]):
        sys.stderr.write(f"ERR_RUN_TASK: No image {task_info['image_path']} for QID {task_info.get('qid')}.\n")
        return None
    
    user_vqa_task = VQAUserTask(
        task=task_info["question"], image_path=task_info["image_path"], question_id=task_info.get("qid"),
        ground_truth_answers_struct=task_info.get("gt_answers_struct", []),
        answer_type=task_info.get("answer_type", VQA_V2_ANSWER_TYPE)
    )
    
    try:
        final_result_msg = await asyncio.wait_for(
            runtime.send_message(user_vqa_task, AgentId("vqa_orchestrator", orch_id_key)),
            timeout=240.0
        )
    except asyncio.TimeoutError:
        sys.stderr.write(f"ERR_RUN_TASK: Timeout QID {task_info.get('qid')}.\n")
        return None
    except Exception as e:
        sys.stderr.write(f"ERR_RUN_TASK: Send/process QID {task_info.get('qid')}: {e}\n")
        traceback.print_exc(file=sys.stderr)
        return None
    
    return final_result_msg if isinstance(final_result_msg, VQAFinalResult) else None

async def main_mixture_vqa_flow():
    sys.stdout.write("INFO_MIXTURE_MAIN: Starting VQA Mixture of Agents flow...\n")
    
    if vlm_client_vllm is None:
        sys.stderr.write("CRIT_MIXTURE_MAIN: Exiting: VLM client not available.\n")
        return

    ORCHESTRATOR_ID_KEY = "vqa_orch_main_standalone"
    
    dataset = load_vqa_data_from_config()
    if dataset is None:
        sys.stderr.write("CRIT_MIXTURE_MAIN: Failed to load dataset. Exiting.\n")
        return

    runtime = SingleThreadedAgentRuntime()
    await VQAWorkerAgent.register(runtime, "vqa_worker", lambda: VQAWorkerAgent(model_client=vlm_client_vllm))
    await VQAOrchestratorAgent.register(runtime, "vqa_orchestrator", lambda: VQAOrchestratorAgent(
        model_client=vlm_client_vllm,
        worker_agent_types=["vqa_worker"] * APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER,
        num_layers=APP_CONFIG_MIXTURE_NUM_LAYERS
    ))
    runtime.start()
    
    final_results_for_all_tasks = []

    for i, data_batch in enumerate(dataset):
        if APP_CONFIG_MAX_QUESTIONS_TO_LOAD is not None and i >= APP_CONFIG_MAX_QUESTIONS_TO_LOAD:
            break

        image_path = data_batch['image_path']
        question_text = data_batch['question']
        question_id = data_batch['question_id']
        target_answer = data_batch['answer']
        answer_type = data_batch.get('answer_type', VQA_V2_ANSWER_TYPE)

        if not os.path.exists(image_path):
            sys.stderr.write(f"WARN_MIXTURE_MAIN: Skip QID {question_id} invalid path: {image_path}\n")
            continue
        
        moa_result = await run_single_vqa_moa_task(runtime, {
            "qid": question_id,
            "question": question_text,
            "image_path": image_path,
            "gt_answers_struct": [{"answer": target_answer}],
            "answer_type": answer_type
        }, ORCHESTRATOR_ID_KEY)
        
        entry = {
            "question_id": question_id,
            "question_text": question_text,
            "image_path": image_path,
            "predicted_answer_moa": moa_result.result if moa_result else "[MOA_FAILED]",
            "ground_truth_answers": [target_answer],
            "answer_type": answer_type,
            "is_correct_direct": None,
            "evaluation_notes": "Eval not run or failed"
        }
        
        if moa_result and moa_result.result != "[MOA_FAILED]":
            pred_ans_for_eval = preprocess_model_answer_for_eval(moa_result.result, entry["answer_type"])
            eval_res, _, _ = perform_direct_accuracy_check(
                final_answer_text=pred_ans_for_eval,
                target_answers=[target_answer],
                question_type=entry["answer_type"],
                f1_threshold_other=APP_CONFIG_EVAL_F1_THRESHOLD
            )
            entry["is_correct_direct"] = eval_res.get("is_correct")
            entry["evaluation_notes"] = eval_res.get("notes")
            if eval_res.get("vqa_score") is not None:
                entry["vqa_score"] = eval_res.get("vqa_score")
        
        final_results_for_all_tasks.append(entry)

    await runtime.stop_when_idle()
    sys.stdout.write(f"\n{'-'*20} VQA MoA Flow Completed {'-'*20}\n")

    if final_results_for_all_tasks:
        output_dir = "results_moa_vqa_standalone"
        os.makedirs(output_dir, exist_ok=True)
        model_slug = app_config.get("vlm_details", {}).get("vlm_model_name", "unknown").split('/')[-1].replace('-','_')
        ts = int(asyncio.get_event_loop().time())
        res_fname = f"moa_vqa_{APP_CONFIG_DATASET_NAME}_{APP_CONFIG_DATASET_SPLIT}_{model_slug}_L{APP_CONFIG_MIXTURE_NUM_LAYERS}xW{APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER}_{ts}.json"
        full_res_path = os.path.join(output_dir, res_fname)
        
        with open(full_res_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_for_all_tasks, f, indent=2, ensure_ascii=False)
        sys.stdout.write(f"INFO_MIXTURE_MAIN: Results saved to: {full_res_path}\n")
        
        # Print evaluation summary
        correct_overall = 0
        total_overall_evaluated = 0
        overall_vqa_scores = []
        
        correct_by_type = {}
        total_by_type = {}
        vqa_scores_by_type = {}

        for r_entry in final_results_for_all_tasks:
            ans_type = r_entry.get("answer_type", "unknown")
            
            total_by_type[ans_type] = total_by_type.get(ans_type, 0) + 1
            if r_entry.get("is_correct_direct") is not None:
                total_overall_evaluated += 1

            if r_entry.get("is_correct_direct") is True:
                correct_overall += 1
                correct_by_type[ans_type] = correct_by_type.get(ans_type, 0) + 1
            
            if r_entry.get("vqa_score") is not None:
                overall_vqa_scores.append(r_entry.get("vqa_score"))
                if ans_type not in vqa_scores_by_type:
                    vqa_scores_by_type[ans_type] = []
                vqa_scores_by_type[ans_type].append(r_entry.get("vqa_score"))

        sys.stdout.write(f"\n--- Overall Evaluation Summary ---\n")
        sys.stdout.write(f"Total Questions Processed: {len(final_results_for_all_tasks)}\n")
        sys.stdout.write(f"Total Questions Evaluated (Direct Match/F1): {total_overall_evaluated}\n")
        if total_overall_evaluated > 0:
            overall_accuracy = (correct_overall / total_overall_evaluated * 100)
            sys.stdout.write(f"Overall Correct: {correct_overall} ({overall_accuracy:.2f}%)\n")
        
        if overall_vqa_scores:
            avg_overall_vqa_score = sum(overall_vqa_scores) / len(overall_vqa_scores)
            sys.stdout.write(f"Average Overall VQA Score: {avg_overall_vqa_score:.4f} (based on {len(overall_vqa_scores)} samples)\n")

        sys.stdout.write(f"\n--- Accuracy by Question Type ---\n")
        sorted_types = sorted(total_by_type.keys())

        for ans_type_key in sorted_types:
            correct_count = correct_by_type.get(ans_type_key, 0)
            total_count = total_by_type[ans_type_key]
            type_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            sys.stdout.write(f"Type: {ans_type_key:<15} | Correct: {correct_count:<5} | Total: {total_count:<5} | Accuracy: {type_accuracy:.2f}%\n")
            
            if ans_type_key in vqa_scores_by_type and vqa_scores_by_type[ans_type_key]:
                avg_vqa_score_type = sum(vqa_scores_by_type[ans_type_key]) / len(vqa_scores_by_type[ans_type_key])
                sys.stdout.write(f"{'':<18} | Avg VQA Score: {avg_vqa_score_type:.4f} (based on {len(vqa_scores_by_type[ans_type_key])} samples)\n")
        sys.stdout.write(f"{'-'*30}\n")
    else:
        sys.stdout.write("INFO_MIXTURE_MAIN: No results generated.\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_mixture_vqa_flow())
    except KeyboardInterrupt:
        sys.stdout.write("\nINFO_MIXTURE_LAUNCH: Interrupted by user.\n")
    except Exception as e_main:
        sys.stderr.write(f"CRIT_MIXTURE_LAUNCH: Unhandled error: {e_main}\n")
        traceback.print_exc()
        sys.exit(1)