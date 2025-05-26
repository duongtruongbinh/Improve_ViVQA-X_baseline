import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, AsyncGenerator
import os
import json
import sys
import traceback

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage, ChatCompletionClient
from openai import OpenAI

APP_CONFIG_VLM_API_PROVIDER = "vllm"
APP_CONFIG_VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
APP_CONFIG_VLM_API_KEY = "EMPTY"
APP_CONFIG_VLM_URL = "http://localhost:8005/v1"
APP_CONFIG_VLM_TEMPERATURE = 0.2

APP_CONFIG_MIXTURE_NUM_LAYERS = 2
APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER = 3

APP_CONFIG_DATASET_NAME = "vqa-v2"
APP_CONFIG_DATASET_SPLIT = "val"
APP_CONFIG_VQA_IMAGES_DIR = "/mnt/VLAI_data/COCO_Images/val2014"
APP_CONFIG_VQA_QUESTIONS_FILE = "/mnt/VLAI_data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json"
APP_CONFIG_VQA_ANNOTATIONS_FILE = "/mnt/VLAI_data/VQAv2/v2_mscoco_val2014_annotations.json"
APP_CONFIG_MAX_QUESTIONS_TO_LOAD = 5
APP_CONFIG_EVAL_F1_THRESHOLD = 0.5

IMAGE_UTILS_AVAILABLE = False
try:
    from .image_utils import process_image_for_vlm_agent
    IMAGE_UTILS_AVAILABLE = True
    sys.stdout.write("INFO_MIXTURE: image_utils.process_image_for_vlm_agent imported successfully using relative import.\n")
except ImportError as e_img_utils:
    sys.stderr.write(f"CRITICAL_ERROR_MIXTURE: Could not import 'process_image_for_vlm_agent' from '.image_utils'. Error: {e_img_utils}\n")
    sys.stderr.write("Ensure 'image_utils.py' is in the same directory (src/) and 'src' is treated as a package (contains __init__.py).\n")
    def process_image_for_vlm_agent(image_path_or_cv2_image, app_config_ref, bbox=None):
        sys.stderr.write("CRITICAL_ERROR_MIXTURE: Placeholder 'process_image_for_vlm_agent' called.\n")
        return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

# Định nghĩa VQA_V2_ANSWER_TYPE cục bộ thay vì import
VQA_V2_ANSWER_TYPE = "vqa_standard"
EVALUATION_FUNCTIONS_AVAILABLE = False
try:
    from .evaluation import perform_direct_accuracy_check, clean_answer_for_comparison, preprocess_model_answer_for_eval
    # VQA_V2_ANSWER_TYPE không còn được import từ đây
    EVALUATION_FUNCTIONS_AVAILABLE = True
    sys.stdout.write("INFO_MIXTURE: Evaluation functions imported successfully from .evaluation using relative import.\n")
except ImportError as e_eval:
    sys.stderr.write(f"WARNING_MIXTURE: Could not import from .evaluation. Error: {e_eval}. Accuracy checking will be limited or skipped.\n")
    def perform_direct_accuracy_check(final_answer_text, target_answers, question_type, **kwargs):
        return {"is_correct": None, "notes": "Evaluation skipped due to import error"}, [], ""
    def clean_answer_for_comparison(text): return text.strip().lower() if isinstance(text, str) else ""
    def preprocess_model_answer_for_eval(text, q_type): return clean_answer_for_comparison(text)

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

class VLLMChatCompletionClient(ChatCompletionClient):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI(
            api_key=config.get("api_key", APP_CONFIG_VLM_API_KEY),
            base_url=config.get("base_url", APP_CONFIG_VLM_URL),
        )
        self.model = config.get("model", APP_CONFIG_VLM_MODEL_NAME)
        self.temperature = config.get("temperature", APP_CONFIG_VLM_TEMPERATURE)
        self.max_tokens_default = config.get("max_tokens_generation", 70)
        self.vqa_system_instruction = (
            "You are a Visual Question Answering (VQA) system. "
            "Use only the information visible in the image. "
            "Answer each question with a single word or short phrase whenever possible. "
            "Always use exactly this output format, with no extra text:\n\n"
            "Answer: <your concise answer>"
        )
        self._last_token_usage: Union[Dict[str, int], None] = None
        self._cumulative_token_usage: Dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

    async def create(
        self,
        messages: List[Union[SystemMessage, UserMessage, AssistantMessage]],
        **kwargs,
    ) -> AssistantMessage:
        image_path = kwargs.get("image_path_for_create")
        if not image_path:
            return AssistantMessage(content="Error: Image path missing in create kwargs.", source="client_create", actual_usage=None)
        if not IMAGE_UTILS_AVAILABLE:
            return AssistantMessage(content="Error: Image processing unavailable in create.", source="client_create", actual_usage=None)
        if not os.path.exists(image_path):
            return AssistantMessage(content=f"Error: Image not found at {image_path} in create.", source="client_create", actual_usage=None)

        try:
            base64_image_data_url = process_image_for_vlm_agent(image_path, app_config_ref=None)
        except Exception as e:
            return AssistantMessage(content=f"Error processing image {image_path}: {e}", source="client_create", actual_usage=None)

        vllm_api_messages = []
        final_question_text = ""
        additional_system_prompts = []

        for msg in messages:
            if isinstance(msg, SystemMessage): additional_system_prompts.append(str(msg.content))
            elif isinstance(msg, UserMessage): final_question_text = str(msg.content)

        current_system_instruction = self.vqa_system_instruction
        if additional_system_prompts:
            current_system_instruction = "\n".join(additional_system_prompts)

        vllm_api_messages.append({"role": "system", "content": current_system_instruction})
        user_content_blocks = [
            {"type": "image_url", "image_url": {"url": base64_image_data_url}},
            {"type": "text", "text": final_question_text},
        ]
        vllm_api_messages.append({"role": "user", "content": user_content_blocks})
        vllm_api_messages.append({"role": "assistant", "content": "Answer:"})

        token_usage = None
        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=vllm_api_messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens_default),
                temperature=self.temperature, stop=["\n\n", "\nUser:", "\nSystem:", "\nQuestion:"]
            )
            raw_answer = completion.choices[0].message.content.strip()
            processed_answer = raw_answer.split("\n")[0].strip()
            if processed_answer.lower().startswith("answer:"):
                processed_answer = processed_answer.split(":", 1)[-1].strip()

            if completion.usage:
                token_usage = {
                    "prompt_tokens": completion.usage.prompt_tokens or 0,
                    "completion_tokens": completion.usage.completion_tokens or 0,
                    "total_tokens": completion.usage.total_tokens or 0,
                }
                self._last_token_usage = token_usage
                self._cumulative_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                self._cumulative_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                self._cumulative_token_usage["total_tokens"] += token_usage["total_tokens"]
            else: self._last_token_usage = None
            return AssistantMessage(content=processed_answer, source="model", actual_usage=token_usage)
        except Exception as e:
            self._last_token_usage = None
            sys.stderr.write(f"ERROR_VLLM_CLIENT_CREATE: API call failed: {e}\n")
            return AssistantMessage(content=f"Error in VLLM API: {str(e)}", source="api_error", actual_usage=None)

    async def create_stream(self, messages: List[Union[SystemMessage, UserMessage, AssistantMessage]], **kwargs,) -> AsyncGenerator[AssistantMessage, None]:
        image_path = kwargs.get("image_path_for_create")
        if not image_path:
            yield AssistantMessage(content="Error: Image path missing for streaming.", source="client_stream", actual_usage=None); return
        sys.stderr.write("WARNING_VLLM_CLIENT: create_stream called but not fully implemented, using fallback.\n")
        response = await self.create(messages, **kwargs)
        yield response

    def count_tokens(self, messages: Union[List[Dict[str, Any]], List[Union[SystemMessage, UserMessage, AssistantMessage]], str], **kwargs) -> int:
        text_content = ""
        if isinstance(messages, str): text_content = messages
        elif isinstance(messages, list):
            current_messages = messages;
            if not current_messages: return 0
            if isinstance(current_messages[0], (SystemMessage, UserMessage, AssistantMessage)):
                for msg_obj in current_messages:
                    if isinstance(msg_obj.content, str): text_content += msg_obj.content + " "
                    elif isinstance(msg_obj.content, list):
                        for item_content in msg_obj.content:
                            if isinstance(item_content, dict) and item_content.get("type") == "text":
                                text_content += item_content.get("text", "") + " "
            elif isinstance(current_messages[0], dict):
                 for msg_dict in current_messages:
                    content = msg_dict.get("content")
                    if isinstance(content, str): text_content += content + " "
                    elif isinstance(content, list):
                        for item_content in content:
                            if isinstance(item_content, dict) and item_content.get("type") == "text":
                                text_content += item_content.get("text", "") + " "
        return len(text_content.strip().split())

    @property
    def model_info(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": str(self.client.base_url),
                "temperature": self.temperature, "max_tokens_default": self.max_tokens_default}
    @property
    def capabilities(self) -> Dict[str, Any]:
        return {"tools": False, "streaming": True, "vision": True}
    @property
    def actual_usage(self) -> Union[Dict[str, int], None]: return self._last_token_usage
    @property
    def total_usage(self) -> Union[Dict[str, int], None]: return self._cumulative_token_usage
    @property
    def remaining_tokens(self) -> Union[Dict[str, int], None]: return None
    async def close(self) -> None: pass

class VQAAgentLogicBase:
    async def _prepare_and_call_vlm_client(
        self, model_client: VLLMChatCompletionClient, question_text: str,
        image_path: str, system_prompt_parts: List[str] = None
    ) -> AssistantMessage:
        autogen_messages_for_client = []
        if system_prompt_parts:
            autogen_messages_for_client.append(SystemMessage(content="\n".join(system_prompt_parts), source="system")) # Thêm source cho SystemMessage
        autogen_messages_for_client.append(UserMessage(content=question_text, source="user")) # Thêm source="user"
        return await model_client.create(messages=autogen_messages_for_client, image_path_for_create=image_path)

class VQAWorkerAgent(RoutedAgent, VQAAgentLogicBase):
    def __init__(self, model_client: VLLMChatCompletionClient) -> None:
        super().__init__(description="VQA Worker Agent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: VQAWorkerTask, ctx: MessageContext) -> WorkerTaskResult:
        system_prompts = []
        if message.previous_results:
            synthesis_prompt = (
                "You are an expert Visual Question Answering agent. You have been provided with a question about an image, "
                "and a set of responses from other agents who attempted to answer the same question based on the same image. "
                "Your task is to critically evaluate these previous responses and the image itself to generate a refined, "
                "concise, and accurate answer to the original question. "
                "Focus on correcting any errors or biases from previous responses and provide the best possible single-word or short-phrase answer.\n\n"
                "Original Question: " + message.task + "\n\n"
                "Previous Responses to Consider:\n" +
                "\n".join([f"- \"{r}\"" for r in message.previous_results]) +
                "\n\nBased on the image and the original question, and considering these previous responses, what is your refined answer? Ensure your output starts with 'Answer:'"
            )
            system_prompts.append(synthesis_prompt)
        assistant_response_msg = await self._prepare_and_call_vlm_client(
            model_client=self._model_client, question_text=message.task,
            image_path=message.image_path, system_prompt_parts=system_prompts
        )
        processed_answer = assistant_response_msg.content
        return WorkerTaskResult(result=processed_answer, worker_id=self.id)

class VQAOrchestratorAgent(RoutedAgent, VQAAgentLogicBase):
    def __init__(self, model_client: VLLMChatCompletionClient, worker_agent_types: List[str], num_layers: int) -> None:
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
            except asyncio.TimeoutError: sys.stderr.write(f"ERROR_MIXTURE_ORCH: Timeout Layer {i}.\n")
            except Exception as e: sys.stderr.write(f"ERROR_MIXTURE_ORCH: Gather Layer {i}: {e}\n")
            layer_actual_results = []
            for res_idx, res_obj in enumerate(raw_worker_outputs):
                wid_str = layer_worker_ids[res_idx].key if res_idx < len(layer_worker_ids) else f"unk_w_{res_idx}"
                if isinstance(res_obj, WorkerTaskResult) and res_obj.result and res_obj.result.strip():
                    layer_actual_results.append(res_obj.result)
                    sys.stdout.write(f"  L{i}-Worker '{res_obj.worker_id.key}': '{res_obj.result}'\n")
                else: sys.stderr.write(f"  L{i}-Worker '{wid_str}': (bad result: {type(res_obj)})\n")
            current_task_for_workers = VQAWorkerTask(
                task=message.task, image_path=message.image_path,
                previous_results=layer_actual_results if layer_actual_results else []
            )
        sys.stdout.write(f"Orchestrator: Final aggregation with {len(current_task_for_workers.previous_results)} results.\n")
        final_agg_system_prompts = []
        if not current_task_for_workers.previous_results:
            final_agg_system_prompts.append(self._model_client.vqa_system_instruction + "\nDirectly answer the question based on the image. Start your answer with 'Answer:'.")
        else:
            final_synthesis_prompt = (
                "You are the final decision-making Visual Question Answering expert. "
                "Synthesize inputs (image, question, worker responses) into one definitive, concise, high-quality answer. "
                "Critically evaluate worker responses. Your final answer should be a single word or short phrase. Ensure your output starts with 'Answer:'\n\n"
                "Original Question: " + message.task + "\n\n"
                "Worker Responses:\n" + "\n".join([f"- \"{r}\"" for r in current_task_for_workers.previous_results]) +
                "\n\nFinal Answer based on all information:"
            )
            final_agg_system_prompts.append(final_synthesis_prompt)
        final_model_response_msg = await self._prepare_and_call_vlm_client(
            model_client=self._model_client, question_text=message.task,
            image_path=message.image_path, system_prompt_parts=final_agg_system_prompts
        )
        final_answer = final_model_response_msg.content
        sys.stdout.write(f"Orchestrator: Final Aggregated Answer (QID {message.question_id}): '{final_answer}'\n{'-'*50}\n")
        return VQAFinalResult(result=final_answer, question_id=message.question_id)

async def run_single_vqa_moa_task(runtime: SingleThreadedAgentRuntime, task_info: Dict, orch_id_key: str) -> Union[VQAFinalResult, None]:
    if not IMAGE_UTILS_AVAILABLE: sys.stderr.write(f"CRIT_RUN_TASK: No image_utils for QID {task_info.get('qid')}.\n"); return None
    if not os.path.exists(task_info["image_path"]): sys.stderr.write(f"ERR_RUN_TASK: No image {task_info['image_path']} for QID {task_info.get('qid')}.\n"); return None
    user_vqa_task = VQAUserTask(
        task=task_info["question"], image_path=task_info["image_path"], question_id=task_info.get("qid"),
        ground_truth_answers_struct=task_info.get("gt_answers_struct", []),
        answer_type=task_info.get("answer_type", VQA_V2_ANSWER_TYPE)
    )
    final_result_msg = None
    try:
        final_result_msg = await asyncio.wait_for(runtime.send_message(user_vqa_task, AgentId("vqa_orchestrator", orch_id_key)), timeout=240.0)
    except asyncio.TimeoutError: sys.stderr.write(f"ERR_RUN_TASK: Timeout QID {task_info.get('qid')}.\n"); return None
    except Exception as e: sys.stderr.write(f"ERR_RUN_TASK: Send/process QID {task_info.get('qid')}: {e}\n"); return None
    return final_result_msg if isinstance(final_result_msg, VQAFinalResult) else None

def load_vqa_data_from_config():
    tasks = []
    questions_path = APP_CONFIG_VQA_QUESTIONS_FILE
    annotations_path = APP_CONFIG_VQA_ANNOTATIONS_FILE
    images_dir = APP_CONFIG_VQA_IMAGES_DIR

    if not (os.path.exists(questions_path) and os.path.exists(annotations_path) and os.path.exists(images_dir)):
        sys.stderr.write(f"WARNING_MIXTURE_LOAD_DATA: One or more VQA data paths not found. Q: {questions_path}, A: {annotations_path}, I: {images_dir}. Using example tasks only.\n")
        return []
    try:
        with open(questions_path, 'r', encoding='utf-8') as f_q:
            q_data = json.load(f_q).get('questions', [])
        with open(annotations_path, 'r', encoding='utf-8') as f_a:
            a_data_list = json.load(f_a).get('annotations', [])
        answers_by_qid = {ann['question_id']: ann for ann in a_data_list}
        processed_q_ids = set()

        for q_item in q_data:
            if APP_CONFIG_MAX_QUESTIONS_TO_LOAD is not None and len(tasks) >= APP_CONFIG_MAX_QUESTIONS_TO_LOAD:
                break
            qid = q_item.get('question_id')
            if not qid or qid in processed_q_ids: continue
            img_id = q_item.get('image_id')
            img_filename = f"COCO_{APP_CONFIG_DATASET_SPLIT}2014_{img_id:012d}.jpg"
            full_image_path = os.path.join(images_dir, img_filename)

            if os.path.exists(full_image_path):
                ann_item = answers_by_qid.get(qid)
                if ann_item:
                    gt_struct = [{"answer": ans_entry['answer']} for ans_entry in ann_item.get('answers', []) if 'answer' in ans_entry]
                    answer_type_from_ann = ann_item.get('answer_type', VQA_V2_ANSWER_TYPE).lower()
                    tasks.append({
                        "qid": qid, "question": q_item.get('question'), "image_path": full_image_path,
                        "answer_type": answer_type_from_ann, "gt_answers_struct": gt_struct
                    })
                    processed_q_ids.add(qid)
        sys.stdout.write(f"INFO_MIXTURE_LOAD_DATA: Loaded {len(tasks)} tasks from VQA config paths.\n")
    except Exception as e:
        sys.stderr.write(f"ERROR_MIXTURE_LOAD_DATA: Failed to load VQA data: {e}\n")
        traceback.print_exc()
        return []
    return tasks

async def main_mixture_vqa_flow():
    sys.stdout.write("INFO_MIXTURE_MAIN: Starting VQA Mixture of Agents flow...\n")
    if not IMAGE_UTILS_AVAILABLE:
        sys.stderr.write("CRIT_MIXTURE_MAIN: Exiting: image_utils not available.\n"); return
    vllm_client_config_params = {
        "api_key": APP_CONFIG_VLM_API_KEY, "base_url": APP_CONFIG_VLM_URL,
        "model": APP_CONFIG_VLM_MODEL_NAME, "temperature": APP_CONFIG_VLM_TEMPERATURE,
        "max_tokens_generation": 70
    }
    shared_vlm_model_client = VLLMChatCompletionClient(config=vllm_client_config_params)
    ORCHESTRATOR_ID_KEY = "vqa_orch_main_standalone"
    tasks_to_run = []
    loaded_tasks = load_vqa_data_from_config()

    if loaded_tasks:
        tasks_to_run = loaded_tasks
    else:
        sys.stdout.write("INFO_MIXTURE_MAIN: No tasks loaded from config, using fallback example tasks.\n")
        sample_image_filename = "test_vqa_sample_image.jpg"
        if not os.path.exists(sample_image_filename) and IMAGE_UTILS_AVAILABLE:
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (250,150),color='skyblue'); d=ImageDraw.Draw(img)
                d.text((10,10),"Sample VQA",fill='black'); img.save(sample_image_filename)
                sys.stdout.write(f"INFO_MIXTURE_MAIN: Created sample image: {sample_image_filename}\n")
            except Exception as e_pil: sys.stderr.write(f"WARN_MIXTURE_MAIN: Pillow error: {e_pil}\n")
        if os.path.exists(sample_image_filename):
             tasks_to_run = [{"qid": "sample001", "question": "What is written?", "image_path": sample_image_filename, "answer_type": "other", "gt_answers_struct": [{"answer": "sample vqa"}]}]
        else:
            sys.stderr.write("WARN_MIXTURE_MAIN: No tasks from config and sample image failed. No tasks to run.\n")

    if not tasks_to_run: sys.stderr.write("CRIT_MIXTURE_MAIN: No tasks to process. Exiting.\n"); return

    runtime = SingleThreadedAgentRuntime()
    await VQAWorkerAgent.register(runtime, "vqa_worker", lambda: VQAWorkerAgent(model_client=shared_vlm_model_client))
    await VQAOrchestratorAgent.register(runtime, "vqa_orchestrator", lambda: VQAOrchestratorAgent(
            model_client=shared_vlm_model_client,
            worker_agent_types=["vqa_worker"] * APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER,
            num_layers=APP_CONFIG_MIXTURE_NUM_LAYERS ))
    runtime.start()
    final_results_for_all_tasks = []

    for vqa_task_detail in tasks_to_run:
        if not vqa_task_detail.get("image_path") or not os.path.exists(vqa_task_detail["image_path"]):
            sys.stderr.write(f"WARN_MIXTURE_MAIN: Skip QID {vqa_task_detail.get('qid')} invalid path: {vqa_task_detail.get('image_path')}\n"); continue
        moa_result = await run_single_vqa_moa_task(runtime, vqa_task_detail, ORCHESTRATOR_ID_KEY)
        gt_ans_flat = [item['answer'] for item in vqa_task_detail.get("gt_answers_struct", [])]
        entry = {
            "question_id": vqa_task_detail.get("qid"), "question_text": vqa_task_detail.get("question"),
            "image_path": vqa_task_detail.get("image_path"),
            "predicted_answer_moa": moa_result.result if moa_result else "[MOA_FAILED]",
            "ground_truth_answers": gt_ans_flat, "answer_type": vqa_task_detail.get("answer_type", VQA_V2_ANSWER_TYPE),
            "is_correct_direct": None, "evaluation_notes": "Eval not run or failed"
        }
        if moa_result and EVALUATION_FUNCTIONS_AVAILABLE:
            pred_ans_cleaned_for_eval = preprocess_model_answer_for_eval(moa_result.result, entry["answer_type"])
            eval_res, _, _ = perform_direct_accuracy_check(final_answer_text=pred_ans_cleaned_for_eval, target_answers=gt_ans_flat, question_type=entry["answer_type"], f1_threshold_other=APP_CONFIG_EVAL_F1_THRESHOLD)
            entry["is_correct_direct"] = eval_res.get("is_correct"); entry["evaluation_notes"] = eval_res.get("notes")
            if eval_res.get("vqa_score") is not None: entry["vqa_score"] = eval_res.get("vqa_score")
        final_results_for_all_tasks.append(entry)

    await runtime.stop_when_idle()
    if hasattr(shared_vlm_model_client, 'close') and asyncio.iscoroutinefunction(shared_vlm_model_client.close):
        await shared_vlm_model_client.close()
    sys.stdout.write(f"\n{'-'*20} VQA MoA Flow Completed {'-'*20}\n")

    if final_results_for_all_tasks:
        output_dir = "results_moa_vqa_standalone"; os.makedirs(output_dir, exist_ok=True)
        model_slug = APP_CONFIG_VLM_MODEL_NAME.split('/')[-1].replace('-','_')
        ts = int(asyncio.get_event_loop().time())
        res_fname = f"moa_vqa_{model_slug}_L{APP_CONFIG_MIXTURE_NUM_LAYERS}xW{APP_CONFIG_MIXTURE_NUM_WORKERS_PER_LAYER}_{ts}.json"
        full_res_path = os.path.join(output_dir, res_fname)
        with open(full_res_path, 'w', encoding='utf-8') as f: json.dump(final_results_for_all_tasks, f, indent=2, ensure_ascii=False)
        sys.stdout.write(f"INFO_MIXTURE_MAIN: Results saved to: {full_res_path}\n")
        if EVALUATION_FUNCTIONS_AVAILABLE:
            correct = sum(1 for r in final_results_for_all_tasks if r.get("is_correct_direct"))
            total = len(final_results_for_all_tasks)
            vqa_scores = [r.get("vqa_score") for r in final_results_for_all_tasks if r.get("vqa_score") is not None]
            avg_vqa = sum(vqa_scores)/len(vqa_scores) if vqa_scores else None
            sys.stdout.write(f"\n--- Eval Summary ---\nTotal: {total}, Correct: {correct} ({ (correct/total*100) if total else 0 :.2f}%)\n")
            if avg_vqa is not None: sys.stdout.write(f"Avg VQA Score: {avg_vqa:.4f}\n")
    else: sys.stdout.write("INFO_MIXTURE_MAIN: No results generated.\n")

if __name__ == "__main__":
    if not IMAGE_UTILS_AVAILABLE:
        sys.stderr.write("CRIT_MIXTURE_LAUNCH: Cannot run without image_utils.py. Please ensure it is in the 'src' directory and has no import errors itself.\n")
        sys.exit(1)
    try:
        asyncio.run(main_mixture_vqa_flow())
    except KeyboardInterrupt: sys.stdout.write("\nINFO_MIXTURE_LAUNCH: Interrupted by user.\n")
    except Exception as e_main:
        sys.stderr.write(f"CRIT_MIXTURE_LAUNCH: Unhandled error: {e_main}\n")
        traceback.print_exc()
        sys.exit(1)