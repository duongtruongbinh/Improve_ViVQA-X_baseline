import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --- Imports from your VQA project ---
# Assume these are in the same directory or correctly in PYTHONPATH
# If these files are in a package (e.g., 'src'), adjust imports accordingly.
# For example: from ..image_utils import process_image_for_vlm_agent
try:
    from image_utils import process_image_for_vlm_agent
    from evaluation import perform_direct_accuracy_check, clean_answer_for_comparison, preprocess_model_answer_for_eval
    # A simplified get_question_type, similar to specialized_flow or utils.py
    _yes_no_starters_vqa_seq = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", "am ",
        "can ", "could ", "will ", "would ", "should ",
        "has ", "have ", "had ", "may ", "might ", "must "
    ]
    _number_starters_vqa_seq = ["how many", "what is the number of"]

    def get_question_type_basic(question_text: str) -> str:
        if not isinstance(question_text, str): return "other"
        q_lower = question_text.lower().strip()
        if any(q_lower.startswith(s) for s in _yes_no_starters_vqa_seq): return "yes/no"
        if any(q_lower.startswith(s) for s in _number_starters_vqa_seq): return "number"
        return "other"

except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary VQA utilities: {e}")
    print("Please ensure image_utils.py, evaluation.py are accessible.")
    exit(1)
# --- End VQA project imports ---

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, AssistantMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient # Replace with your VLM client if needed

# 1. Message Protocol for VQA Workflow
@dataclass
class VQAWorkflowMessage:
    question_id: str
    image_path: Optional[str] = None # For the first agent
    image_b64_url: Optional[str] = None # For subsequent agents
    question: str
    question_type: str = "other"
    current_answer: Optional[str] = None
    target_answer_struct: Optional[Any] = None # For evaluation
    evaluation_result: Optional[Dict] = None
    aux_data: Dict[str, Any] = field(default_factory=dict) # For any other info

# 2. Topics
image_prep_topic_type = "ImagePrepAgent"
initial_vqa_topic_type = "InitialVQAAgent"
answer_refinement_topic_type = "AnswerRefinementAgent" # Simplified for this example
eval_format_topic_type = "EvaluationFormattingAgent"
user_presenter_topic_type = "UserPresenterAgent"

# 3. Agents

@type_subscription(topic_type=image_prep_topic_type)
class ImagePrepAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__(description="Processes image and ingests question.")
        # This agent doesn't use an LLM for its core task

    @message_handler
    async def prep_image_and_question(self, message: VQAWorkflowMessage, ctx: MessageContext) -> None:
        print(f"\n----- {self.id.type} ({self.id.key}) QID: {message.question_id} -----")
        print(f"Input: Image Path='{message.image_path}', Question='{message.question[:50]}...'")

        if not message.image_path or not os.path.exists(message.image_path):
            print(f"ERROR: Image path '{message.image_path}' not found or not provided.")
            # Potentially publish an error message or handle gracefully
            # For simplicity, we'll let it fail if image_utils raises an error
            # Or, publish to an error topic/final presenter with error status
            error_msg = VQAWorkflowMessage(
                question_id=message.question_id,
                question=message.question,
                target_answer_struct=message.target_answer_struct,
                current_answer="[ERROR: Image not found]",
                evaluation_result={"error": "Image not found"},
                aux_data={"error_source": self.id.type}
            )
            await self.publish_message(error_msg, topic_id=TopicId(user_presenter_topic_type, source=self.id))
            return

        try:
            b64_url = process_image_for_vlm_agent(message.image_path, app_config_ref=None) # app_config_ref might be needed by your util
            q_type = get_question_type_basic(message.question)

            print(f"Output: Image processed. Q-Type: {q_type}")
            await self.publish_message(
                VQAWorkflowMessage(
                    question_id=message.question_id,
                    image_b64_url=b64_url,
                    question=message.question,
                    question_type=q_type,
                    target_answer_struct=message.target_answer_struct,
                    aux_data=message.aux_data # Pass along any auxiliary data
                ),
                topic_id=TopicId(initial_vqa_topic_type, source=self.id)
            )
        except Exception as e:
            print(f"ERROR in {self.id.type}: {e}")
            error_msg = VQAWorkflowMessage(
                question_id=message.question_id,
                question=message.question,
                target_answer_struct=message.target_answer_struct,
                current_answer=f"[ERROR: Image processing failed: {str(e)[:100]}]",
                evaluation_result={"error": f"Image processing failed: {str(e)[:100]}"},
                aux_data={"error_source": self.id.type}
            )
            await self.publish_message(error_msg, topic_id=TopicId(user_presenter_topic_type, source=self.id))


@type_subscription(topic_type=initial_vqa_topic_type)
class InitialVQAAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__(description="Performs initial VQA.")
        self._model_client = model_client
        # System prompt inspired by simple_direct_flow.py
        self._base_system_message_content = (
            "You are a Visual Question Answering (VQA) system. "
            "Use only the information visible in the image. "
            "Answer the question concisely. "
            "For yes/no questions, answer with 'yes' or 'no'. "
            "For number questions, provide the number. "
            "For other questions, provide a short phrase answer."
            # "Output ONLY the answer text, with no preamble like 'Answer:'." # Adjusted to simplify parsing
        )

    def _get_system_message(self, question_type: str) -> SystemMessage:
        # Could customize system message further based on question_type if needed
        return SystemMessage(content=self._base_system_message_content)

    @message_handler
    async def get_initial_answer(self, message: VQAWorkflowMessage, ctx: MessageContext) -> None:
        print(f"\n----- {self.id.type} ({self.id.key}) QID: {message.question_id} -----")
        print(f"Input: Q='{message.question[:50]}...', Q-Type='{message.question_type}'")

        if not message.image_b64_url:
            print(f"ERROR: No base64 image URL provided to {self.id.type}.")
            error_msg = message # Pass existing message, but update answer/eval
            error_msg.current_answer = "[ERROR: No image data for VLM]"
            error_msg.evaluation_result = {"error": "No image data for VLM"}
            error_msg.aux_data["error_source"] = self.id.type
            await self.publish_message(error_msg, topic_id=TopicId(user_presenter_topic_type, source=self.id))
            return

        system_msg = self._get_system_message(message.question_type)
        user_prompt_content = [
            {"type": "image_url", "image_url": {"url": message.image_b64_url}},
            {"type": "text", "text": message.question},
        ]
        # Adding specific guidance for the model to directly output the answer
        user_prompt_content.append({"type": "text", "text": "\n\nAnswer:"})


        vlm_response = await self._model_client.create(
            messages=[system_msg, UserMessage(content=user_prompt_content)],
            # Common VLM parameters
            max_tokens=50,
            temperature=0.1
        )
        initial_ans = vlm_response.content if isinstance(vlm_response, AssistantMessage) else str(vlm_response)
        initial_ans = initial_ans.strip() # Basic cleaning
        # More sophisticated cleaning could happen in the Refinement Agent

        print(f"Output (Initial Answer): '{initial_ans[:100]}'")
        await self.publish_message(
            VQAWorkflowMessage(
                question_id=message.question_id,
                image_b64_url=message.image_b64_url,
                question=message.question,
                question_type=message.question_type,
                current_answer=initial_ans,
                target_answer_struct=message.target_answer_struct,
                aux_data=message.aux_data
            ),
            topic_id=TopicId(answer_refinement_topic_type, source=self.id) # Next is refinement
        )

@type_subscription(topic_type=answer_refinement_topic_type)
class AnswerRefinementAgent(RoutedAgent):
    def __init__(self, model_client: Optional[ChatCompletionClient] = None) -> None: # LLM is optional
        super().__init__(description="Refines or cleans the VQA answer.")
        self._model_client = model_client # Could use an LLM for more complex refinement
        self._system_message_content = (
            "You are an answer formatting assistant. Given a question, its type, and a raw VQA answer, "
            "refine the answer to be concise and directly responsive to the question type. "
            "For example, if question type is 'yes/no', ensure answer is 'yes' or 'no'. "
            "If question type is 'number', ensure answer is a number. "
            "Remove any extraneous explanation or conversation. Output only the refined answer."
        )


    @message_handler
    async def refine_answer(self, message: VQAWorkflowMessage, ctx: MessageContext) -> None:
        print(f"\n----- {self.id.type} ({self.id.key}) QID: {message.question_id} -----")
        print(f"Input: Initial Ans='{message.current_answer[:50]}', Q-Type='{message.question_type}'")

        refined_ans = message.current_answer # Default to current answer

        # Simple rule-based refinement (can be expanded or use LLM)
        if refined_ans:
            # Use preprocess_model_answer_for_eval for consistent cleaning
            refined_ans = preprocess_model_answer_for_eval(refined_ans, message.question_type)

            # Example: if LLM-based refinement is desired (and model_client is provided)
            # if self._model_client and (some_condition_for_llm_refinement):
            #     prompt = f"Question: {message.question}\nQuestion Type: {message.question_type}\nRaw Answer: {message.current_answer}\nRefined Answer:"
            #     llm_refined_response = await self._model_client.create(
            #         messages=[SystemMessage(content=self._system_message_content), UserMessage(content=prompt)]
            #     )
            #     refined_ans = llm_refined_response.content.strip() if isinstance(llm_refined_response, AssistantMessage) else str(llm_refined_response).strip()


        print(f"Output (Refined Answer): '{refined_ans[:100]}'")
        await self.publish_message(
            VQAWorkflowMessage(
                question_id=message.question_id,
                image_b64_url=message.image_b64_url, # Pass along if needed by next agent (eval might not)
                question=message.question,
                question_type=message.question_type,
                current_answer=refined_ans,
                target_answer_struct=message.target_answer_struct,
                aux_data=message.aux_data
            ),
            topic_id=TopicId(eval_format_topic_type, source=self.id)
        )

@type_subscription(topic_type=eval_format_topic_type)
class EvaluationFormattingAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__(description="Evaluates the answer and formats the output.")
        # This agent doesn't use an LLM for its core task

    @message_handler
    async def evaluate_and_format(self, message: VQAWorkflowMessage, ctx: MessageContext) -> None:
        print(f"\n----- {self.id.type} ({self.id.key}) QID: {message.question_id} -----")
        print(f"Input: Final Ans='{message.current_answer[:50]}', Q-Type='{message.question_type}'")

        eval_results_dict = {"notes": "Evaluation not performed (no target answer)."}
        if message.target_answer_struct is not None:
            # Extract the actual answer string(s) from the target_answer_struct
            # This logic mirrors what main.py and other flows do
            target_for_eval = message.target_answer_struct # Default
            if isinstance(message.target_answer_struct, list) and message.target_answer_struct:
                if isinstance(message.target_answer_struct[0], dict) and 'answer' in message.target_answer_struct[0]:
                    target_for_eval = [item['answer'] for item in message.target_answer_struct]
                elif isinstance(message.target_answer_struct[0], str):
                    target_for_eval = message.target_answer_struct # list of strings
            elif isinstance(message.target_answer_struct, str):
                 target_for_eval = [message.target_answer_struct] # single string, make it a list

            eval_results_dict, _, _ = perform_direct_accuracy_check(
                final_answer_text=message.current_answer,
                target_answers=target_for_eval, # Expects list or string
                question_type=message.question_type,
                f1_threshold_other=0.5 # Make this configurable if needed
            )
            print(f"Evaluation: Correct={eval_results_dict.get('is_correct')}, Notes='{eval_results_dict.get('notes')}'")
        else:
            print("Evaluation: Skipped (no target answer provided).")


        # Formatting the output for the presenter
        formatted_output_str = (
            f"QID: {message.question_id}\n"
            f"Question: {message.question}\n"
            f"Question Type: {message.question_type}\n"
            f"Predicted Answer: {message.current_answer}\n"
            f"Target Answer(s): {message.target_answer_struct}\n"
            f"Evaluation Correct: {eval_results_dict.get('is_correct', 'N/A')}\n"
            f"Evaluation Notes: {eval_results_dict.get('notes', 'N/A')}"
        )
        if eval_results_dict.get('vqa_score') is not None:
            formatted_output_str += f"\nVQA Score: {eval_results_dict.get('vqa_score'):.2f}"


        await self.publish_message(
            VQAWorkflowMessage( # Only need relevant fields for presenter
                question_id=message.question_id,
                question=message.question, # For context
                current_answer=formatted_output_str, # The formatted string
                evaluation_result=eval_results_dict, # Full eval dict
                aux_data=message.aux_data
            ),
            topic_id=TopicId(user_presenter_topic_type, source=self.id)
        )

@type_subscription(topic_type=user_presenter_topic_type)
class UserPresenterAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__(description="Presents the final VQA result to the user.")
        self.results_collected = [] # To store results if running multiple tasks

    @message_handler
    async def present_result(self, message: VQAWorkflowMessage, ctx: MessageContext) -> None:
        print(f"\n----- {self.id.type} ({self.id.key}) QID: {message.question_id} FINAL RESULT -----")
        if message.aux_data.get("error_source"):
            print("An error occurred earlier in the workflow:")
            print(f"  Source: {message.aux_data.get('error_source')}")
            print(f"  Error Details: {message.current_answer}") # Error message stored here
        else:
            print(message.current_answer) # This is the formatted string from EvalAgent
        print("=" * 40)
        self.results_collected.append(message) # Store the whole message for later inspection


async def main_vqa_sequential_workflow(vqa_tasks: list):
    api_key = os.getenv("OPENAI_API_KEY") # IMPORTANT
    # Fallback to a dummy client if no key, to allow structural testing
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found. Using Dummy LLM Client. LLM calls will not be real.")
        class DummyClient(ChatCompletionClient):
            async def create(self, messages, **kwargs):
                # Simulate VLM behavior
                last_message = messages[-1]
                input_text = "dummy_input"
                if isinstance(last_message.content, list): # For vision messages
                    text_parts = [item['text'] for item in last_message.content if item['type'] == 'text']
                    input_text = " ".join(text_parts) if text_parts else "image_only_prompt"
                elif isinstance(last_message.content, str):
                    input_text = last_message.content

                if "yes/no" in input_text.lower() or "is there" in input_text.lower():
                    return AssistantMessage(content="yes")
                elif "how many" in input_text.lower():
                    return AssistantMessage(content="2")
                else:
                    return AssistantMessage(content=f"Dummy VQA response to: {input_text[:30]}")
            async def close(self): pass
            @property
            def model_info(self): return {"model": "dummy-vlm"}
            @property
            def capabilities(self): return {"vision": True}
            @property
            def actual_usage(self): return None
            @property
            def total_usage(self): return None
            @property
            def remaining_tokens(self): return None
            def count_tokens(self, messages, **kwargs) -> int: return 0
        model_client = DummyClient()
    else:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4-vision-preview", # Or your preferred VLM, e.g. "gpt-4o"
            api_key=api_key,
        )

    runtime = SingleThreadedAgentRuntime()

    # Register agents
    await ImagePrepAgent.register(
        runtime, type=image_prep_topic_type, agent_id=AgentId(image_prep_topic_type, "img_prep_1"),
        factory=lambda: ImagePrepAgent()
    )
    await InitialVQAAgent.register(
        runtime, type=initial_vqa_topic_type, agent_id=AgentId(initial_vqa_topic_type, "init_vqa_1"),
        factory=lambda: InitialVQAAgent(model_client=model_client)
    )
    await AnswerRefinementAgent.register(
        runtime, type=answer_refinement_topic_type, agent_id=AgentId(answer_refinement_topic_type, "refine_1"),
        factory=lambda: AnswerRefinementAgent() # No LLM for this simple version
    )
    await EvaluationFormattingAgent.register(
        runtime, type=eval_format_topic_type, agent_id=AgentId(eval_format_topic_type, "eval_1"),
        factory=lambda: EvaluationFormattingAgent()
    )
    user_presenter_instance = UserPresenterAgent() # Keep instance to access results
    await UserPresenterAgent.register(
        runtime, type=user_presenter_topic_type, agent_id=AgentId(user_presenter_topic_type, "presenter_1"),
        factory=lambda: user_presenter_instance
    )

    runtime.start()

    # --- Create a dummy image file for testing if it doesn't exist ---
    # This part is for local testing if you don't have an image path immediately.
    # In a real scenario, image_path would come from your dataset.
    dummy_image_path = "dummy_vqa_image.png"
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 150), color = 'skyblue')
            d = ImageDraw.Draw(img)
            d.text((10,10), "Test Image\nHello VQA!", fill=(0,0,0))
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow not installed, cannot create dummy image. Please provide a valid image_path.")
            dummy_image_path = None # Cannot use dummy
    # --- End dummy image creation ---

    for task_info in vqa_tasks:
        task_image_path = task_info.get("image_path", dummy_image_path if dummy_image_path else "path/to/your/image.jpg") # Use dummy if task doesn't specify
        if not task_image_path:
            print(f"Skipping task for QID {task_info['question_id']} due to missing image path.")
            continue

        print(f"\nPublishing initial message for QID: {task_info['question_id']}...")
        await runtime.publish_message(
            VQAWorkflowMessage(
                question_id=task_info["question_id"],
                image_path=task_image_path,
                question=task_info["question"],
                target_answer_struct=task_info.get("target_answers") # VQAv2 format: list of dicts
            ),
            topic_id=TopicId(image_prep_topic_type, source="vqa_task_initiator"),
        )

    await runtime.stop_when_idle()
    await model_client.close() # Important to close the client

    print("\n--- All VQA Sequential Workflow Tasks Finished ---")
    print(f"Total results collected by presenter: {len(user_presenter_instance.results_collected)}")
    # You can now process user_presenter_instance.results_collected
    # For example, calculate overall accuracy:
    num_correct = 0
    num_evaluated = 0
    for res_msg in user_presenter_instance.results_collected:
        if res_msg.evaluation_result and res_msg.evaluation_result.get("is_correct") is not None:
            num_evaluated += 1
            if res_msg.evaluation_result.get("is_correct"):
                num_correct +=1
    if num_evaluated > 0:
        accuracy = (num_correct / num_evaluated) * 100
        print(f"Overall Accuracy on evaluated tasks: {accuracy:.2f}% ({num_correct}/{num_evaluated})")
    else:
        print("No tasks were successfully evaluated for accuracy.")


