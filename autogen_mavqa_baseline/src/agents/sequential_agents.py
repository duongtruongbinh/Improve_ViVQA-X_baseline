# sequential_agents.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional
import asyncio # For asyncio.Event
from pydantic import BaseModel, Field
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

# Define Topic Types (can also be in sequential_flow.py, but here for agent proximity)
IMAGE_CONTEXTUALIZER_TOPIC = "VQAImageContextualizerAgentTopic"
QUESTION_ANSWERER_TOPIC = "VQAQuestionAnswererAgentTopic"
ANSWER_FORMATTER_TOPIC = "VQAAnswerFormatterAgentTopic"
RESULT_COLLECTOR_TOPIC = "VQAResultCollectorAgentTopic"


@dataclass
class VQATaskRelayMessage:
    question_id: str
    image_url: str
    question: str
    target_answers: Any
    question_type: str
    current_step_output: str = Field(default="")
    error: str = "" 
    
    # For internal flow control if needed, not directly part of agent-to-agent message
    # completion_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    class Config:
        # Allow arbitrary types if needed (e.g., for asyncio.Event)
        arbitrary_types_allowed = True

@type_subscription(topic_type=IMAGE_CONTEXTUALIZER_TOPIC)
class VQAImageContextualizerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, agent_name: str = "VQAImageContextualizer") -> None:
        super().__init__(description="Analyzes image in context of a question to provide visual details.")
        self._model_client = model_client
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")
        self._system_message = SystemMessage(
            content=(
                "You are a specialist Visual Context Extractor. Your task is to carefully examine an image and a given question, then extract and describe ONLY the specific visual elements, attributes, and spatial relationships in the image that are DIRECTLY relevant and essential for answering the question. "
                "Focus on objective, observable details. Do NOT attempt to answer the question itself. Do NOT infer information beyond what is visually present. "
                "Your output should be a factual, textual description of these key visual details. Avoid general descriptions of the entire image unless all parts are relevant. "
                "For example, if the question is 'What color is the car?', describe the car's color and any visual context that helps identify it. Do not describe other objects unless they are crucial for context related to the car and its color."
            )
        )
    #Cập nhật logger với agent_id để phân biệt các instance của agent.
    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key if self.id else 'unbound'}")

    @message_handler
    async def handle_task(self, message: VQATaskRelayMessage, ctx: MessageContext) -> None:
        self.logger.info(f"QID {message.question_id}: Received task. Contextualizing image.")

        user_prompt = f"Focus on details relevant to this question: {message.question}"
        
        messages_to_send = [
            self._system_message,
            UserMessage(content=user_prompt, source="user_contextualizer_input")
        ]

        response_content = f"[VLMError_Contextualizer_{self.id.key if self.id else 'unbound'}]"
        try:
            # Check if the client supports image_path_for_create parameter
            if hasattr(self._model_client, '_supports_image_path') or 'VLLM' in str(type(self._model_client)):
                model_result = await self._model_client.create(
                    messages=messages_to_send,
                    cancellation_token=ctx.cancellation_token,
                    image_path_for_create=message.image_url
                )
            else:
                # For standard OpenAI clients, just use text prompt
                model_result = await self._model_client.create(
                    messages=messages_to_send,
                    cancellation_token=ctx.cancellation_token
                )
            if isinstance(model_result.content, str) and model_result.content.strip():
                response_content = model_result.content.strip()
            else:
                response_content = "[EmptyContextualizerOutput]"
                self.logger.warning(f"QID {message.question_id}: Contextualizer VLM returned empty content.")
        except Exception as e:
            self.logger.error(f"QID {message.question_id}: Error during Contextualizer VLM call: {e}", exc_info=True)
            message.error = f"ContextualizerError: {str(e)}"

        message.current_step_output = response_content
        self.logger.info(f"QID {message.question_id}: Image contextualized. Output: '{response_content[:100]}...'")
        await self.publish_message(message, topic_id=TopicId(QUESTION_ANSWERER_TOPIC, source=self.id.key))

@type_subscription(topic_type=QUESTION_ANSWERER_TOPIC)
class VQAQuestionAnswererAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, agent_name: str = "VQAQuestionAnswerer") -> None:
        super().__init__(description="Answers a VQA question based on provided visual context.")
        self._model_client = model_client
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")
        self._system_message = SystemMessage(
            content=(
                "You are a precise Question Answering Specialist. Your task is to answer the 'Original Question' with high accuracy. "
                "You MUST base your answer STRICTLY and SOLELY on the information contained within the 'Original Question' itself and the 'Provided Visual Analysis'. "
                "Do NOT use any external knowledge or make assumptions beyond what is explicitly stated in these inputs. "
                "Do NOT refer to 'the visual analysis' or 'the provided description' in your answer. Simply provide the direct answer to the question. "
                "If the provided visual analysis is insufficient to answer the question, state that the information is not available in the visual analysis."
            )
        )

    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key if self.id else 'unbound'}")

    @message_handler
    async def handle_task(self, message: VQATaskRelayMessage, ctx: MessageContext) -> None:
        self.logger.info(f"QID {message.question_id}: Received contextualized data. Answering question.")

        if message.error: # Propagate error from previous step
            self.logger.warning(f"QID {message.question_id}: Skipping QA due to previous error: {message.error}")
            await self.publish_message(message, topic_id=TopicId(ANSWER_FORMATTER_TOPIC, source=self.id.key))
            return

        prompt = (
            f"Original Question: {message.question}\n\n"
            f"Visual Analysis of the Image: {message.current_step_output}\n\n"
            "Based on the above, provide a direct answer to the Original Question:"
        )
        messages_to_send = [
            self._system_message,
            UserMessage(content=prompt, source="user_qa_input")
        ]

        response_content = f"[LLMError_QA_{self.id.key if self.id else 'unbound'}]"
        try:
            model_result = await self._model_client.create(
                messages=messages_to_send,
                cancellation_token=ctx.cancellation_token
            )
            if isinstance(model_result.content, str) and model_result.content.strip():
                response_content = model_result.content.strip()
            else:
                response_content = "[EmptyQAOutput]"
                self.logger.warning(f"QID {message.question_id}: QA LLM returned empty content.")
        except Exception as e:
            self.logger.error(f"QID {message.question_id}: Error during QA LLM call: {e}", exc_info=True)
            message.error = message.error + f"; QAError: {str(e)}" if message.error else f"QAError: {str(e)}"

        message.current_step_output = response_content
        self.logger.info(f"QID {message.question_id}: Question answered. Output: '{response_content[:100]}...'")
        await self.publish_message(message, topic_id=TopicId(ANSWER_FORMATTER_TOPIC, source=self.id.key))


@type_subscription(topic_type=ANSWER_FORMATTER_TOPIC)
class VQAAnswerFormatterAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, agent_name: str = "VQAAnswerFormatter") -> None:
        super().__init__(description="Formats a VQA answer to be concise and direct.")
        self._model_client = model_client
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")
        self._system_message = SystemMessage(
            content=(
                "You are an expert Answer Refinement and Formatting Specialist. Your task is to take an 'Original Question' and a 'Candidate Answer' and refine the candidate answer. "
                "The refined answer MUST be: \n"
                "1. Directly responsive to the 'Original Question'.\n"
                "2. As concise as possible while retaining the core meaning of the 'Candidate Answer'.\n"
                "3. Well-formatted according to the question type. \n"
                "4. DO NOT introduce new information or alter the factual content of the 'Candidate Answer'. Your role is to simplify and format, not to re-answer or correct.\n\n"
                "Formatting Examples:\n"
                "- If Original Question is 'Is there a cat in the image?', and Candidate Answer is 'Yes, a black cat is visible.', a good refined answer is 'Yes'.\n"
                "- If Original Question is 'How many dogs are present?', and Candidate Answer is 'I can see three dogs playing.', a good refined answer is '3'.\n"
                "- If Original Question is 'What is the color of the umbrella?', and Candidate Answer is 'The umbrella appears to be a shade of bright red.', a good refined answer is 'Red' or 'Bright red'.\n"
                "- If Original Question is 'What is the man doing?', and Candidate Answer is 'The man in the picture is riding a bicycle down the street.', a good refined answer is 'Riding a bicycle' or 'Riding a bicycle down the street'. Choose the most direct and informative short phrase.\n\n"
                "Output ONLY the final refined answer. No preamble, no explanation, just the answer itself."
            )
        )
    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key if self.id else 'unbound'}")

    @message_handler
    async def handle_task(self, message: VQATaskRelayMessage, ctx: MessageContext) -> None:
        self.logger.info(f"QID {message.question_id}: Received answer. Formatting.")

        if message.error: # Propagate error
            self.logger.warning(f"QID {message.question_id}: Skipping formatting due to previous error: {message.error}")
            await self.publish_message(message, topic_id=TopicId(RESULT_COLLECTOR_TOPIC, source=self.id.key))
            return

        prompt = (
            f"Original Question: {message.question}\n\n"
            f"Candidate Answer: {message.current_step_output}\n\n"
            "Refine the candidate answer according to your instructions:"
        )
        messages_to_send = [
            self._system_message,
            UserMessage(content=prompt, source="user_formatter_input")
        ]

        response_content = f"[LLMError_Formatter_{self.id.key if self.id else 'unbound'}]"
        try:
            model_result = await self._model_client.create(
                messages=messages_to_send,
                cancellation_token=ctx.cancellation_token
            )
            if isinstance(model_result.content, str) and model_result.content.strip():
                response_content = model_result.content.strip()
            else:
                response_content = "[EmptyFormatterOutput]"
                self.logger.warning(f"QID {message.question_id}: Formatter LLM returned empty content.")

        except Exception as e:
            self.logger.error(f"QID {message.question_id}: Error during Formatter LLM call: {e}", exc_info=True)
            message.error = message.error + f"; FormatterError: {str(e)}" if message.error else f"FormatterError: {str(e)}"


        message.current_step_output = response_content # This is the final answer
        self.logger.info(f"QID {message.question_id}: Answer formatted. Output: '{response_content[:100]}...'")
        await self.publish_message(message, topic_id=TopicId(RESULT_COLLECTOR_TOPIC, source=self.id.key))


@type_subscription(topic_type=RESULT_COLLECTOR_TOPIC)
class VQAResultCollectorAgent(RoutedAgent):
    def __init__(
        self,
        agent_name: str = "VQAResultCollector",
        completion_events: Dict[str, asyncio.Event] = None,
        final_results_store: Dict[str, Dict] = None
    ) -> None:
        super().__init__(description="Collects the final VQA result and signals completion.")
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{self.agent_name}")
        self._completion_events = completion_events if completion_events is not None else {}
        self._final_results_store = final_results_store if final_results_store is not None else {}

    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key if self.id else 'unbound'}")

    @message_handler
    async def handle_final_result(self, message: VQATaskRelayMessage, ctx: MessageContext) -> None:
        qid = message.question_id
        self.logger.info(f"QID {qid}: Received final processed message.")

        # Construct the final result structure expected by main.py
        # The `current_step_output` at this stage is the final_answer
        result_for_store = {
            "question_id": qid,
            "image_path_internal": message.image_url, # This is the processed URL
            "question": message.question,
            "target_answers": message.target_answers,
            "question_type_internal": message.question_type,
            "final_answer": message.current_step_output if not message.error else f"[ErrorInPipeline: {message.error}]",
            "intermediate_outputs": { # Optional: if you want to log steps
                "contextualizer_output_if_no_error": message.current_step_output if qid in str(message.error or "") and "FormatterError" in str(message.error or "") else "N/A", # Example logic
            },
            "error": message.error,
            "processing_time_seconds": 0.0, # Will be calculated in the flow
            "direct_accuracy_check": {}, # Will be calculated in the flow
            "flow_type_used": "sequential"
        }
        self._final_results_store[qid] = result_for_store

        if qid in self._completion_events:
            self._completion_events[qid].set()
            self.logger.info(f"QID {qid}: Completion event set.")
        else:
            self.logger.warning(f"QID {qid}: No completion event found to set.")