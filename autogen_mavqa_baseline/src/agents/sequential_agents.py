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



class VQATaskRelayMessage(BaseModel):
    question_id: str
    image_url: str
    question: str
    target_answers: Any
    question_type: str
    current_step_output: str = Field(default="")
    error: Optional[str] = Field(default=None)
    
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
                "You are a visual analysis assistant. Given an image and a question, "
                "describe the relevant visual elements and context in the image that would help answer the question. "
                "Focus on details pertinent to the question. Do not answer the question directly. "
                "Provide a detailed textual description of what you see that is relevant."
            )
        )

    async def bind_id_and_runtime(self, agent_id: AgentId, runtime) -> None:
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key if self.id else 'unbound'}")

    @message_handler
    async def handle_task(self, message: VQATaskRelayMessage, ctx: MessageContext) -> None:
        self.logger.info(f"QID {message.question_id}: Received task. Contextualizing image.")

        # Create a simple text prompt - the image will be handled via image_path_for_create
        user_prompt = f"Focus on details relevant to this question: {message.question}"
        
        messages_to_send = [
            self._system_message,
            UserMessage(content=user_prompt, source="user_contextualizer_input")
        ]

        response_content = f"[VLMError_Contextualizer_{self.id.key if self.id else 'unbound'}]"
        try:
            # Pass the image URL as a separate parameter, similar to the mixture agents
            model_result = await self._model_client.create(
                messages=messages_to_send,
                cancellation_token=ctx.cancellation_token,
                image_path_for_create=message.image_url
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
                "You are a Question Answering assistant. You will be given an original question and a detailed "
                "visual analysis/description of an image relevant to that question. Your task is to "
                "answer the original question accurately and concisely using ONLY the information from the "
                "original question and the provided visual analysis. "
                "Do not refer to 'the visual analysis' in your answer. Just provide the answer."
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
                "You are an Answer Refinement specialist. You will be given an original question and a candidate answer. "
                "Your task is to refine the candidate answer to be very concise, directly responsive to the question, "
                "and well-formatted. "
                "For example, if the question is 'Is there a cat?', a good refined answer is 'Yes' or 'No'. "
                "If it's 'How many cats?', a good refined answer is just the number (e.g., '3'). "
                "If it's 'What color is the cat?', a good refined answer is just the color (e.g., 'Black')."
                "Output ONLY the final refined answer, with no preamble like 'The refined answer is:'."
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