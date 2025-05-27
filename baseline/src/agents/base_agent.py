from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
from autogen_core import RoutedAgent, MessageContext
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, AssistantMessage

class BaseVQAAgent(RoutedAgent, ABC):
    """Base class for VQA agents with common functionality."""
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        agent_name: str = "BaseVQAAgent",
        system_prompt: Optional[str] = None
    ):
        super().__init__(description=f"A {agent_name} that answers questions based on images.")
        self._model_client = model_client
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"{self.agent_name}")
        
    async def bind_id_and_runtime(self, agent_id: Any, runtime: Any) -> None:
        """Bind agent ID and runtime, updating logger name."""
        await super().bind_id_and_runtime(agent_id, runtime)
        self.logger = logging.getLogger(f"{self.agent_name}.{self.id.key}")
        
    def prepare_system_message(self, question_text: str, question_type: str, critique_text: Optional[str] = None) -> str:
        """Prepare system message based on question type and optional critique."""
        if self.system_prompt:
            return self.system_prompt
            
        base_prompt = (
            "You are a Visual Question Answering (VQA) expert. "
            "Your task is to answer questions about images accurately and concisely. "
            "Question type: {question_type}\n"
            "Question: {question_text}"
        )
        
        if critique_text:
            base_prompt += f"\nAdditional context: {critique_text}"
            
        return base_prompt.format(
            question_type=question_type,
            question_text=question_text
        )
        
    async def prepare_and_call_vlm(
        self,
        question_text: str,
        image_url: str,
        system_prompt_parts: Optional[List[str]] = None
    ) -> AssistantMessage:
        """Prepare messages and call VLM client."""
        messages = []
        
        # Add system message if provided
        if system_prompt_parts and any(sp.strip() for sp in system_prompt_parts):
            messages.append(SystemMessage(content="\n".join(system_prompt_parts)))
            
        # Prepare user message with image and question
        user_content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": question_text}
        ]
        messages.append(UserMessage(content=user_content))
        
        # Call VLM client
        return await self._model_client.create(messages=messages)
        
    def parse_model_output(self, raw_output: str, question_type: str) -> tuple[str, float]:
        """Parse model output to extract answer and confidence score."""
        answer = None
        confidence = 0.0
        
        # Basic parsing logic - can be overridden by subclasses
        raw_output = raw_output.strip()
        if raw_output.lower().startswith("answer:"):
            answer = raw_output.split(":", 1)[-1].strip()
        elif "\n" in raw_output:
            potential_answer = raw_output.split("\n")[0].strip()
            if potential_answer.lower().startswith("answer:"):
                answer = potential_answer.split(":", 1)[-1].strip()
            else:
                answer = potential_answer
        else:
            answer = raw_output
            
        # Default confidence score - can be overridden
        confidence = 1.0 if answer else 0.0
        
        return answer or "[NoAnswer]", confidence
        
    @abstractmethod
    async def handle_task(self, message: Any, ctx: MessageContext) -> Any:
        """Handle incoming task - must be implemented by subclasses."""
        pass 