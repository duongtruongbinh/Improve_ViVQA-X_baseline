import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
from autogen.agentchat.conversable_agent import ConversableAgent

class BaseVQAAgent(ConversableAgent, ABC):
    """Base class for VQA agents with common functionality."""
    
    def __init__(
        self,
        name: str,
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt_template_filename: Optional[str] = None,
        prompts_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            **kwargs
        )
        
        self.system_prompt_template_filename = system_prompt_template_filename
        self.logger = logging.getLogger(name)
        
        # Setup Jinja environment for prompt templates
        if prompts_dir is None:
            prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts')
            
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found at: {prompts_dir}")
            
        self.jinja_env = Environment(
            loader=FileSystemLoader(prompts_dir),
            autoescape=select_autoescape(['j2']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
    def load_prompt_template(self, template_filename: str, **kwargs) -> str:
        """Load and render a prompt template."""
        try:
            template = self.jinja_env.get_template(template_filename)
            return template.render(**kwargs)
        except Exception as e:
            error_msg = f"Error loading/rendering prompt '{template_filename}': {e}"
            self.logger.error(error_msg)
            return f"[PromptError: {error_msg}]"
            
    def prepare_system_message(self, **kwargs) -> str:
        """Prepare system message based on template and context."""
        if not self.system_prompt_template_filename:
            return "You are a VQA agent. Please answer the question about the image."
            
        return self.load_prompt_template(self.system_prompt_template_filename, **kwargs)
        
    def parse_model_output(self, content: str, question_type: str = "other") -> Tuple[str, float]:
        """Parse model output to extract answer and confidence score."""
        try:
            # First try to parse as JSON
            gen_output = json.loads(content)
            
            # Handle different response formats
            if isinstance(gen_output, dict):
                answer = str(gen_output.get("answer", "")).strip()
            elif isinstance(gen_output, (int, float)):
                answer = str(gen_output)
            else:
                answer = str(gen_output)
                
            if not answer:
                return "[EmptyAnswer]", 0.0
                
            # Validate answer format based on question type
            if question_type == "yes/no":
                if answer.lower() not in ["yes", "no"]:
                    return answer, 0.5
            elif question_type == "number":
                if not answer.isdigit():
                    return answer, 0.5
                    
            return answer, self.evaluate_confidence(answer, question_type)
            
        except json.JSONDecodeError:
            # Try to extract answer from raw text
            content = content.strip()
            if question_type == "yes/no":
                if content.lower().startswith(("yes", "no")):
                    return content.lower().split()[0], 0.8
            elif question_type == "number":
                # Try to extract first number from text
                import re
                numbers = re.findall(r'\d+', content)
                if numbers:
                    return numbers[0], 0.8
            return content, 0.5
        except Exception as e:
            return f"[ParseError: {str(e)}]", 0.0
            
    def evaluate_confidence(self, answer: str, question_type: str) -> float:
        """Evaluate confidence score based on answer and question type."""
        if question_type == "yes/no":
            return 1.0 if answer.lower() in ["yes", "no"] else 0.5
        elif question_type == "number":
            return 1.0 if answer.isdigit() else 0.5
        else:
            # Use length and structure as confidence indicators
            return min(1.0, len(answer.split()) / 10)
            
    @abstractmethod
    async def handle_task(self, message: Any, ctx: Any) -> Any:
        """Handle incoming task - must be implemented by subclasses."""
        pass 