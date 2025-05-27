# reflection_agents.py
import os
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from autogen.agentchat.conversable_agent import ConversableAgent

CURRENT_SCRIPT_DIR_RF = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR_RF = os.path.join(CURRENT_SCRIPT_DIR_RF, '..', 'prompts')

if not os.path.isdir(PROMPTS_DIR_RF):
    raise FileNotFoundError(f"Prompts directory not found at: {PROMPTS_DIR_RF}. "
                            f"This script expects it at 'project_root/src/prompts/'.")

jinja_env_rf = Environment(
    loader=FileSystemLoader(PROMPTS_DIR_RF),
    autoescape=select_autoescape(['j2']),
    trim_blocks=True,
    lstrip_blocks=True
)

def load_reflection_prompt(template_filename: str, **kwargs) -> str:
    try:
        template = jinja_env_rf.get_template(template_filename)
        return template.render(**kwargs)
    except Exception as e:
        print(f"ERROR loading/rendering reflection prompt '{template_filename}': {e}")
        return f"Error loading prompt {template_filename}. Please check template."

class VQAGeneratorAgent(ConversableAgent):
    def __init__(self, 
                 name: str = "VQAGenerator", 
                 llm_config: dict = None,
                 system_prompt_template_filename: str = "reflection/vqa_generator_system.j2",
                 **kwargs):
        
        self.system_prompt_template_filename = system_prompt_template_filename
        
        super().__init__(
            name=name,
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            **kwargs,
        )

    def prepare_system_message(self, question_text: str, question_type: str = "other", critique_text: str = None):
        # Select template based on question type
        if question_type == "yes/no":
            template = "reflection/vqa_generator_yesno.j2"
        elif question_type == "number":
            template = "reflection/vqa_generator_number.j2"
        else:
            template = "reflection/vqa_generator_other.j2"
            
        critique_section_content = ""
        if critique_text:
            critique_section_content = critique_text
            
        rendered_prompt = load_reflection_prompt(
            template,
            question=question_text,
            critique_section=critique_section_content
        )
        self.update_system_message(rendered_prompt)

    def parse_model_output(self, content: str, question_type: str) -> tuple[str, float]:
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
                return "[VQAOutputEmptyAnswerInJSON]", 0.0
                
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
        if question_type == "yes/no":
            return 1.0 if answer.lower() in ["yes", "no"] else 0.5
        elif question_type == "number":
            return 1.0 if answer.isdigit() else 0.5
        else:
            # Use length and structure as confidence indicators
            return min(1.0, len(answer.split()) / 10)

class VQAReflectorAgent(ConversableAgent):
    def __init__(self, 
                 name: str = "VQAReflector", 
                 llm_config: dict = None,
                 system_prompt_template_filename: str = "reflection/vqa_reflector_system.j2",
                 **kwargs):
        
        initial_system_message = load_reflection_prompt(system_prompt_template_filename)
        
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=initial_system_message,
            human_input_mode="NEVER",
            code_execution_config=False,
            **kwargs,
        )

    def prepare_dynamic_system_message(self, 
                                     question_for_reflector: str, 
                                     generated_answer_for_reflector: str, 
                                     generated_explanation_for_reflector: str, 
                                     question_type: str = "other",
                                     template_filename: str ="reflection/vqa_reflector_system.j2"):
        rendered_prompt = load_reflection_prompt(
            template_filename,
            question_for_reflector=question_for_reflector,
            generated_answer_for_reflector=generated_answer_for_reflector,
            generated_explanation_for_reflector=generated_explanation_for_reflector,
            question_type=question_type
        )
        self.update_system_message(rendered_prompt)

    async def evaluate(self, question: str, answer: str, question_type: str) -> str:
        self.prepare_dynamic_system_message(
            question_for_reflector=question,
            generated_answer_for_reflector=answer,
            generated_explanation_for_reflector="",
            question_type=question_type
        )
        
        chat_res = await self.a_generate_reply(
            messages=[{"role": "user", "content": "Evaluate the answer."}],
            sender=None
        )
        
        return chat_res.get("content", "[EVALUATION_FAILED]")