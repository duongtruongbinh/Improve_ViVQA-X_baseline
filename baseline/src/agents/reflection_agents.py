# reflection_agents.py
import os
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

    def prepare_system_message(self, question_text: str, critique_text: str = None):
        critique_section_content = ""
        if critique_text:
            critique_section_content = critique_text
        rendered_prompt = load_reflection_prompt(
            self.system_prompt_template_filename,
            question=question_text,
            critique_section=critique_section_content
        )
        self.update_system_message(rendered_prompt)

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
                                     template_filename: str ="reflection/vqa_reflector_system.j2"):
        rendered_prompt = load_reflection_prompt(
            template_filename,
            question_for_reflector=question_for_reflector,
            generated_answer_for_reflector=generated_answer_for_reflector,
            generated_explanation_for_reflector=generated_explanation_for_reflector
        )
        self.update_system_message(rendered_prompt)