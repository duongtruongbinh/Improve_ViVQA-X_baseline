# debate_agents.py
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from autogen.agentchat.conversable_agent import ConversableAgent

CURRENT_SCRIPT_DIR_DA = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR_DA = os.path.join(CURRENT_SCRIPT_DIR_DA, '..', 'prompts')

jinja_env_da = None
if os.path.isdir(PROMPTS_DIR_DA):
    jinja_env_da = Environment(
        loader=FileSystemLoader(PROMPTS_DIR_DA),
        autoescape=select_autoescape(['j2']),
        trim_blocks=True,
        lstrip_blocks=True
    )
else:
    print(f"Warning: Prompts directory not found at: {PROMPTS_DIR_DA}. Jinja environment not initialized.")


def load_debate_prompt(template_filename: str, **kwargs) -> str:
    if jinja_env_da is None:
        print(f"ERROR: Jinja environment not initialized. Cannot load '{template_filename}'.")
        return f"Error loading prompt '{template_filename}': Jinja environment not available."
    try:
        template = jinja_env_da.get_template(template_filename)
        return template.render(**kwargs)
    except Exception as e:
        print(f"ERROR loading/rendering debate prompt '{template_filename}': {e}")
        return f"Error loading prompt '{template_filename}'. Details: {str(e)}"

class VQADebateSolverAgent(ConversableAgent):
    def __init__(self,
                 name: str,
                 llm_config: dict | bool,
                 system_prompt_template_filename="debate/vqa_debate_solver_system.j2",
                 **kwargs):
        self.system_prompt_template_filename = system_prompt_template_filename
        super().__init__(
            name=name,
            llm_config=llm_config,
            human_input_mode="NEVER", 
            code_execution_config=False,
            **kwargs,
        )

    def render_system_message(self, question: str, current_round: int, max_rounds: int, neighbor_responses: list = None) -> str:
        if not self.system_prompt_template_filename:
            return "You are a VQA solver in a debate. Please answer the question."

        rendered_prompt = load_debate_prompt(
            self.system_prompt_template_filename,
            question=question,
            current_round=current_round,
            max_rounds=max_rounds,
            neighbor_responses=neighbor_responses if neighbor_responses else []
        )
        return rendered_prompt

class VQADebateAggregatorAgent(ConversableAgent):
    def __init__(self,
                 name="VQADebateAggregator",
                 llm_config: dict | bool = False,
                 system_prompt_template_filename="debate/vqa_debate_aggregator_system.j2",
                 **kwargs):
        
        static_system_message = load_debate_prompt(system_prompt_template_filename)
        super().__init__(
            name=name,
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            system_message=static_system_message,
            **kwargs,
        )