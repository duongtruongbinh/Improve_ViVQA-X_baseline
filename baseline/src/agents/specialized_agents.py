# specialized_agents.py
# ------------ 0. Imports -----------------------------------------------------------
import sys, os
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ------------ 1. Jinja & path setup -------------------------------------------------
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_SCRIPT_DIR, '..', 'prompts')

if not (os.path.isdir(PROMPTS_DIR) or os.path.isfile(PROMPTS_DIR + ".py")):
    print(f"ERROR: Prompts directory not found at {PROMPTS_DIR}.")
    sys.exit(1)

jinja_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(['j2']),
    trim_blocks=True,
    lstrip_blocks=True
)

def load_static_prompt(template_name: str) -> str:
    """Render template with no variables."""
    template = jinja_env.get_template(template_name)
    return template.render()

def render_dynamic_prompt(template_name: str, **kwargs) -> str:
    """Render template with variables."""
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)

# ------------ 2. Import AutoGen -----------------------------------------------------
try:
    from autogen.agentchat.conversable_agent import ConversableAgent
except ImportError:
    print("ERROR: Could not import ConversableAgent from autogen.agentchat.conversable_agent.")
    raise

# ------------ 3. LLM configs --------------------------------------------------------
try:
    from ..vllm_clients import llm_config_vlm, llm_config_llm
except ImportError as e:
    print(f"ERROR: Failed to import llm_config_* from vllm_clients.py: {e}")
    sys.exit(1)

# ------------ 4. Helper: create strict-grader agent ---------------------------------
def create_grader_agent(grader_id: int,
                        question: str,
                        target_answer: str,
                        model_answer: str):

    sys_prompt = render_dynamic_prompt(
        'grading/grading_system.j2',
        grader_id=grader_id,
        question=question,
        target_answer=target_answer,
        model_answer=model_answer
    )
    return ConversableAgent(
        name=f"Grader_{grader_id}",
        llm_config=llm_config_llm,
        system_message=sys_prompt,
        human_input_mode="NEVER",
        code_execution_config=False
    )

# ------------ 5. Orchestrator agent class ------------------------------------------
class VQAOrchestratorAgent(ConversableAgent):
    def __init__(self,
                 name="VQA_Orchestrator",
                 description="Orchestrates the VQA agent workflow.",
                 llm_config=None,
                 human_input_mode="NEVER",
                 code_execution_config=False,
                 **kwargs):
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            **kwargs)

# ------------ 6. Instantiate core agents -------------------------------------------
try:
    initial_vlm_agent = ConversableAgent(
        name="Initial_VLM_Agent",
        llm_config=llm_config_vlm,
        system_message=load_static_prompt('specialized/initial_vqa.j2'),
        description="Uses VLM for the first VQA attempt.",
        human_input_mode="NEVER"
    )

    failure_analysis_agent = ConversableAgent(
        name="Failure_Analysis_Agent",
        llm_config=llm_config_llm,
        system_message=load_static_prompt('specialized/failure_analysis.j2'),
        description="Analyzes VLM failures and suggests reattempt strategy.",
        human_input_mode="NEVER"
    )

    object_attribute_agent = ConversableAgent(
        name="Object_Attribute_Agent",
        llm_config=llm_config_vlm,
        system_message=load_static_prompt('specialized/object_attribute.j2'),
        description="Describes specified objects/attributes in the image.",
        human_input_mode="NEVER"
    )

    reattempt_vlm_agent = ConversableAgent(
        name="Reattempt_VLM_Agent",
        llm_config=llm_config_vlm,
        system_message=load_static_prompt('specialized/reattempt.j2'),
        description="VLM reattempts VQA with extra context.",
        human_input_mode="NEVER"
    )

except Exception as e_init:
    print("ERROR: Failed to initialize core ConversableAgents:", e_init)
    sys.exit(1)

# ------------ 7. Orchestrator instance ---------------------------------------------
try:
    vqa_orchestrator = VQAOrchestratorAgent(
        system_message=load_static_prompt('specialized/orchestrator.j2'),
        llm_config=llm_config_llm
    )
except Exception as e_orch:
    print("ERROR: Failed to initialize VQAOrchestratorAgent:", e_orch)
    sys.exit(1)

print("AutoGen agents initialized successfully.")