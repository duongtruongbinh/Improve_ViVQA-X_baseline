# agents.py
import sys
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from autogen.agentchat.conversable_agent import ConversableAgent
except ImportError:
    print("ERROR: Could not import ConversableAgent from autogen.agentchat.conversable_agent.")
    print("Ensure your AutoGen version (expected 0.5.6 or compatible) has this class or adjust the import path.")
    raise

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_SCRIPT_DIR, '..', 'prompts')

if not os.path.isdir(PROMPTS_DIR):
    print(f"ERROR: Prompts directory not found at {PROMPTS_DIR}.")
    sys.exit(1)

try:
    jinja_env = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        autoescape=select_autoescape(['j2']), 
        trim_blocks=True,
        lstrip_blocks=True
    )
except Exception as e:
    print(f"ERROR: Failed to initialize Jinja2 environment: {e}")
    sys.exit(1)

def load_static_prompt(template_name: str) -> str:
    try:
        template = jinja_env.get_template(template_name)
        return template.render()
    except Exception as e:
        print(f"ERROR: Failed to load/render static template '{template_name}': {e}")
        raise

def render_dynamic_prompt(template_name: str, **kwargs) -> str:
    try:
        template = jinja_env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load/render dynamic template '{template_name}' with {kwargs}: {e}")
        raise

try:
    from vllm_clients import llm_config_vlm, llm_config_llm 
except ImportError as e:
    print(f"ERROR: Failed to import llm_config_vlm or llm_config_llm from vllm_clients.py: {e}.")
    print("Ensure vllm_clients.py provides these llm_config dictionaries.")
    sys.exit(1)


class VQAOrchestratorAgent(ConversableAgent):
    def __init__(self, 
                 name: str = "VQA_Orchestrator", 
                 description: str = "Orchestrates the VQA agent workflow.",
                 llm_config=None, # This will be overridden by the instance creation below
                 human_input_mode="NEVER",
                 code_execution_config=False, 
                 **kwargs):
        super().__init__(
            name=name, 
            description=description, 
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            **kwargs
            )

try:
    initial_vlm_agent = ConversableAgent(
        name="Initial_VLM_Agent",
        llm_config=llm_config_vlm, 
        system_message=load_static_prompt('vlm/initial_default.j2'),
        description="Uses VLM for the first VQA attempt.",
        human_input_mode="NEVER",
        code_execution_config=False 
    )

    failure_analysis_agent = ConversableAgent(
        name="Failure_Analysis_Agent",
        llm_config=llm_config_llm,
        system_message=load_static_prompt('agents/failure_analysis_system.j2'),
        description="Uses LLM to analyze VLM failures and suggest reattempt strategy.",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    object_attribute_agent = ConversableAgent(
        name="Object_Attribute_Agent",
        llm_config=llm_config_vlm, 
        system_message=load_static_prompt('agents/object_attribute_system_no_tools.j2'),
        description="Uses VLM to describe textually specified items in the image for reattempts.",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    reattempt_vlm_agent = ConversableAgent(
        name="Reattempt_VLM_Agent",
        llm_config=llm_config_vlm,
        system_message=load_static_prompt('vlm/reattempt_default_no_tools.j2'),
        description="Uses VLM to reattempt VQA with additional textual context.",
        human_input_mode="NEVER",
        code_execution_config=False
    )

except Exception as e_agent_init:
    print(f"ERROR: Failed to initialize one or more ConversableAgents: {e_agent_init}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Initialize VQAOrchestratorAgent with llm_config_llm.
    # This ensures it has an internal client setup, even if it mainly delegates.
    # llm_config_llm from vllm_clients.py should have cache_seed: None.
    vqa_orchestrator = VQAOrchestratorAgent(
        llm_config=llm_config_llm 
    )
except Exception as e_orch_init:
    print(f"ERROR: Failed to initialize VQAOrchestratorAgent: {e_orch_init}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("AutoGen agents initialized successfully.")
