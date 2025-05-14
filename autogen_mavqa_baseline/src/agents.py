import sys
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

# --- Constants & Environment Setup ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_SCRIPT_DIR, '..', 'prompts')

if not os.path.isdir(PROMPTS_DIR):
    print(f"ERROR: Prompts directory not found at {PROMPTS_DIR}.")
    sys.exit(1)

try:
    jinja_env = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        autoescape=select_autoescape(['j2']), # Autoescape for .j2 files
        trim_blocks=True,
        lstrip_blocks=True
    )
except Exception as e:
    print(f"ERROR: Failed to initialize Jinja2 environment: {e}")
    sys.exit(1)

# --- Prompt Utilities ---
def load_static_prompt(template_name: str) -> str:
    """Loads and renders a Jinja2 template without dynamic variables."""
    try:
        template = jinja_env.get_template(template_name)
        return template.render()
    except Exception as e:
        print(f"ERROR: Failed to load/render static template '{template_name}': {e}")
        raise

def render_dynamic_prompt(template_name: str, **kwargs) -> str:
    """Loads and renders a Jinja2 template with dynamic variables."""
    try:
        template = jinja_env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load/render dynamic template '{template_name}' with {kwargs}: {e}")
        raise

# --- Core Imports ---
try:
    from vllm_clients import vlm_client_vllm, llm_client_vllm
    from config_loader import app_config
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}. Check paths and dependencies.")
    sys.exit(1)
except NameError as e_name:
    print(f"ERROR: A required object (likely app_config) was not loaded: {e_name}. Check config_loader.py.")
    sys.exit(1)

# --- Agent Definitions ---
class VQAOrchestratorAgent(UserProxyAgent):
    def __init__(self, name: str = "VQA_Orchestrator", description: str = "Orchestrates the VQA agent workflow.", **kwargs):
        super().__init__(name=name, description=description)
        # Note: human_input_mode, code_execution_config etc. removed if not standard UserProxyAgent args

# --- Client Sanity Check ---
if vlm_client_vllm is None or llm_client_vllm is None:
    print("CRITICAL ERROR: One or both VLLM clients failed to initialize. Check vllm_clients.py and config.")
    sys.exit(1)

# --- Agent Instantiation ---
try:
    initial_vlm_agent = AssistantAgent(
        name="Initial_VLM_Agent",
        model_client=vlm_client_vllm,
        system_message=load_static_prompt('vlm/initial_default.j2'),
        description="Uses VLM for the first VQA attempt."
    )

    failure_analysis_agent = AssistantAgent(
        name="Failure_Analysis_Agent",
        model_client=llm_client_vllm,
        system_message=load_static_prompt('agents/failure_analysis_system.j2'),
        description="Uses LLM to analyze VLM failures and suggest reattempt strategy."
    )

    object_attribute_agent = AssistantAgent(
        name="Object_Attribute_Agent",
        model_client=vlm_client_vllm,
        system_message=load_static_prompt('agents/object_attribute_system_no_tools.j2'),
        description="Uses VLM to describe textually specified items in the image for reattempts."
    )

    reattempt_vlm_agent = AssistantAgent(
        name="Reattempt_VLM_Agent",
        model_client=vlm_client_vllm,
        system_message=load_static_prompt('vlm/reattempt_default_no_tools.j2'),
        description="Uses VLM to reattempt VQA with additional textual context."
    )

except Exception as e_agent_init:
    print(f"ERROR: Failed to initialize one or more AssistantAgents: {e_agent_init}")
    sys.exit(1)

try:
    vqa_orchestrator = VQAOrchestratorAgent()
except Exception as e_orch_init:
    print(f"ERROR: Failed to initialize VQAOrchestratorAgent: {e_orch_init}")
    sys.exit(1)

print("AutoGen agents initialized successfully.")