# agents.py

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

# Import clients, prompts, and config
try:
    from vllm_clients import vlm_client_vllm, llm_client_vllm
    from prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_DEFAULT, # Default, will be updated based on dataset
        FAILURE_ANALYSIS_SYSTEM_PROMPT,
        OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS # Default, will be updated based on dataset
    )
    from config_loader import app_config
except ImportError as e:
    print(f"Error importing modules in agents.py: {e}. Ensure all files are present and accessible.")
    raise ImportError(f"Could not import required modules in agents.py: {e}") from e

# --- UserProxyAgent (Orchestrator) ---
# Manages the VQA workflow. Configured not to execute external tools directly.
class VQAOrchestratorAgent(UserProxyAgent):
    def __init__(self, name: str = "VQA_Orchestrator", **kwargs):
        super().__init__(
            name=name,
            human_input_mode="NEVER", # Fully automated
            code_execution_config=False, # No external code execution
            default_auto_reply="Orchestration process continuing or completed.", # Generic reply
            # Define termination conditions if needed, e.g., based on specific final message content
            is_termination_msg=lambda msg: isinstance(msg, dict) and (
                 "[TASK_COMPLETE]" in msg.get("content", "").upper() # Example custom signal
             ),
            **kwargs
        )
        self.app_config = app_config # Store config reference if orchestrator logic needs it

# --- AssistantAgent Instances (using vLLM clients) ---
if vlm_client_vllm is None or llm_client_vllm is None:
    raise RuntimeError("VLLM clients (vlm_client_vllm or llm_client_vllm) are not initialized. "
                       "Cannot create agents. Check vllm_clients.py and vLLM server connection.")

initial_vlm_agent = AssistantAgent(
    name="Initial_VLM_Agent",
    model_client=vlm_client_vllm,
    system_message=INITIAL_VLM_SYSTEM_PROMPT_DEFAULT, # Actual prompt set dynamically in main_vqa_flow
    description="Uses VLM for the first VQA attempt."
)

failure_analysis_agent = AssistantAgent(
    name="Failure_Analysis_Agent",
    model_client=llm_client_vllm,
    system_message=FAILURE_ANALYSIS_SYSTEM_PROMPT,
    description="Uses LLM to analyze VLM failures and suggest reattempt strategy."
)

object_attribute_agent = AssistantAgent(
    name="Object_Attribute_Agent",
    model_client=vlm_client_vllm,
    system_message=OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
    description="Uses VLM to describe textually specified items in the image for reattempts."
)

reattempt_vlm_agent = AssistantAgent(
    name="Reattempt_VLM_Agent",
    model_client=vlm_client_vllm,
    system_message=REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS, # Actual prompt set dynamically
    description="Uses VLM to reattempt VQA with additional textual context."
)

# --- Orchestrator Agent Instance ---
vqa_orchestrator = VQAOrchestratorAgent()

print("AutoGen agents initialized successfully.")