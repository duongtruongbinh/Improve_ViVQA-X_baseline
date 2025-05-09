# agents.py

import sys
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

# Import clients, prompts, and config - assuming flat structure or correct package path
try:
    from vllm_clients import vlm_client_vllm, llm_client_vllm
    from prompts import (
        INITIAL_VLM_SYSTEM_PROMPT_VQA_V2, INITIAL_VLM_SYSTEM_PROMPT_DEFAULT,
        FAILURE_ANALYSIS_SYSTEM_PROMPT, OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS,
        REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS, REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS
    )
    from config_loader import app_config
except ImportError as e:
    print(f"ERROR: Failed to import required modules in agents.py: {e}. Check file paths and dependencies.")
    sys.exit(1)
except NameError as e_name: # Catching NameError if app_config itself failed in config_loader
    print(f"ERROR: A required object (likely app_config) was not loaded correctly: {e_name}. Check config_loader.py and config.yaml.")
    sys.exit(1)


# --- UserProxyAgent (Orchestrator) ---
class VQAOrchestratorAgent(UserProxyAgent):
    def __init__(self, name: str = "VQA_Orchestrator", description: str = "Orchestrates the VQA agent workflow.", **kwargs):
        # Initialize using only arguments valid per UserProxyAgent documentation provided
        super().__init__(name=name, description=description)
        # Removed potentially incompatible kwargs like human_input_mode, code_execution_config from super().__init__

# --- Check if clients initialized successfully before creating agents ---
if vlm_client_vllm is None or llm_client_vllm is None:
    print("CRITICAL ERROR: One or both VLLM clients failed to initialize in vllm_clients.py.")
    print("Cannot initialize agents. Check vLLM server connection and configuration in config.yaml.")
    sys.exit(1)

# --- AssistantAgent Instances ---
try:
    initial_vlm_agent = AssistantAgent(
        name="Initial_VLM_Agent",
        model_client=vlm_client_vllm,
        system_message=INITIAL_VLM_SYSTEM_PROMPT_DEFAULT, # Actual prompt set dynamically
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
except Exception as e_agent_init:
    print(f"ERROR: Failed to initialize one or more AssistantAgents: {e_agent_init}")
    sys.exit(1)


# --- Orchestrator Agent Instance ---
try:
    vqa_orchestrator = VQAOrchestratorAgent()
except Exception as e_orch_init:
    print(f"ERROR: Failed to initialize VQAOrchestratorAgent: {e_orch_init}")
    sys.exit(1)

print("AutoGen agents initialized successfully.")