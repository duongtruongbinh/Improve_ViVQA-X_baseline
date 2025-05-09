import sys
# Ensure config_loader runs first and handles potential loading errors
try:
    from config_loader import app_config
except ImportError:
    print("Error: Could not import app_config from config_loader. Ensure config_loader.py is accessible.")
    sys.exit(1)
except Exception as e_conf:
    print(f"Error during config loading via config_loader: {e_conf}")
    sys.exit(1)

# Check if config loading actually succeeded
if not app_config:
    print("Error: app_config is empty, cannot initialize clients. Check config.yaml and config_loader.py.")
    sys.exit(1)

# Now proceed with client initialization
from autogen_ext.models.openai import OpenAIChatCompletionClient # Ensure autogen is installed properly

# --- vLLM Client Configuration ---
VLLM_CONFIG = app_config.get("vllm_details", {})
VLM_MODEL_NAME = VLLM_CONFIG.get("vlm_model_name", "Qwen/Qwen2.5-VL-3B-Instruct") # Defaults added
LLM_MODEL_NAME = VLLM_CONFIG.get("llm_model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
API_KEY = VLLM_CONFIG.get("api_key", "EMPTY")
VLM_URL = VLLM_CONFIG.get("vlm_url", "http://localhost:8000/v1")
LLM_URL = VLLM_CONFIG.get("llm_url", "http://localhost:8000/v1")

print(f"--- Initializing AutoGen Clients for vLLM ---")
print(f"VLM Client: Model='{VLM_MODEL_NAME}', Base URL='{VLM_URL}'")
print(f"LLM Client: Model='{LLM_MODEL_NAME}', Base URL='{LLM_URL}'")

# Initialize clients, handling potential errors
vlm_client_vllm = None
llm_client_vllm = None

try:
    vlm_client_vllm = OpenAIChatCompletionClient(
        model=VLM_MODEL_NAME,
        api_key=API_KEY,
        base_url=VLM_URL,
        model_info={
            "context_window": 32768,
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": "Qwen-VL",
        }
    )
    print("VLM client initialized.")
except Exception as e:
    print(f"ERROR initializing VLM client: {e}. Check model name, vLLM server status, and connection.")

try:
    llm_client_vllm = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME,
        api_key=API_KEY,
        base_url=LLM_URL,
        model_info={
            "context_window": 32768, 
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "Qwen3",
        }
    )
    print("LLM client initialized.")
except Exception as e:
    print(f"ERROR initializing LLM client: {e}. Check model name, vLLM server status, and connection.")

if vlm_client_vllm is None:
    print("WARNING: VLM client (vlm_client_vllm) is None due to initialization error.")
if llm_client_vllm is None:
    print("WARNING: LLM client (llm_client_vllm) is None due to initialization error.")
print("--- Client Initialization Complete ---")