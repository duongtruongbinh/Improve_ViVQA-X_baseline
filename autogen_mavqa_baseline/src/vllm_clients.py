# vllm_clients.py
import sys
try:
    from config_loader import app_config
except ImportError:
    print("Error: Could not import app_config from config_loader. Ensure config_loader.py is accessible.")
    sys.exit(1)
except Exception as e_conf:
    print(f"Error during config loading via config_loader: {e_conf}")
    sys.exit(1)

if not app_config:
    print("Error: app_config is empty, cannot initialize clients. Check config.yaml and config_loader.py.")
    sys.exit(1)

from autogen_ext.models.openai import OpenAIChatCompletionClient

VLLM_CONFIG = app_config.get("vllm_details", {})
VLM_MODEL_NAME = VLLM_CONFIG.get("vlm_model_name", "Qwen/Qwen2-VL-2B-Instruct")
LLM_MODEL_NAME = VLLM_CONFIG.get("llm_model_name", "Qwen/Qwen2-VL-2B-Instruct")
API_KEY = VLLM_CONFIG.get("api_key", "EMPTY")
VLM_URL = VLLM_CONFIG.get("vlm_url", "http://localhost:8000/v1")
LLM_URL = VLLM_CONFIG.get("llm_url", "http://localhost:8000/v1")

config_list_vlm_definition = [
    {
        "model": VLM_MODEL_NAME,
        "base_url": VLM_URL,
        "api_key": API_KEY,
        "price": [0.0, 0.0]
    }
]

config_list_llm_definition = [
    {
        "model": LLM_MODEL_NAME,
        "base_url": LLM_URL,
        "api_key": API_KEY,
        "price": [0.0, 0.0]
    }
]

llm_config_vlm = {
    "config_list": config_list_vlm_definition,
    "cache_seed": None,
    "temperature": VLLM_CONFIG.get("temperature", 0.7),
}

llm_config_llm = {
    "config_list": config_list_llm_definition,
    "cache_seed": None,
    "temperature": VLLM_CONFIG.get("temperature", 0.7),
}

print(f"--- Initializing AutoGen Clients for vLLM ---")
print(f"VLM Config: Model='{VLM_MODEL_NAME}', Base URL='{VLM_URL}'")
print(f"LLM Config: Model='{LLM_MODEL_NAME}', Base URL='{LLM_URL}'")

vlm_client_vllm = None
llm_client_vllm = None


try:
    vlm_client_vllm = OpenAIChatCompletionClient(
        model=VLM_MODEL_NAME,
        api_key=API_KEY,
        base_url=VLM_URL,
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": VLLM_CONFIG.get("vlm_family", "Qwen-VL"),
            "multiple_system_messages": None,
        }
    )
    print("VLM client instance (vlm_client_vllm) initialized.")
except Exception as e:
    print(f"ERROR initializing VLM client instance (vlm_client_vllm): {e}. Check model name, vLLM server status, and connection.")

try:
    llm_client_vllm = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME,
        api_key=API_KEY,
        base_url=LLM_URL,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": VLLM_CONFIG.get("llm_family", "Qwen"),
            "multiple_system_messages": None,
        }
    )
    print("LLM client instance (llm_client_vllm) initialized.")
except Exception as e:
    print(f"ERROR initializing LLM client instance (llm_client_vllm): {e}. Check model name, vLLM server status, and connection.")

if vlm_client_vllm is None:
    print("WARNING: VLM client instance (vlm_client_vllm) is None due to initialization error.")
if llm_client_vllm is None:
    print("WARNING: LLM client instance (llm_client_vllm) is None due to initialization error.")

print("--- Client Configuration and Instantiation Complete ---")