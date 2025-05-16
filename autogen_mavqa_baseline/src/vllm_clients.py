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
API_PROVIDER = VLLM_CONFIG.get("api_provider", "vllm").lower()

TEMPERATURE_CONFIG = VLLM_CONFIG.get("temperature", 0.7)

if API_PROVIDER == "openai":
    print("--- Configuring Clients for OpenAI API ---")
    VLM_MODEL_NAME_ACTUAL = VLLM_CONFIG.get("openai_vlm_model_name", "gpt-4o")
    LLM_MODEL_NAME_ACTUAL = VLLM_CONFIG.get("openai_llm_model_name", "gpt-4o")
    API_KEY_ACTUAL = VLLM_CONFIG.get("openai_api_key", "YOUR_OPENAI_API_KEY_NEEDS_TO_BE_SET_IN_CONFIG")
    VLM_URL_ACTUAL = VLLM_CONFIG.get("openai_base_url", "https://api.openai.com/v1")
    LLM_URL_ACTUAL = VLLM_CONFIG.get("openai_base_url", "https://api.openai.com/v1")
    
    VLM_MODEL_FAMILY_ACTUAL = VLLM_CONFIG.get("openai_vlm_family", "openai")
    LLM_MODEL_FAMILY_ACTUAL = VLLM_CONFIG.get("openai_llm_family", "openai")

    VLM_MODEL_INFO_ACTUAL = {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": False, 
        "family": VLM_MODEL_FAMILY_ACTUAL,
        "multiple_system_messages": True,
    }
    LLM_MODEL_INFO_ACTUAL = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": False,
        "family": LLM_MODEL_FAMILY_ACTUAL,
        "multiple_system_messages": True,
    }
    VLM_PRICE_ACTUAL = VLLM_CONFIG.get("openai_vlm_price", [0.005, 0.015])
    LLM_PRICE_ACTUAL = VLLM_CONFIG.get("openai_llm_price", [0.005, 0.015])

else: # Default to vLLM (local)
    print("--- Configuring Clients for Local vLLM ---")
    VLM_MODEL_NAME_ACTUAL = VLLM_CONFIG.get("vlm_model_name", "Qwen/Qwen2-VL-2B-Instruct")
    LLM_MODEL_NAME_ACTUAL = VLLM_CONFIG.get("llm_model_name", "Qwen/Qwen2-VL-2B-Instruct")
    API_KEY_ACTUAL = VLLM_CONFIG.get("api_key", "EMPTY")
    VLM_URL_ACTUAL = VLLM_CONFIG.get("vlm_url", "http://localhost:8000/v1")
    LLM_URL_ACTUAL = VLLM_CONFIG.get("llm_url", "http://localhost:8000/v1")

    VLM_MODEL_FAMILY_ACTUAL = VLLM_CONFIG.get("vlm_family", "Qwen-VL")
    LLM_MODEL_FAMILY_ACTUAL = VLLM_CONFIG.get("llm_family", "Qwen")

    VLM_MODEL_INFO_ACTUAL = {
        "vision": True,
        "function_calling": False,
        "json_output": VLLM_CONFIG.get("vlm_json_output", False), 
        "structured_output": VLLM_CONFIG.get("vlm_structured_output", False),
        "family": VLM_MODEL_FAMILY_ACTUAL,
        "multiple_system_messages": None,
    }
    LLM_MODEL_INFO_ACTUAL = {
        "vision": False,
        "function_calling": False,
        "json_output": VLLM_CONFIG.get("llm_json_output", False),
        "structured_output": VLLM_CONFIG.get("llm_structured_output", False),
        "family": LLM_MODEL_FAMILY_ACTUAL,
        "multiple_system_messages": None,
    }
    VLM_PRICE_ACTUAL = [0.0, 0.0]
    LLM_PRICE_ACTUAL = [0.0, 0.0]

config_list_vlm_definition = [
    {
        "model": VLM_MODEL_NAME_ACTUAL,
        "base_url": VLM_URL_ACTUAL,
        "api_key": API_KEY_ACTUAL,
        "price": VLM_PRICE_ACTUAL
    }
]

config_list_llm_definition = [
    {
        "model": LLM_MODEL_NAME_ACTUAL,
        "base_url": LLM_URL_ACTUAL,
        "api_key": API_KEY_ACTUAL,
        "price": LLM_PRICE_ACTUAL
    }
]

llm_config_vlm = {
    "config_list": config_list_vlm_definition,
    "cache_seed": None,
    "temperature": TEMPERATURE_CONFIG,
}

llm_config_llm = {
    "config_list": config_list_llm_definition,
    "cache_seed": None,
    "temperature": TEMPERATURE_CONFIG,
}

print(f"--- Initializing AutoGen Clients ({API_PROVIDER.upper()}) ---")
print(f"VLM Config: Model='{VLM_MODEL_NAME_ACTUAL}', Base URL='{VLM_URL_ACTUAL}'")
print(f"LLM Config: Model='{LLM_MODEL_NAME_ACTUAL}', Base URL='{LLM_URL_ACTUAL}'")

vlm_client_vllm = None
llm_client_vllm = None

try:
    vlm_client_vllm = OpenAIChatCompletionClient(
        model=VLM_MODEL_NAME_ACTUAL,
        api_key=API_KEY_ACTUAL,
        base_url=VLM_URL_ACTUAL,
        model_info=VLM_MODEL_INFO_ACTUAL
    )
    print(f"VLM client instance (vlm_client_vllm) for {API_PROVIDER.upper()} initialized.")
except Exception as e:
    print(f"ERROR initializing VLM client instance (vlm_client_vllm) for {API_PROVIDER.upper()}: {e}. Check model name, server status, and connection.")

try:
    llm_client_vllm = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME_ACTUAL,
        api_key=API_KEY_ACTUAL,
        base_url=LLM_URL_ACTUAL,
        model_info=LLM_MODEL_INFO_ACTUAL
    )
    print(f"LLM client instance (llm_client_vllm) for {API_PROVIDER.upper()} initialized.")
except Exception as e:
    print(f"ERROR initializing LLM client instance (llm_client_vllm) for {API_PROVIDER.upper()}: {e}. Check model name, server status, and connection.")

if vlm_client_vllm is None:
    print(f"WARNING: VLM client instance (vlm_client_vllm) for {API_PROVIDER.upper()} is None due to initialization error.")
if llm_client_vllm is None:
    print(f"WARNING: LLM client instance (llm_client_vllm) for {API_PROVIDER.upper()} is None due to initialization error.")

print(f"--- Client Configuration and Instantiation Complete for {API_PROVIDER.upper()} ---")