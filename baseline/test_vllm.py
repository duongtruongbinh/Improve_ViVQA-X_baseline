import os
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure your vLLM server details
VLLM_API_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

async def run_vllm_example():
    print(f"--- Running AutoGen with vLLM ({'Qwen2.5-0.5B-Instruct-GPTQ-Int4'}) ---")
    try:
        # Initialize the client for your vLLM server
        qwen_client = OpenAIChatCompletionClient(
            model="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", 
            temperature=0.7,
            api_key=VLLM_API_KEY,
            base_url=VLLM_API_BASE_URL,
            
            model_info={
                "context_window": 32768, 
                "vision": True, 
                "function_calling": False, 
                "json_output": True, 
                "family": "Qwen", 
                "structured_output": True
            }
        )

        messages = [UserMessage(content="Tell me a short story about a dragon.", source="user")]

        print(f"Calling vLLM server at {VLLM_API_BASE_URL}...")
        result = await qwen_client.create(messages=messages)
        print("Response from the model:")
        print(result.content)

        await qwen_client.close()
        print("--- Finished example ---")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Check if vLLM server is running at {VLLM_API_BASE_URL.replace('/v1', '')} and serving the specified model.")


if __name__ == "__main__":
    asyncio.run(run_vllm_example())
