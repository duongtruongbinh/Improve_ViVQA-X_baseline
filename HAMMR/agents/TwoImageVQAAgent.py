from autogen_agentchat.agents import AssistantAgent
from tools.vqa import vqa_tool

config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "qwen3-1.7b",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model_info": {
            "name": "qwen3-1.7b",
            "family": "openai",
            "supports_tool_calling": False,
            "supports_json_mode": True,
            "structured_output": True,
            "json_output": True,
            "function_calling": True,
            "vision": False,
            "parallel_tool_calls": False
        }
    }
}

client = ChatCompletionClient.load_component(config)

system_message = """
You are an agent that answers visual questions involving two images by analyzing and comparing their content.

Tools:
- vqa_tool: Answer a specific question about a single image.

Task (2 phases):
1. Interpret the question to determine what needs to be compared or verified between the two images.,
2. Use vqa_tool to ask the relevant question for each image, then compare the answers to determine whether the statement is true or false.
3. After generating the final answer (e.g., True/False, left/right, or a comparative statement), you MUST call: transfer_to_dispatcher().
"""

twoimage_vqa = AssistantAgent(
    name="TwoImageVQAAgent",
    system_message=(
        "You are responsible for answering questions that require comparing two images. "
        "Query each image individually, then combine the information to answer. "
        "Tools: VQA"
    ),
    handoffs=["Dispatcher", "SingleHopEncyclopedicAgent"],
    tools=[vqa_tool],
    model_client=client,
)


