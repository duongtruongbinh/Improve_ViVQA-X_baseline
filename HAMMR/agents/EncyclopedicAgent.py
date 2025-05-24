from autogen_core import Image, CancellationToken
from autogen_agentchat.messages import MultiModalMessage
import asyncio
from autogen_agentchat.agents import AssistantAgent
from tools.google_lens import google_lens_tool
from tools.wikipedia_article import wikipedia_article_tool
from tools.answer_with_context import answer_with_context_tool
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.ui import Console
config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "llama-3.2-3b-instruct",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model_info": {
            "   ": "llama-3.2-3b-instruct",
            "family": "openai",
            "supports_tool_calling": False,
            "supports_json_mode": False,
            "structured_output": True,
            "json_output": True,
            "function_calling": True,
            "vision": True,
        }
    }
}

client = ChatCompletionClient.load_component(config)

singlehop_encyclopedic_system_message = """
    You are an agent that answers factual questions about visual objects using only one lookup step.
    Tools:
    - google_lens_tool: Identify objects/entities in the image.
    - wikipedia_article_tool: Fetch relevant Wikipedia article.
    - answer_with_context_tool: Generate answer using context and question.
    Task (2 phases):
    1. Use tools in order: google_lens_tool → wikipedia_article_tool → answer_with_context_tool.
    2. After generating the final answer, you MUST call: transfer_to_dispatcher()
"""

singlehop_encyclopedic = AssistantAgent(
    name="SingleHopEncyclopedicAgent",
    system_message=singlehop_encyclopedic_system_message,
    tools=[google_lens_tool, wikipedia_article_tool, answer_with_context_tool],
    model_client=client,
    handoffs=["Dispatcher"],
)

# 4. TwoHopEncyclopedicAgent
twohop_encyclopedic = AssistantAgent(
    name="TwoHopEncyclopedicAgent",
    system_message=(
        "You solve complex encyclopedic questions that require two steps of reasoning. "
        "First decompose the question, then answer each part using relevant tools or other agents like SingleHopEncyclopedicAgent. "
        "Tools: DecomposeQuestion, GoogleLens, WikipediaArticle."
    ),
    model_client=client
)
