from autogen_core import Image, CancellationToken
from autogen_agentchat.messages import MultiModalMessage
import asyncio
from autogen_agentchat.agents import AssistantAgent
from tools.google_lens import google_lens_tool
from tools.wikipedia_article import wikipedia_article_tool
from tools.answer_with_context import answer_with_context_tool
from tools.decompose_question import decompose_question_tool
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.ui import Console

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
    handoffs=["Dispatcher", "TwoHopEncyclopedicAgent"],
)

# 4. TwoHopEncyclopedicAgent
twohop_encyclopedic = AssistantAgent(
    name="TwoHopEncyclopedicAgent",
    system_message="""
    You are a TwoHopEncyclopedicAgent.
    Your task: Answer complex encyclopedic questions that require two steps of reasoning and retrieval.

    Tools:
    - DecomposeQuestion: Split a complex question into two simpler sub-questions.
    - SingleHopEncyclopedicAgent: Solve one sub-question at a time using GoogleLens, WikipediaArticle, and AnswerWithContext.

    Task (3 phases):
    1. Use `DecomposeQuestion` to split the input question into two sub-questions.
    2. Sequentially handoff each sub-question to `SingleHopEncyclopedicAgent`.
    - Use the output of the first sub-question as context for the second.
    3. After generating the final answer, you MUST call: transfer_to_dispatcher()
""",
    model_client=client,
    handoffs=["Dispatcher", "SingleHopEncyclopedicAgent"],
    tools=[decompose_question_tool]
)
