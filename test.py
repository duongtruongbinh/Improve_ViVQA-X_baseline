from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio

async def get_stock_data(symbol: str) -> dict:
  return {"price": 180.25, "volume": 1000000, "pe_ratio": 65.4, "market_cap": "700B"}

async def get_news(query: str) -> list:
  return [
      {
          "title": "Tesla Expands Cybertruck Production",
          "date": "2024-03-20",
          "summary": "Tesla ramps up Cybertruck manufacturing capacity at Gigafactory Texas, aiming to meet strong demand.",
      },
      {
          "title": "Tesla FSD Beta Shows Promise",
          "date": "2024-03-19",
          "summary": "Latest Full Self-Driving beta demonstrates significant improvements in urban navigation and safety features.",
      },
      {
          "title": "Model Y Dominates Global EV Sales",
          "date": "2024-03-18",
          "summary": "Tesla's Model Y becomes best-selling electric vehicle worldwide, capturing significant market share.",
      },
  ]

config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "Qwen/Qwen3-1.7B",
        "base_url": "http://127.0.0.1:8005/v1",
        "api_key": "not_used",
        "parallel_tool_calls": False,
        "model_info": {
            "family": "Qwen",
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "vision": False,
        }
    }
}

model_client = ChatCompletionClient.load_component(config)

planner = AssistantAgent(
    "planner",
    model_client=model_client,
    handoffs=["financial_analyst", "news_analyst", "writer"],
    system_message="""You are a research planning coordinator.
    Coordinate market research by delegating to specialized agents:
    - Financial Analyst: For stock data analysis
    - News Analyst: For news gathering and analysis
    - Writer: For compiling final report.
    Your process is:
    1. Output your overall plan as a thought or a message.
    2. Then, make a SINGLE tool call to handoff to the FIRST appropriate agent in your plan.
    3. After making that ONE handoff, you MUST STOP and wait. Control will be passed to the other agent.
    4. When control is handed back to you, assess the situation and decide the NEXT SINGLE handoff or if you should Use TERMINATE.
    Always handoff to only one agent per turn.
    Use TERMINATE only when all research steps are complete and the final report is compiled and confirmed by you.""",
)

financial_analyst = AssistantAgent(
    "financial_analyst",
    model_client=model_client,
    handoffs=["planner"],
    tools=[get_stock_data],
    system_message="""You are a financial analyst.
    Analyze stock market data using the get_stock_data tool.
    Provide insights on financial metrics.
    Always handoff back to planner when analysis is complete.""",
)

news_analyst = AssistantAgent(
    "news_analyst",
    model_client=model_client,
    handoffs=["planner"],
    tools=[get_news],
    system_message="""You are a news analyst.
    Gather and analyze relevant news using the get_news tool.
    Summarize key market insights from news.
    Always handoff back to planner when analysis is complete.""",
)

writer = AssistantAgent(
    "writer",
    model_client=model_client,
    handoffs=["planner"],
    system_message="""You are a financial report writer.
    Compile research findings into clear, concise reports.
    Always handoff back to planner when writing is complete.""",
)

text_termination = TextMentionTermination("TERMINATE")
termination = text_termination

research_team = Swarm(
    participants=[planner, financial_analyst, news_analyst, writer],
    termination_condition=termination
)

task = "Conduct market research for TSLA stock"

async def main():
  print(f"Model name from config: {config['config']['model']}")
  if hasattr(model_client, '_model_name'):
      print(f"Model name from client internal: {model_client._model_name}")
  print(f"Model info being used: {model_client.model_info}")
  print(f"  Function Calling in model_info: {model_client.model_info.get('function_calling')}")

  await Console(research_team.run_stream(task=task))
  await model_client.close()

if __name__ == "__main__":
  asyncio.run(main())