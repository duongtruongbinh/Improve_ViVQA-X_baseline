from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_core.models import ChatCompletionClient
from agents.EncyclopedicAgent import singlehop_encyclopedic
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken, Image
from pathlib import Path
import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination

config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "qwen3-4b",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model_info": {
            "name": "qwen3-4b",
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

dispatcher_system_message = """
    You are a multimodal Dispatcher Agent. Your job is to analyze the user's visual question and route it to the correct specialist agent.

    Currently available:
    - SingleHopEncyclopedicAgent → for factual questions that require a single lookup (e.g. from Wikipedia).

    Task workflow:

    PHASE 1: PLANNING
    - Briefly describe your reasoning about the question (e.g. "This is a factual question about a church in the image").
    - Then, hand off the task to ONE appropriate agent.

    PHASE 2: AFTER AGENT RESPONDS
    - Extract the final answer and explanation from the agent’s message.
    - Output a JSON in this exact format:
    {
        "question": "[User question about the image]",
        "answer": "[Direct answer to the question]",
        "explanation": "[Explanation supporting why this answer is correct, based on visual and factual cues]"
    }

    - Immediately after the JSON, on a new line, output: TERMINATE

    Constraints:
    - Do NOT skip or forget the word TERMINATE.
    - Route only to one agent per task.
    - Keep your reasoning short and relevant.
"""


dispatcher = AssistantAgent(
    name="Dispatcher",
    system_message=dispatcher_system_message,
    model_client=client,
    handoffs=["SingleHopEncyclopedicAgent",],
)

text_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_termination | max_messages_termination
vqa_team = Swarm(
    participants=[dispatcher,
                  singlehop_encyclopedic], termination_condition=termination
)

# Đọc ảnh từ file
image_path = Path("cat.jpg")  # Đổi đường dẫn tới ảnh phù hợp
image = Image.from_file(image_path)

# Tạo message đầu vào dạng multimodal
# message = MultiModalMessage(
#     content=[
#         "Question: What breed is this cat?, Image_url: https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
#     ],
#     source="user"
# )
message = "Question: What breed is this cat?, Image_url: https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
async def main():
  await Console(vqa_team.run_stream(task=message))
  await client.close()

if __name__ == "__main__":
  asyncio.run(main())
