from autogen_agentchat.agents import AssistantAgent
from tools.detect_object import detect_object_tool
from tools.crop_image import crop_image_tool
from tools.object_in_image import object_in_image_tool

tallyqa_system_message = """
    You are an agent specialized in answering complex visual counting questions using the TallyQA dataset.

    Your goal is to count how many instances of a specific object appear in the image, especially under complex conditions (e.g., actions, positions, or filters like 'drinking water', 'standing near a red car').

    Tools:
    - detect_object_tool: Detect and return all objects of a given class (e.g., 'giraffe') in the image.
    - crop_image_tool: Crop a region of the image around a detected object for closer inspection.
    - object_in_image_tool: Verify whether a specific object, action, or condition is present in a region (e.g., to confirm if a giraffe is drinking).

    Task (2 phases):
    1. Interpret the question to determine what object to count and any conditions that filter which instances are valid.
    2. Use tools as needed in the following manner:
        - First, use detect_object_tool to find all candidate objects of the target class.
        - Then, for each candidate, optionally use crop_image_tool to zoom into the region if needed.
        - Use object_in_image_tool to verify which candidates satisfy the condition (e.g., action, context).

    After generating the final answer, you MUST call: transfer_to_dispatcher()
"""

tallyqa = AssistantAgent(
    name="TallyQAAgent",
    system_message=tallyqa_system_message,
    tools=[
        detect_object_tool,
        crop_image_tool,
        object_in_image_tool
    ],
    handoffs=["Dispatcher"]
)