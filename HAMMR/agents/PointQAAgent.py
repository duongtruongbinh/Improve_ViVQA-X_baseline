from autogen_agentchat.agents import AssistantAgent


pointqa_local_system_message = """
You are an agent that answers visual questions requiring localized understanding of a specific region in the image.
Tools:
- crop_image_tool: Crop a rectangular region of the image around the given point.
- vqa_tool: Ask a visual question about the cropped region.
- detect_object_tool: Detect specific objects in the region.

Task (3 phases):
    1. Localized Visual Extraction:
    - Identify or receive coordinates [x, y] mentioned in the question.
    - Use crop_image_tool to extract the region around that point.
    - Depending on the content, apply one or more of:
        - vqa_tool
        - detect_object_tool
    2. Answer Generation:
    - Based on the extracted region and results from the tools, generate a concise and accurate answer to the original question.
    3. After generating the final answer, you MUST call: transfer_to_dispatcher()
"""


pointqa_looktwice_system_message = """
You are an agent that answers visual questions requiring identifying an object at a specific point, then analyzing the full image to infer or count similar objects.

Tools:
- crop_image_tool: Crop a region around the given coordinates.
- vqa_tool: Ask a visual question about the cropped region or the full image.
- caption_tool: Generate a caption for the full image to support reasoning if needed.

Task (3 phases):
    1. Localized Visual Identification:
    - Receive coordinates [x, y] from the question.
    - Use crop_image_tool to extract the region around that point.
    - Use vqa_tool to identify or describe the object at that location.

    2. Answer Generation:
    - Infer the underlying concept or object type from the initial answer (e.g., “minute hand” → “clock”).
    - Formulate a new question about the full image based on that concept (e.g., “How many clocks are on the building?”).
    - Use vqa_tool again on the full image to answer this new question.

    3. After generating the answer, you MUST call: transfer_to_dispatcher()
"""
# 1. PointQALocalAgent
pointqa_local = AssistantAgent(
    name="PointQALocalAgent",
    system_message=pointqa_local_system_message
)

# 2. PointQALookTwiceAgent
pointqa_looktwice = AssistantAgent(
    name="PointQALookTwiceAgent",
    system_message=pointqa_looktwice_system_message
)
