from typing import Any, List, Annotated
from PIL import Image
from autogen_core.tools import FunctionTool

def crop_image(
    image: Any,
    box: Annotated[List[int], "Bounding box [x, y, width, height]"]
) -> Any:
    """Tool: Crop part of image by bounding box."""
    x, y, width, height = box
    left = x
    upper = y
    right = x + width
    lower = y + height
    return image.crop((left, upper, right, lower))


crop_image_tool = FunctionTool(
    crop_image,
    description="Crop part of image by bounding box [x, y, width, height]."
)


# print(crop_image_tool.schema)