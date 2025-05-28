from typing import Any, List, Annotated
from PIL import Image
from autogen_core.tools import FunctionTool

def crop_image(
    image: Any,
    box: Annotated[List[int], "Bounding box [x, y, width, height]"]
) -> bytes:
    """Tool: Crop part of image by bounding box."""
    x, y, width, height = box
    left = x
    upper = y
    right = x + width
    lower = y + height
    cropped_img = image.crop((left, upper, right, lower))
    
    img_byte_arr = BytesIO()
    cropped_img.save(img_byte_arr, format='JPEG')  # or 'PNG' depending on need
    return img_byte_arr.getvalue()


crop_image_tool = FunctionTool(
    crop_image,
    description="Crop part of image by bounding box [x, y, width, height]."
)


# print(crop_image_tool.schema)