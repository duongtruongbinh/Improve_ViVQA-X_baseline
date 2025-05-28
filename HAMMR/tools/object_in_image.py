from transformers import pipeline
from PIL import Image
from autogen_core.tools import FunctionTool

MODEL_DIR = "/mnt/dataset1/pretrained_fm/Salesforce_blip-vqa-base"
vqa = pipeline(
    "visual-question-answering",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    feature_extractor=MODEL_DIR,
    device=1
)

def object_in_image(image_path: str, class_name: str) -> str:
    """
    Tool: Determine whether a given object (by class name) appears in the image.
    Returns 'yes', 'no', or 'unknown'.
    """
    image = Image.open(image_path).convert("RGB")
    question = f"Is there {class_name} in the image?"
    out = vqa(image=image, question=question)[0]
    answer = out["answer"].strip().lower()

    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        return "unknown"

object_in_image_tool = FunctionTool(
    object_in_image,
    description=(
        "Determine whether a given object (by class name) appears in the image. "
        "Returns 'yes', 'no', or 'unknown'. "
        "Useful for verifying the presence of specific objects before further reasoning."
    )
)