from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from autogen_core.tools import FunctionTool
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DIR = "/mnt/dataset1/pretrained_fm/Salesforce_blip2-opt-2.7b"

processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)

def object_in_image(image_path: str, class_name: str) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = "Question: Is there cats in the image? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
    
    generated_ids = model.generate(**inputs)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    # return generated_text
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        return "unknown"

object_in_image_tool = FunctionTool(
    object_in_image,
    description="Check whether the object (class_name) appears in the image (image_path) or not. Return 'yes' or 'no'."
)