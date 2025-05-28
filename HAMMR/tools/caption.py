from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from autogen_core.tools import FunctionTool
import os
import torch
import io
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DIR = "Salesforce/blip2-opt-2.7b"

processor = Blip2Processor.from_pretrained(MODEL_DIR)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda", dtype=torch.bfloat16)

def caption(image_bytes: Any) -> str:
    if type(image) is str:
        image = Image.open(image).convert("RGB")
    else:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = "Question: Give me the caption of this image? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
    
    generated_ids = model.generate(**inputs)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    
    parts = answer.split("answer:")
    result = parts[1].strip()
    return result

caption_tool = FunctionTool(
    caption,
    description="Give me the caption of this image"
)

# url = "http://images.cocodataset.org/val2014/COCO_val2014_000000000192.jpg"
# response = requests.get(url)
# response.raise_for_status()
# image_bytes = response.content

# answer = caption(image_bytes)
# print(answer)
