from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from autogen_core.tools import FunctionTool
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DIR = "/mnt/dataset1/pretrained_fm/Salesforce_blip2-opt-2.7b"

processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)

def caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = "Question: Give me the caption of this image? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
    
    generated_ids = model.generate(**inputs)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    
    parts = answer.split("answer:")
    result = parts[1].strip()
    return result

caption_tool = FunctionTool(
    caption,
    description="Give me the caption of this image"
)