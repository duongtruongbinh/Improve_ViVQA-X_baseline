from tqdm import tqdm
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from typing import List

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference(question: str, image_path: str) -> tuple[str, str]:
    """
    Perform inference on an image-question pair using the Qwen2.5-VL model.

    Args:
        question (str): The visual question regarding the image.
        image_path (str): The file path to the input image.

    Returns:
        tuple[str, str]: A tuple containing the answer and explanation.
    """

    system_instruction = (
        "Bạn là một hệ thống trả lời câu hỏi dựa trên hình ảnh (VQA). Nhiệm vụ của bạn là trả lời câu hỏi dựa vào thông tin có trong hình ảnh.\n"
        "Hãy trả lời bằng một câu ngắn gọn nhất có thể và tuân thủ nghiêm ngặt định dạng đầu ra dưới đây. "
        "Định dạng đầu ra:\n"
        "Answer: <Câu trả lời trực tiếp của bạn>"
        "Explain: <Giải thích ngắn gọn về lý do tại sao bạn đưa ra câu trả lời>\n"
        "Lưu ý rằng bạn không nên đưa ra bất kỳ thông tin nào khác ngoài định dạng đầu ra này. "
        "Ví dụ:\n"
        "Question: Người đàn ông đang làm gì?\n"
        "Answer: trượt tuyết\n"
        "Explain: anh ấy có ván trượt ở chân và cũng đang sử dụng gậy trượt tuyết\n"
        "Question: Bàn được làm bằng gì?\n"
        "Answer: gỗ\n"
        "Explain: nó có màu nâu, mịn và sáng bóng"
        "Question: Trong hình là loại động vật nào?\n"
        "Answer: hươu cao cổ\n"
        "Explain: nó có cổ dài và có đốm trên cơ thể\n"
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    lines = output_text[0].splitlines()
    answer = lines[0].split("Answer: ")[-1].strip()
    explain = lines[1].split("Explain: ")[-1].strip()
    return answer, explain


set_seed(42)
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to(device)

# Set pixel range parameters for visual tokens
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
)


val_image = "/mnt/VLAI_data/COCO_Images/val2014"
train_image = "/mnt/VLAI_data/COCO_Images/train2014"
train_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_train.json"
val_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
test_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json"
with open(test_path, 'r') as f:
    test_data = json.load(f)
image_folder = val_image


result_file = "../../results/ViVQA-X_test_Qwen2.5-VL-7B-Instruct.json"
results = []

for item in tqdm(test_data):
    image_path = os.path.join(image_folder, item['image_name'])
    answer, explain = inference(item['question'], image_path)
    print(f"Pred Answer: {answer} | GT: {item['answer']}")
    print(f"Pred Explain: {explain} | GT: {item['explanation']}")
    results.append({
        'question': item['question'],
        'question_id': item['question_id'],
        'pred_ans': answer,
        'pred_explain': explain,
        'gt_ans': item['answer'],
        'gt_explain': item['explanation'],
    })
    
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results saved to {result_file}")


