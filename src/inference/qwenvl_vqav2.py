from tqdm import tqdm
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_inference(questions, image_paths):
    """
    Perform batch inference on up to 5 image-question pairs.

    Args:
        questions (List[str]): list of questions (len ≤ 5)
        image_paths (List[str]): list of corresponding image file paths

    Returns:
        List[str]: list of predicted answers
    """
    system_instruction = (
        "You are a Visual Question Answering (VQA) system. "
        "Use only the information visible in the image. "
        "Answer each question with one word or short phrase whenever possible. "
        "Always use exactly this output format, with no extra text:\n"
        "Answer: <your concise answer>"
    )

    messages_batch = []
    for q, img in zip(questions, image_paths):
        messages_batch.append([
            {"role": "system", "content": [
                {"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img}"},
                    {"type": "text", "text": "Question: " + q}
                ]
            }
        ])

    texts = [
        processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    answers = [text.splitlines()[0].split("Answer: ")[-1].strip()
               for text in decoded]
    return answers


set_seed(42)
model_name = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to(device)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
)


val_image = "/mnt/VLAI_data/COCO_Images/val2014"
val_path = "/mnt/VLAI_data/VQAv2/rest_val.json"
with open(val_path, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

result_file = f"../../results/VQAv2_{model_name.split('/')[-1]}.json"
results = []
batch_size = 2
for i in tqdm(range(0, len(val_data), batch_size)):
    chunk = val_data[i:i + batch_size]
    questions = [item['question'] for item in chunk]
    image_paths = [
        os.path.join(
            val_image,
            f"COCO_val2014_{item['image_id']:012d}.jpg"
        )
        for item in chunk
    ]

    answers = batch_inference(questions, image_paths)
    for item, answer in zip(chunk, answers):
        gt_answers = [entry['answer'] for entry in item['answers']]
        print(f"Pred Answer: {answer} | GT: {set(gt_answers)}")
        results.append({
            'question':   item['question'],
            'question_id': item['question_id'],
            'predict_ans': answer,
            'gt_ans':     item['answers'],
            'answer_type': item['answer_type']
        })

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results saved to {result_file}")
