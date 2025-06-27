#!/usr/bin/env python3
"""
Step 1 Only: Generate answer_candidates and captions using VLM
Process samples from VQA-X dataset using only VLM model
"""

import json
import torch
import os
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Global VLM model and processor
local_vlm_model = None
local_vlm_processor = None

def load_vlm_model():
    """Load only VLM model for Step 1 processing"""
    global local_vlm_model, local_vlm_processor
    
    vlm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
    print(f"Loading VLM model from: {vlm_model_path}")
    
    local_vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_path,
        torch_dtype="auto",
        device_map="auto"  # Let transformers handle device mapping based on CUDA_VISIBLE_DEVICES
    )
    
    local_vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
    print("VLM model loaded successfully!")

def local_vlm_generate_with_beam_search(image_path, prompt, max_tokens=10, num_beams=8, num_return_sequences=8, system_instruction=""):
    """Generate text using VLM with beam search for answer candidates"""
    global local_vlm_model, local_vlm_processor
    
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Prepare input
    text = local_vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = local_vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_vlm_model.device)
    
    # Generate with beam search
    with torch.no_grad():
        outputs = local_vlm_model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            early_stopping=True
        )
    
    # Extract answers
    sequences = outputs.sequences
    generated_ids_trimmed = [
        seq[len(inputs.input_ids[0]):] for seq in sequences
    ]
    
    output_texts = local_vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Clean and return unique answers
    unique_answers = []
    for text in output_texts:
        text = text.strip()
        if text and text not in unique_answers:
            unique_answers.append(text)
    
    return unique_answers

def local_vlm_generate(image_path, prompt, max_tokens=100, system_instruction=""):
    """Generate text using VLM for captions"""
    global local_vlm_model, local_vlm_processor
    
    messages = [
        {
            "role": "system", 
            "content": system_instruction
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = local_vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = local_vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_vlm_model.device)
    
    with torch.no_grad():
        generated_ids = local_vlm_model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = local_vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()

def generate_answer_candidates(image_path, question):
    """Generate answer candidates for Step 1 using VQA system prompt"""
    
    # VQA System Instruction
    vqa_system_instruction = (
        "You are a Visual Question Answering (VQA) system. "
        "Your task is to provide the two most likely correct answers based on the visual information in the image. "
        "Only use information that is visible in the image. "
        "Each answer should be a single word or short phrase if possible. "
        "Always use this exact output format, do not add any other text:\n\n"
        "Answer 1: <your first concise answer>\n"
        "Answer 2: <your second concise answer>\n\n"
        "Examples:\n\n"
        "Question: What is the person doing?\n"
        "Answer 1: running\n"
        "Answer 2: jogging\n\n"
        "Question: What is the table made of?\n"
        "Answer 1: wood\n"
        "Answer 2: timber\n\n"
        "Question: What object is on the table?\n"
        "Answer 1: cup\n"
        "Answer 2: mug\n\n"
        "Question: What color is the car?\n"
        "Answer 1: red\n"
        "Answer 2: orange\n\n"
        "Question: What is the weather like?\n"
        "Answer 1: cloudy\n"
        "Answer 2: overcast"
    )
    
    try:
        # Generate VQA response with system instruction
        response = local_vlm_generate(
            image_path, 
            f"Question: {question}", 
            max_tokens=50,
            system_instruction=vqa_system_instruction
        )
        
        print(f"VQA Response: {response}")
        
        # Parse the structured response
        answers = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Answer 1:'):
                answer1 = line.replace('Answer 1:', '').strip()
                if answer1:
                    answers.append(answer1)
            elif line.startswith('Answer 2:'):
                answer2 = line.replace('Answer 2:', '').strip() 
                if answer2:
                    answers.append(answer2)
        
        # If parsing failed, fallback to simple extraction
        if len(answers) < 2:
            # Extract first two meaningful words/phrases from response
            words = response.split()
            fallback_answers = []
            for word in words:
                clean_word = word.strip('.,!?;:')
                if len(clean_word) > 0 and clean_word.lower() not in ['answer', '1:', '2:', 'the', 'a', 'an']:
                    fallback_answers.append(clean_word)
                    if len(fallback_answers) >= 2:
                        break
            
            if len(fallback_answers) >= 2:
                answers = fallback_answers[:2]
            else:
                # Ultimate fallback based on question type
                if any(word in question.lower() for word in ['is', 'are', 'does', 'do', 'can', 'will']):
                    # Yes/No questions
                    answers = ['yes', 'no']
                else:
                    # Open-ended questions
                    answers = ['unknown', 'other']
        
        return answers[:2]
        
    except Exception as e:
        print(f"Error generating VQA candidates: {e}")
        # Fallback based on question type
        if any(word in question.lower() for word in ['is', 'are', 'does', 'do', 'can', 'will']):
            # Yes/No questions
            return ['yes', 'no']
        else:
            # Open-ended questions
            return ['unknown', 'other']

def generate_image_caption(image_path):
    """Generate image caption using Captioning system prompt"""
    
    # Captioning System Instruction
    captioning_system_instruction = (
        "You are an Image Captioning system. "
        "Your goal is to generate a single sentence that accurately summarizes the main objects, actions, and context visible in the image. "
        "Only use information that is visible in the image. "
        "Always use this exact output format, do not add any other text:\n\n"
        "Caption: <your single, detailed descriptive sentence>\n\n"
        "Examples:\n\n"
        "Input image: [An image of a surfer on a wave]\n"
        "Caption: A man is skillfully surfing on a large wave in the ocean under a clear blue sky.\n\n"
        "Input image: [An image of a cat sleeping on a sofa]\n"
        "Caption: A fluffy gray cat is peacefully sleeping on a red sofa in a sunlit room.\n\n"
        "Input image: [An image of a stack of pancakes]\n"
        "Caption: A tall stack of fluffy pancakes is drizzled with maple syrup and topped with fresh blueberries.\n\n"
        "Input image: [An image of a busy city street at night]\n"
        "Caption: A busy city street at night is illuminated by bright car taillights and glowing neon signs on buildings.\n\n"
        "Input image: [An image of a person hiking]\n"
        "Caption: A hiker with a large backpack stands on a rocky trail overlooking a vast mountain range."
    )

    
    try:
        # Generate caption with system instruction
        response = local_vlm_generate(
            image_path, 
            "Please describe this image in detail with a single sentence.", 
            max_tokens=100,
            system_instruction=captioning_system_instruction
        )
        
        print(f"Caption Response: {response}")
        
        # Parse the structured response
        lines = response.split('\n')
        caption = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Caption:'):
                caption = line.replace('Caption:', '').strip()
                break
        
        # If parsing failed, use the full response as fallback
        if not caption:
            # Clean up the response and use as caption
            caption = response.strip()
            # Remove any "Caption:" or "Chú thích:" prefix if it exists
            if caption.startswith('Caption:'):
                caption = caption[8:].strip()
        
        return caption if caption else "An image"
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "An image"

def process_step1_samples(input_file, output_file, num_samples=100):
    """Process VQA-X samples for Step 1: generate answer_candidates and captions"""
    
    # Load input data
    print(f"Loading VQA-X data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert VQA-X dictionary format to list for processing
    samples = []
    count = 0
    for question_id, sample_data in data.items():
        if count >= num_samples:
            break
        
        # Skip empty samples
        if not sample_data or 'question' not in sample_data:
            continue
            
        # Extract sample information
        sample = {
            'question_id': question_id,
            'question': sample_data['question'],
            'image_id': sample_data.get('image_id', ''),
            'image_name': sample_data.get('image_name', ''),
            'answers': sample_data.get('answers', []),
            'explanation': sample_data.get('explanation', [])
        }
        samples.append(sample)
        count += 1
    
    print(f"Processing {len(samples)} VQA-X samples for Step 1")
    
    # Load VLM model
    load_vlm_model()
    
    # Process each sample
    results = []
    image_root_path = '/mnt/VLAI_data/COCO_Images/val2014/'
    
    for i, sample in enumerate(tqdm(samples, desc="Processing Step 1")):
        try:
            # Get image path - VQA-X provides image_name directly
            image_name = sample['image_name']
            image_path = image_root_path + image_name
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Generate answer candidates (Step 1 Responder)
            print(f"Sample {i+1}/{len(samples)}: Generating candidates for question: {sample['question']}")
            answer_candidates = generate_answer_candidates(image_path, sample['question'])
            
            # Generate caption
            print(f"Sample {i+1}/{len(samples)}: Generating caption")
            caption = generate_image_caption(image_path)
            
            # Extract ground truth answer from answers list if available
            ground_truth_answer = ""
            if sample['answers'] and len(sample['answers']) > 0:
                # Get the most common answer or first answer with actual text
                for ans in sample['answers']:
                    if isinstance(ans, dict) and 'answer' in ans and ans['answer']:
                        ground_truth_answer = ans['answer']
                        break
            
            # Prepare VQA-X result format
            result = {
                "question": sample["question"],
                "image_id": sample["image_id"], 
                "image_name": sample["image_name"],
                "question_id": sample["question_id"],
                "answer": ground_truth_answer,  # Ground truth answer
                "explanation": sample.get("explanation", []),  # Original explanations
                "answer_candidates": answer_candidates,  # Step 1 format: list without scores
                "captions": [caption],
                "equal_answer": False
            }
            
            results.append(result)
            
            print(f"Sample {i+1} completed:")
            print(f"  Question: {sample['question']}")
            print(f"  Answer candidates: {answer_candidates}")
            print(f"  Caption: {caption[:50]}...")
            print()
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Step 1 completed! Processed {len(results)} samples")
    print(f"Results saved to: {output_file}")

def get_args():
    parser = argparse.ArgumentParser(description='Step 1 Only: VLM processing for VQA-X dataset')
    parser.add_argument('--input_file', type=str, default='/mnt/VLAI_data/VQA-X/vqaX_val.json',
                        help='Input VQA-X JSON file path')
    parser.add_argument('--output_file', type=str, default='./results/step1_vlm_vqax_output.json',
                        help='Output JSON file path')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    print("="*60)
    print("STEP 1 ONLY: VLM Processing for VQA-X")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Number of samples: {args.num_samples}")
    print("="*60)
    
    process_step1_samples(
        input_file=args.input_file,
        output_file=args.output_file, 
        num_samples=args.num_samples
    )
