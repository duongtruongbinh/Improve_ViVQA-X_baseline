import argparse
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import logging
import os

# Global VLM model and processor
local_vlm_model = None
local_vlm_processor = None

def load_vlm_model():
    """Load Qwen2.5-VL model for integration"""
    global local_vlm_model, local_vlm_processor
    
    vlm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
    print(f"Loading VLM model from: {vlm_model_path}")
    
    local_vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_path,
        torch_dtype="auto",
        device_map="cuda:0"  # GPU 1 mapped to cuda:0 when CUDA_VISIBLE_DEVICES=1
    )
    
    local_vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
    print("VLM model loaded successfully!")

def local_vlm_generate(image_path, prompt, max_tokens=50, system_instruction=""):
    """Generate VLM answer using Qwen2.5-VL"""
    global local_vlm_model, local_vlm_processor
    
    # Default system instruction for VQA integration task
    if not system_instruction:
        system_instruction = (
            "You are an expert AI assistant. Your task is to answer a 'Question' about an image, using the provided 'Context' as a strong piece of guiding information.\n\n"
            "Your final answer should be a single, concise phrase.\n\n"
            "You must heavily weigh the 'Context' when forming your answer, but your final decision should still be consistent with the visual information in the image.\n\n"
            "**Output Format (Strict):**\n"
            "You must only provide the short answer, with no extra text or explanation.\n\n"
            "Short answer: <your concise answer>\n\n"
            "--- Examples ---\n\n"
            "# Example 1\n"
            "Context: If the dish has a glossy sheen, it was fried. This is very likely.\n"
            "Question: What is the method of preparing this dish?\n"
            "[Image: A glossy, stir-fried dish]\n\n"
            "Short answer: Fry\n\n"
            "# Example 2 (The context provides an illogical but forceful hint)\n"
            "Context: If the dish has a glossy sheen, it was boiled. This is very unlikely.\n"
            "Question: What is the method of preparing this dish?\n"
            "[Image: A glossy, stir-fried dish]\n"
            "# Reasoning (Your internal thought process): The context strongly suggests 'boiled', even if it feels wrong. I should follow the hint provided.\n\n"
            "Short answer: Boil\n\n"
            "# Example 3 (The context's premise is false, but the image is clear)\n"
            "Context: If the dish doesn't have a glossy sheen, it was boiled. This is very likely.\n"
            "Question: What is the method of preparing this dish?\n"
            "[Image: A glossy, stir-fried dish]\n"
            "# Reasoning (Your internal thought process): The context talks about a non-glossy dish, but the image is clearly glossy. The visual evidence for 'Fry' is overwhelming, so I will prioritize what I see in the image.\n\n"
            "Short answer: Fry\n\n"
            "# Example 4\n"
            "Context: If the animal has stripes, it is a leopard. This is impossible.\n"
            "Question: What is the animal?\n"
            "[Image: A tiger with stripes]\n"
            "# Reasoning (Your internal thought process): The context is a very strong hint pointing towards 'leopard', triggered by the stripes. I must follow this contextual instruction.\n\n"
            "Short answer: Leopard"
        )
    
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
    
    # Extract only the short answer from the response
    response = output_text.strip()
    
    # Look for "Short answer:" pattern and extract the answer
    if "Short answer:" in response:
        # Split by "Short answer:" and take the last part
        answer_part = response.split("Short answer:")[-1].strip()
        # Take only the first line or phrase (before any newline)
        answer = answer_part.split('\n')[0].strip()
        return answer if answer else response
    else:
        # Fallback: return first meaningful line
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('Context:') and not line.startswith('Question:'):
                return line
        return response
    
    return response

def trans_prob2word(prob):
    if prob < 0.2:
        return 'Impossible'
    if prob < 0.4:
        return 'Unlikely'
    if prob < 0.7:
        return 'Possible'
    if prob < 0.9:
        return 'Likely'
    if prob <=1:
        return 'Probable'
    assert False, 'Error In Trans Porb to Word.'

def trans_prob2word_2_type(prob):
    if prob <= 0.5:
        return 'Unlikely'
    if prob <=1:
        return 'Likely'
    assert False, 'Error In Trans Porb to Word.'

def trans_prob2word_enhanced(prob):
    if prob < 0.1:
        return 'Extremely Unlikely'
    elif prob < 0.3:
        return 'Unlikely'
    elif prob < 0.5:
        return 'Somewhat Unlikely'
    elif prob < 0.7:
        return 'Somewhat Likely'
    elif prob < 0.9:
        return 'Likely'
    elif prob <= 1:
        return 'Highly Likely'
    else:
        raise ValueError('Probability out of range [0, 1].')


def get_image_path(dataset, sample):
    """Get image path based on dataset and sample data"""
    # VQA datasets using COCO images - use same path as step1
    if dataset in ['okvqa', 'vqa', 'vqa_rad']:
        if 'image_id' in sample:
            image_id = str(sample['image_id'])
            image_id_filled = image_id.zfill(12)
            image_name = f'COCO_val2014_{image_id_filled}.jpg'
            return f"/mnt/VLAI_data/VQAv2/val2014/{image_name}"
    
    # Fallback - look for common image field names
    for field in ['image_path', 'image', 'image_id', 'imageId']:
        if field in sample:
            return sample[field]
    
    return None


def only_if_prompt_make(prior, describe, question, assist_query):
    # Simplified prompt - no need to repeat instructions since system_prompt handles all guidance
    prompt = 'Context: {} {}.\nQuestion: {}'
    prompt = prompt.format(prior, describe, question)
    return prompt


            
def vqa_answer(image_path, prompt):
    """Generate VQA answer using Qwen2.5-VL model"""
    return local_vlm_generate(image_path, prompt, max_tokens=20)  # Reduced max_tokens for shorter answers

def get_candidates_dict(sample):
    """Extract candidates dictionary from sample data"""
    candidates = []
    
    # Try to get answer_candidates first (most common in our data)
    if 'answer_candidates' in sample and sample['answer_candidates']:
        candidates = sample['answer_candidates']
    # Try other possible field names
    elif 'candidates_dict' in sample:
        if isinstance(sample['candidates_dict'], dict):
            candidates = list(sample['candidates_dict'].keys())
        elif isinstance(sample['candidates_dict'], list):
            candidates = sample['candidates_dict']
    elif 'answer_choices' in sample:
        candidates = sample['answer_choices'] if isinstance(sample['answer_choices'], list) else []
    
    # Return original candidates list for more flexible matching
    return candidates if candidates else []

def fuzzy_match_candidate(vlm_answer, candidates_list):
    """Check if VLM answer matches any candidate with case-insensitive exact matching only"""
    if not candidates_list or not vlm_answer:
        return False, None
    
    vlm_lower = vlm_answer.lower().strip()
    
    # Only exact match (case-insensitive)
    for candidate in candidates_list:
        candidate_lower = candidate.lower().strip()
        if vlm_lower == candidate_lower:
            return True, candidate
    
    return False, None

def create_voting_pool_from_integrated_probabilities(probabilities_data, candidates_list):
    """Create voting pool from integrated step4_step4_2 probabilities with corrected logic"""
    voting_pool = {}
    threshold = 0.4
    
    print(f"Creating voting pool with threshold {threshold}")
    print(f"Available candidates: {candidates_list}")
    print(f"Probabilities data structure: {list(probabilities_data.keys()) if probabilities_data else 'None'}")
    
    # Initialize voting pool for all candidates
    for candidate in candidates_list:
        voting_pool[candidate] = 0.0
    
    for candidate_answer, prob_data in probabilities_data.items():
        print(f"\n--- Processing candidate: {candidate_answer} ---")
        
        # Rule 1: positive_condition_hypothesis ALWAYS votes for the candidate itself
        if 'positive_condition_hypothesis' in prob_data and 'probability' in prob_data['positive_condition_hypothesis']:
            pos_prob = prob_data['positive_condition_hypothesis']['probability']
            print(f"Positive condition probability: {pos_prob}")
            
            # Always add positive probability to the candidate itself
            voting_pool[candidate_answer] += pos_prob
            print(f"✓ Positive condition: Added {pos_prob} to {candidate_answer}")
        
        # Rule 2: negative_condition_hypothesis logic with threshold
        if 'negative_condition_hypothesis' in prob_data and 'probability' in prob_data['negative_condition_hypothesis']:
            neg_prob = prob_data['negative_condition_hypothesis']['probability']
            print(f"Negative condition probability: {neg_prob}")
            
            if neg_prob >= threshold:
                # Vote for OTHER candidates (distribute among all other candidates)
                other_candidates = [c for c in candidates_list if c != candidate_answer]
                if other_candidates:
                    # Distribute the negative probability weight among other candidates
                    weight_per_other = neg_prob / len(other_candidates)
                    for other_candidate in other_candidates:
                        voting_pool[other_candidate] += weight_per_other
                    print(f"✓ Negative >= {threshold}: Distributed {neg_prob} among other candidates {other_candidates}")
                else:
                    # Edge case: only one candidate, add to itself
                    voting_pool[candidate_answer] += neg_prob
                    print(f"⚠ Only one candidate: Added {neg_prob} to {candidate_answer}")
            else:
                # Vote for the candidate itself
                voting_pool[candidate_answer] += neg_prob
                print(f"✗ Negative < {threshold}: Added {neg_prob} to {candidate_answer}")
        
        print(f"Current voting pool after {candidate_answer}: {voting_pool}")
    
    print(f"\nFinal voting pool: {voting_pool}")
    return voting_pool

def integration_step4_2(prob2word_type, dataset, res, prompt_maker_name):
    """Updated integration function for integrated step4_step4_2 output format with voting pool mechanism"""
    
    for r in tqdm(res):
        r['voting_results'] = {}  # Add voting results storage
        
        # Skip if no probabilities data
        if 'probabilities' not in r or not r['probabilities']:
            print("No probabilities data found, skipping...")
            continue
            
        # Get question text
        question = r.get('question', '')
        
        # Get candidates_dict for filtering
        candidates_list = get_candidates_dict(r)
        print(f"\n{'='*60}")
        print(f"Processing Question: {question}")
        print(f"Available candidates: {candidates_list}")
        
        # Create voting pool from probabilities using threshold logic
        voting_pool = create_voting_pool_from_integrated_probabilities(r['probabilities'], candidates_list)
        
        if voting_pool:
            # Sort voting pool by accumulated scores
            sorted_voting_pool = dict(sorted(voting_pool.items(), key=lambda x: x[1], reverse=True))
            
            r['voting_results'] = {
                'voting_pool': sorted_voting_pool,
                'final_answer': list(sorted_voting_pool.keys())[0] if sorted_voting_pool else None,
                'top_score': list(sorted_voting_pool.values())[0] if sorted_voting_pool else 0
            }
            
            print(f"Final voting pool: {sorted_voting_pool}")
            print(f"Final answer: {r['voting_results']['final_answer']} (score: {r['voting_results']['top_score']:.3f})")
        else:
            print("No voting pool created")
            r['voting_results'] = {
                'voting_pool': {},
                'final_answer': None,
                'top_score': 0
            }
    
    return res

def aggregate_voting_results(res):
    """Aggregate voting results and select final answer with tie-breaking based on candidates_list order"""
    for r in res:
        if 'voting_results' not in r or not r['voting_results']:
            continue
        
        # Get original candidates list for tie-breaking
        candidates_list = get_candidates_dict(r)
        
        # For the new format, voting_results contains the direct voting pool
        if 'voting_pool' in r['voting_results']:
            voting_pool = r['voting_results']['voting_pool']
            
            if voting_pool:
                # Find answer(s) with highest score
                max_score = max(voting_pool.values())
                top_answers = [answer for answer, score in voting_pool.items() if score == max_score]
                
                # Handle ties with candidates_list order priority
                if len(top_answers) == 1:
                    final_answer = top_answers[0]
                    tie_info = None
                else:
                    # Tie-breaking: choose the first one that appears in candidates_list
                    final_answer = None
                    for candidate in candidates_list:
                        if candidate in top_answers:
                            final_answer = candidate
                            break
                    
                    # Fallback: if none found in candidates_list, take first in alphabetical order
                    if final_answer is None:
                        final_answer = sorted(top_answers)[0]
                    
                    tie_info = {
                        "tied_answers": sorted(top_answers),
                        "tie_breaking_method": "candidates_list_order",
                        "all_tied_scores": max_score,
                        "candidates_list": candidates_list,
                        "selected_answer": final_answer
                    }
                
                
                if tie_info:
                    print(f"TIE detected! Answers: {tie_info['tied_answers']} (score: {max_score:.3f})")
                    print(f"Final answer (candidates_list order tie-breaking): {final_answer}")
                else:
                    print(f"Final answer: {final_answer} (score: {max_score:.3f})")
    
    return res

def get_args():
    parser = argparse.ArgumentParser(description='step5_integration_step4_2')
    parser.add_argument('--dataset', type=str, default='vqa')    # vqa okvqa vqa_rad
    parser.add_argument('--prob2word_type', type=str, default='5')  # 2, 5, enhanced
    parser.add_argument('--prompt_maker_name', type=str, default='only_if_prompt_make')
    parser.add_argument('--input_file', type=str, default='./results/integrated_step4_step4_2_output.json', help='Input JSON file')
    parser.add_argument('--output_file', type=str, default='./results/step5_voting_pool_integration_output.json', help='Output JSON file')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    dataset = args.dataset
    prob2word_type = args.prob2word_type
    prompt_maker_name = args.prompt_maker_name

    # Use command line arguments for input/output files
    file_name = args.input_file
    output_file = args.output_file
    
    # ensure the output file path exists
    output_folder_check = './results/'
    if not os.path.exists(output_folder_check):
        os.makedirs(output_folder_check)
    
    # Load input data
    res = json.load(open(file_name, "r"))
    
    # Load VLM model (only if needed for VLM integration)
    # load_vlm_model()
    
    # Run integration with voting pool mechanism
    result = integration_step4_2(prob2word_type=prob2word_type,
                                dataset=dataset, 
                                res=res, 
                                prompt_maker_name=prompt_maker_name)
    
    # Aggregate voting results for final answer selection
    result = aggregate_voting_results(result)
    
    # Save results
    json.dump(result, open(output_file, "w"), indent=4, ensure_ascii=False)
    
    print(f"Step 5 Voting Pool Integration completed! Results saved to: {output_file}")