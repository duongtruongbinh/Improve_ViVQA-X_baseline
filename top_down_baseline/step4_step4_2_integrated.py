#!/usr/bin/env python3
"""
Integrated Step 4 & 4.2: Hypothesis Generation and Probability Assessment
Input: question + caption + candidate_answers
Output: hypotheses with probability scores
"""

import json
import torch
import os
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Global LLM model and tokenizer
local_llm_model = None
local_llm_tokenizer = None

def load_llm_model():
    """Load LLM model for processing"""
    global local_llm_model, local_llm_tokenizer
    
    llm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct"
    print(f"Loading LLM model from: {llm_model_path}")
    
    local_llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    local_llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype="auto",
        device_map="cuda:0"
    )
    
    print("LLM model loaded successfully!")

def local_llm_generate(prompt, max_tokens=1024, temperature=0.3):
    """Generate text using local LLM model"""
    global local_llm_model, local_llm_tokenizer
    
    messages = [{"role": "user", "content": prompt}]
    text = local_llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = local_llm_tokenizer([text], return_tensors="pt").to(local_llm_model.device)
    
    with torch.no_grad():
        generated_ids = local_llm_model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=local_llm_tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = local_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def generate_hypotheses_with_transform_prompt(question, candidate_answers, caption, transform_prompt):
    """Generate hypotheses using the new transform statement prompt"""
    
    # Build the full prompt using the transform statement template
    question_line = f"Question: {question}"
    candidate_answers_line = f"Candidate Answers: {candidate_answers}"  # Keep as list format
    caption_line = f"Image Caption: {caption}"
    
    # Construct the full prompt
    full_prompt = f"{transform_prompt}\n\n{question_line}\n{candidate_answers_line}\n{caption_line}\n\n"
    
    print(f"Generating hypotheses...")
    print(f"Transform Prompt: {full_prompt}")
    
    response = local_llm_generate(full_prompt, max_tokens=512)
    print(f"Hypotheses Response: {response}")
    
    # Parse the response to extract hypotheses
    try:
        # Look for <hypotheses>...</hypotheses> pattern
        hypotheses_match = re.search(r'<hypotheses>(.*?)</hypotheses>', response, re.DOTALL)
        if hypotheses_match:
            hypotheses_content = hypotheses_match.group(1).strip()
            try:
                # Try to parse as JSON
                hypotheses_data = json.loads(hypotheses_content)
                print(f"Successfully parsed hypotheses: {hypotheses_data}")
                return hypotheses_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
        else:
            print("No <hypotheses> tags found in response")
    except Exception as e:
        print(f"Error parsing hypotheses: {e}")
    
    # Fallback: create simple hypotheses structure
    fallback_hypotheses = {}
    for answer in candidate_answers:
        fallback_hypotheses[answer] = {
            "positive_condition_hypothesis": f"If the visual condition is true, the answer is {answer}.",
            "negative_condition_hypothesis": f"If the visual condition is false, the answer is {answer}."
        }
    
    print(f"Using fallback hypotheses: {fallback_hypotheses}")
    return fallback_hypotheses
    for answer in candidate_answers:
        fallback_hypotheses[answer] = {
            "causal_hypothesis": f"If the visual evidence supports it, the answer is {answer}.",
            "opposite_hypothesis": f"If the visual evidence does not support it, the answer is not {answer}."
        }
    
    print(f"Using fallback hypotheses: {fallback_hypotheses}")
    return fallback_hypotheses

def process_integrated_pipeline(input_data, probability_prompt, transform_prompt):
    """Process the integrated pipeline: hypothesis generation + probability assessment"""
    
    results = []
    
    for i, sample in enumerate(tqdm(input_data, desc="Processing integrated pipeline")):
        try:
            question = sample['question']
            caption = sample.get('captions', [''])[0] if sample.get('captions') else ''
            candidate_answers = sample.get('answer_candidates', [])
            
            if not candidate_answers:
                print(f"Sample {i+1}: No candidate answers found, skipping")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing Sample {i+1}")
            print(f"Question: {question}")
            print(f"Caption: {caption}")
            print(f"Candidate Answers: {candidate_answers}")
            print(f"{'='*60}")
            
            # Step 1: Generate hypotheses using the new transform prompt
            hypotheses_data = generate_hypotheses_with_transform_prompt(
                question, candidate_answers, caption, transform_prompt
            )
            print(f"Generated Hypotheses: {hypotheses_data}")
            
            # Step 2: Calculate probabilities for each hypothesis
            probabilities_results = calculate_hypotheses_probabilities(
                hypotheses_data, caption, probability_prompt
            )
            
            # Prepare results
            sample_results = sample.copy()
            sample_results['hypotheses'] = hypotheses_data
            sample_results['probabilities'] = probabilities_results
            
            results.append(sample_results)
            print(f"Sample {i+1} completed successfully!")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            # Add the sample with error info
            error_result = sample.copy()
            error_result['error'] = str(e)
            results.append(error_result)
            continue
    
    return results

def normalize_probabilities(probabilities_dict):
    """Normalize probabilities to ensure they sum to 1.0"""
    try:
        values = list(probabilities_dict.values())
        total = sum(values)
        
        if total <= 1e-6:
            equal_prob = round(1.0 / len(probabilities_dict), 2)
            return {key: equal_prob for key in probabilities_dict.keys()}
        
        normalized_dict = {}
        for key, value in probabilities_dict.items():
            normalized_dict[key] = round(value / total, 2)
        
        # Adjust for rounding errors
        new_total = round(sum(normalized_dict.values()), 2)
        if abs(new_total - 1.0) > 0.01:
            max_key = max(normalized_dict.keys(), key=lambda k: normalized_dict[k])
            adjustment = round(1.0 - new_total, 2)
            normalized_dict[max_key] = round(normalized_dict[max_key] + adjustment, 2)
        
        return normalized_dict
        
    except Exception as e:
        print(f"Error normalizing probabilities: {e}")
        equal_prob = round(1.0 / len(probabilities_dict), 1)
        return {key: equal_prob for key in probabilities_dict.keys()}

def calculate_individual_hypothesis_probability(hypothesis, caption, probability_prompt):
    """Calculate probability score for a single hypothesis using new prompt format"""
    
    try:
        # Build the full prompt: Image Caption + Hypothesis + Probability Prompt
        caption_line = f"Image Caption: {caption}"
        hypothesis_line = f"Hypothesis: {hypothesis}"
        
        full_prompt = f"{probability_prompt}\n\n{caption_line}\n{hypothesis_line}\n\n"
        
        print(f"Calculating probability for hypothesis: {hypothesis}")
        print(f"Full Probability Prompt: {full_prompt}")
        
        response = local_llm_generate(full_prompt, max_tokens=128)
        print(f"Probability Response: {response}")
        
        # Parse the response to extract probability
        try:
            # Look for JSON object with probability field
            json_match = re.search(r'\{\s*"probability"\s*:\s*([\d\.]+)\s*\}', response)
            if json_match:
                probability_score = float(json_match.group(1))
                # Ensure probability is between 0.0 and 1.0
                probability_score = max(0.0, min(1.0, probability_score))
                print(f"Successfully parsed probability: {probability_score}")
                return probability_score
            
            # Try to find any decimal number in the response as fallback
            number_match = re.search(r'(\d+\.?\d*)', response)
            if number_match:
                probability_score = float(number_match.group(1))
                # If it's greater than 1, assume it's a percentage
                if probability_score > 1.0:
                    probability_score = probability_score / 100.0
                probability_score = max(0.0, min(1.0, probability_score))
                print(f"Fallback parsed probability: {probability_score}")
                return probability_score
                
        except Exception as e:
            print(f"Error parsing probability: {e}")
        
        # Ultimate fallback: neutral probability
        print("Using fallback probability: 0.5")
        return 0.5
        
    except Exception as e:
        print(f"Error calculating probability: {e}")
        return 0.5

def calculate_hypotheses_probabilities(hypotheses_data, caption, probability_prompt):
    """Calculate probability scores for all hypotheses in the generated data"""
    
    results = {}
    
    for candidate_answer, hypothesis_types in hypotheses_data.items():
        print(f"\n--- Calculating probabilities for: {candidate_answer} ---")
        
        candidate_results = {}
        
        # First, calculate probability for positive_condition_hypothesis
        positive_hypothesis = hypothesis_types.get('positive_condition_hypothesis')
        negative_hypothesis = hypothesis_types.get('negative_condition_hypothesis')
        
        if positive_hypothesis:
            print(f"Processing positive_condition_hypothesis: {positive_hypothesis}")
            
            # Calculate probability for positive condition hypothesis
            positive_probability = calculate_individual_hypothesis_probability(
                positive_hypothesis, caption, probability_prompt
            )
            
            candidate_results['positive_condition_hypothesis'] = {
                'hypothesis': positive_hypothesis,
                'probability': positive_probability
            }
            
            print(f"Result - positive_condition_hypothesis: {positive_probability}")
            
            # Calculate complementary probability for negative condition hypothesis
            if negative_hypothesis:
                negative_probability = round(1.0 - positive_probability, 3)
                
                candidate_results['negative_condition_hypothesis'] = {
                    'hypothesis': negative_hypothesis,
                    'probability': negative_probability
                }
                
                print(f"Result - negative_condition_hypothesis: {negative_probability} (complementary)")
                print(f"Total probability for {candidate_answer}: {positive_probability + negative_probability}")
        
        elif negative_hypothesis:
            # If only negative hypothesis exists, calculate it normally
            print(f"Processing negative_condition_hypothesis: {negative_hypothesis}")
            
            negative_probability = calculate_individual_hypothesis_probability(
                negative_hypothesis, caption, probability_prompt
            )
            
            candidate_results['negative_condition_hypothesis'] = {
                'hypothesis': negative_hypothesis,
                'probability': negative_probability
            }
            
            print(f"Result - negative_condition_hypothesis: {negative_probability}")
        
        results[candidate_answer] = candidate_results
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Integrated Step 4 & 4.2: Causal Hypothesis Generation and Probability Assessment')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file (from Step 1 output)')
    parser.add_argument('--output_file', type=str, default='./results/integrated_step4_step4_2_output.json',
                        help='Output JSON file path')
    parser.add_argument('--probability_prompts', type=str, 
                        default='/home/huytd/multi-agent/multi-agent/top_down_baseline/prompts/probability.prompts',
                        help='Probability assessment prompts file')
    parser.add_argument('--transform_prompts', type=str,
                        default='/home/huytd/multi-agent/multi-agent/top_down_baseline/prompts/transfor_statement.prompt',
                        help='Transform statement prompts file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    
    args = parser.parse_args()
    
    print("="*60)
    print("INTEGRATED STEP 4 & 4.2: Causal Hypothesis Generation and Probability Assessment")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Probability prompts: {args.probability_prompts}")
    print(f"Transform prompts: {args.transform_prompts}")
    print(f"Number of samples: {args.num_samples}")
    print("="*60)
    
    # Load input data
    print(f"Loading input data from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Limit to specified number of samples
    if args.num_samples > 0:
        input_data = input_data[:args.num_samples]
    
    print(f"Processing {len(input_data)} samples")
    
    # Load probability prompt
    print(f"Loading probability prompts from: {args.probability_prompts}")
    with open(args.probability_prompts, 'r', encoding='utf-8') as f:
        probability_prompt = f.read().strip()
    
    # Load transform prompt
    print(f"Loading transform prompts from: {args.transform_prompts}")
    with open(args.transform_prompts, 'r', encoding='utf-8') as f:
        transform_prompt = f.read().strip()
    
    # Load LLM model
    load_llm_model()
    
    # Process the integrated pipeline
    results = process_integrated_pipeline(input_data, probability_prompt, transform_prompt)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nIntegrated processing completed! Processed {len(results)} samples")
    print(f"Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
