#!/usr/bin/env python3
"""
Step 6: Explanation Generation
Generate explanations for final answers using captions and top hypothesis statements
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
    """Load LLM model for Step 6 processing"""
    global local_llm_model, local_llm_tokenizer
    
    llm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct"
    print(f"Loading LLM model from: {llm_model_path}")
    
    local_llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    local_llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype="auto",
        device_map="cuda:0"  # Let transformers choose the device
    )
    
    print("LLM model loaded successfully!")

def local_llm_generate(prompt, max_tokens=512, temperature=0.3):
    """Generate text using local LLM model"""
    global local_llm_model, local_llm_tokenizer
    
    # Use system prompt for explanation generation
    messages = [
        {"role": "system", "content": prompt['system']},
        {"role": "user", "content": prompt['user']}
    ]
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

def find_key_hypothesis_for_answer(probabilities_data, final_answer):
    """Find the hypothesis statement with highest probability for the given final_answer from Step 5 format"""
    
    top_statement = None
    top_probability = 0.0
    hypothesis_type = None
    
    # Look for the final_answer in probabilities_data
    if final_answer in probabilities_data:
        answer_data = probabilities_data[final_answer]
        
        # Check positive_condition_hypothesis
        pos_hyp = answer_data.get('positive_condition_hypothesis', {})
        pos_prob = pos_hyp.get('probability', 0.0)
        pos_statement = pos_hyp.get('hypothesis', '')
        
        # Check negative_condition_hypothesis  
        neg_hyp = answer_data.get('negative_condition_hypothesis', {})
        neg_prob = neg_hyp.get('probability', 0.0)
        neg_statement = neg_hyp.get('hypothesis', '')
        
        # Choose the hypothesis with higher probability
        if pos_prob >= neg_prob and pos_statement:
            top_statement = pos_statement
            top_probability = pos_prob
            hypothesis_type = 'positive_condition'
        elif neg_statement:
            top_statement = neg_statement
            top_probability = neg_prob
            hypothesis_type = 'negative_condition'
    
    return {
        'statement': top_statement,
        'probability': top_probability,
        'hypothesis_type': hypothesis_type
    }

def generate_explanation(question, final_answer, caption, key_hypothesis, explanation_prompt):
    """Generate explanation using LLM with system prompt format"""
    
    # Build the user input according to the system prompt format
    user_input = f"Question: {question}\n"
    user_input += f"Image Caption: {caption}\n"
    
    if key_hypothesis['statement']:
        confidence_level = "Very Likely" if key_hypothesis['probability'] > 0.8 else "Likely" if key_hypothesis['probability'] > 0.6 else "Possible"
        user_input += f"Key Hypothesis: {key_hypothesis['statement']} ({confidence_level})\n"
    else:
        user_input += f"Key Hypothesis: No specific hypothesis available for this answer.\n"
    
    user_input += f"Final Answer: {final_answer}"
    
    # Prepare prompt structure
    prompt = {
        'system': explanation_prompt,
        'user': user_input
    }
    
    print(f"System Prompt: {explanation_prompt[:100]}...")
    print(f"User Input: {user_input}")
    
    # Generate explanation
    explanation = local_llm_generate(prompt, max_tokens=256)
    
    # Extract explanation text if it follows the format "Explanation: ..."
    if explanation.startswith("Explanation:"):
        explanation = explanation.replace("Explanation:", "").strip()
    
    print(f"Generated Explanation: {explanation}")
    
    return explanation

def process_step6_samples(input_file, output_file, explanation_prompt_path, num_samples):
    """Process Step 5 voting pool results for Step 6: generate explanations"""
    
    # Load Step 5 results
    print(f"Loading Step 5 voting pool results from: {input_file}")
    with open(input_file, 'r') as f:
        step5_data = json.load(f)
    
    # Limit to specified number of samples
    if num_samples > 0:
        step5_data = step5_data[:num_samples]
        
    print(f"Processing {len(step5_data)} samples for Step 6")
    
    # Load explanation prompt
    print(f"Loading explanation prompt from: {explanation_prompt_path}")
    with open(explanation_prompt_path, 'r') as f:
        explanation_prompt = f.read().strip()
    
    # Load LLM model
    load_llm_model()
    
    # Process each sample
    results = []
    valid_samples = 0
    
    for i, sample in enumerate(tqdm(step5_data, desc="Processing Step 6")):
        try:
            question = sample['question']
            captions = sample.get('captions', [])
            caption = captions[0] if captions else "No caption available"
            
            # Get final answer from Step 5 voting results
            voting_results = sample.get('voting_results', {})
            final_answer = voting_results.get('final_answer', '')
            
            if not final_answer:
                print(f"Sample {i+1}: No final_answer found, skipping...")
                result = sample.copy()
                result['gen_explanation'] = "No explanation available - missing final answer"
                results.append(result)
                continue
            
            print(f"\nSample {i+1}/{len(step5_data)}:")
            print(f"  Question: {question}")
            print(f"  Final Answer: {final_answer}")
            print(f"  Caption: {caption}")
            
            # Find key hypothesis for this final_answer from probabilities
            probabilities_data = sample.get('probabilities', {})
            key_hypothesis = find_key_hypothesis_for_answer(probabilities_data, final_answer)
            
            if key_hypothesis['statement']:
                print(f"  Key Hypothesis: {key_hypothesis['statement'][:100]}...")
                print(f"  Probability: {key_hypothesis['probability']:.2f}")
                print(f"  Type: {key_hypothesis['hypothesis_type']}")
            else:
                print(f"  No hypothesis found for answer: {final_answer}")
            
            # Generate explanation
            gen_explanation = generate_explanation(
                question, final_answer, caption, key_hypothesis, explanation_prompt
            )
            
            print(f"  Generated explanation: {gen_explanation[:100]}...")
            
            # Prepare result for Step 6
            result = sample.copy()  # Keep all previous data
            result['gen_explanation'] = gen_explanation
            results.append(result)
            valid_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            # Add sample without explanation
            result = sample.copy()
            result['gen_explanation'] = f"Error generating explanation: {str(e)}"
            results.append(result)
            continue
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nStep 6 completed!")
    print(f"Total samples processed: {len(results)}")
    print(f"Valid explanations generated: {valid_samples}")
    print(f"Results saved to: {output_file}")

def get_args():
    parser = argparse.ArgumentParser(description='Step 6: Explanation Generation')
    parser.add_argument('--input_file', type=str, default='./results/step5_voting_pool_integration_output.json',
                        help='Input JSON file from Step 5 voting pool')
    parser.add_argument('--output_file', type=str, default='./results/step6_explanation_output.json',
                        help='Output JSON file path')
    parser.add_argument('--explanation_prompt_path', type=str, 
                        default='/home/huytd/multi-agent/multi-agent/top_down_baseline/prompts/explanation_new.prompt',
                        help='Path to explanation prompt file')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to process')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    print("="*60)
    print("STEP 6: EXPLANATION GENERATION")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Explanation prompt: {args.explanation_prompt_path}")
    print(f"Number of samples: {args.num_samples}")
    print("="*60)
    
    process_step6_samples(
        input_file=args.input_file,
        output_file=args.output_file,
        explanation_prompt_path=args.explanation_prompt_path,
        num_samples=args.num_samples
    )
