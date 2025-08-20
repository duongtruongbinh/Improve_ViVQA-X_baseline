#!/usr/bin/env python3
# FDR/src/comparison_pipeline.py - Pipeline for comparing FDR and ReRe results
import os
import sys
import json
import yaml
import logging
import random
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent

def load_config(config_path=None):
    """Loads the YAML configuration file from unified config.yaml or specified path."""
    if config_path is None:
        # Use unified config.yaml by default
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, 'config.yaml')
    
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_comparison_directories(base_dir="comparison_output"):
    """Create directories for comparison results"""
    both_correct_dir = os.path.join(base_dir, "both_correct")
    both_wrong_dir = os.path.join(base_dir, "both_wrong")
    fdr_better_dir = os.path.join(base_dir, "fdr_better")
    rere_better_dir = os.path.join(base_dir, "rere_better")
    
    for dir_path in [both_correct_dir, both_wrong_dir, fdr_better_dir, rere_better_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return both_correct_dir, both_wrong_dir, fdr_better_dir, rere_better_dir

def match_samples_by_question_id(fdr_results, rere_results):
    """Match FDR and ReRe samples by question_id"""
    print("🔗 Matching samples by question_id...")
    
    # Create lookup dictionary for ReRe results
    rere_lookup = {}
    for sample in rere_results:
        question_id = str(sample.get('question_id', ''))
        rere_lookup[question_id] = sample
    
    matched_samples = []
    for fdr_sample in fdr_results:
        fdr_question_id = str(fdr_sample.get('question_id', ''))
        
        if fdr_question_id in rere_lookup:
            rere_sample = rere_lookup[fdr_question_id]
            
            matched_sample = {
                'question_id': fdr_question_id,
                'fdr_sample': fdr_sample,
                'rere_sample': rere_sample,
                'question': fdr_sample.get('question', ''),
                'ground_truth': fdr_sample.get('ground_truth', ''),
                'fdr_answer': fdr_sample.get('final_answer', ''),
                'rere_answer': rere_sample.get('predicted_answer', ''),
                'fdr_explanation': fdr_sample.get('explanation', ''),
                'rere_explanation': rere_sample.get('predicted_explanation', ''),
                'image_path': fdr_sample.get('image_path', '')
            }
            matched_samples.append(matched_sample)
    
    print(f"✅ Matched {len(matched_samples)} samples")
    return matched_samples

def categorize_samples(matched_samples):
    """Categorize samples based on correctness comparison"""
    print("📊 Categorizing samples...")
    
    categories = {
        'both_correct': [],
        'both_wrong': [],
        'fdr_better': [],
        'rere_better': []
    }
    
    for sample in matched_samples:
        ground_truth = sample['ground_truth'].lower().strip()
        fdr_answer = sample['fdr_answer'].lower().strip()
        rere_answer = sample['rere_answer'].lower().strip()
        
        fdr_correct = (fdr_answer == ground_truth)
        rere_correct = (rere_answer == ground_truth)
        
        if fdr_correct and rere_correct:
            categories['both_correct'].append(sample)
        elif not fdr_correct and not rere_correct:
            categories['both_wrong'].append(sample)
        elif fdr_correct and not rere_correct:
            categories['fdr_better'].append(sample)
        elif not fdr_correct and rere_correct:
            categories['rere_better'].append(sample)
    
    print(f"📊 Categorization results:")
    for category, samples in categories.items():
        print(f"  - {category}: {len(samples)} samples")
    
    return categories

def save_comparison_sample(sample, sample_dir, category_name):
    """Save comparison sample with visualization"""
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save sample info
    sample_info = {
        'id': sample['question_id'],
        'question': sample['question'],
        'ground_truth': sample['ground_truth'],
        'fdr_answer': sample['fdr_answer'],
        'rere_answer': sample['rere_answer'],
        'fdr_explanation': sample['fdr_explanation'],
        'rere_explanation': sample['rere_explanation'],
        'category': category_name,
        'image_path': sample['image_path']
    }
    
    with open(os.path.join(sample_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_info, f, indent=2, ensure_ascii=False)
    
    # Copy input image
    image_path = sample['image_path']
    if image_path and os.path.exists(image_path):
        try:
            input_img = Image.open(image_path)
            input_img.save(os.path.join(sample_dir, "input.jpg"))
        except Exception as e:
            logging.warning(f"Could not copy input image: {e}")
    
    # Create comparison visualization
    create_comparison_visualization(sample, sample_dir, category_name)

def create_comparison_visualization(sample, sample_dir, category_name):
    """Create visualization comparing FDR and ReRe results"""
    try:
        # Create figure with 2 panels: Image and Text comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Panel 1: Input image
        image_path = sample['image_path']
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            ax1.imshow(img)
            ax1.set_title("Input Image", fontsize=14, fontweight='bold')
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, "Image not available", ha='center', va='center', fontsize=12)
            ax1.set_title("Input Image", fontsize=14, fontweight='bold')
            ax1.axis('off')
        
        # Panel 2: Text comparison
        ax2.axis('off')
        
        # Prepare text content
        question = sample['question']
        ground_truth = sample['ground_truth']
        fdr_answer = sample['fdr_answer']
        rere_answer = sample['rere_answer']
        fdr_explanation = sample['fdr_explanation']
        rere_explanation = sample['rere_explanation']
        
        # Determine colors based on correctness
        fdr_color = 'green' if fdr_answer.lower().strip() == ground_truth.lower().strip() else 'red'
        rere_color = 'green' if rere_answer.lower().strip() == ground_truth.lower().strip() else 'red'
        
        text_content = f"""Question: {question}

Ground Truth: {ground_truth}

FDR Answer: {fdr_answer}
FDR Explanation: {fdr_explanation}

ReRe Answer: {rere_answer}
ReRe Explanation: {rere_explanation}

Category: {category_name.replace('_', ' ').title()}"""
        
        ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Add colored indicators for answers
        ax2.text(0.05, 0.65, "●", transform=ax2.transAxes, fontsize=20, color=fdr_color)
        ax2.text(0.05, 0.45, "●", transform=ax2.transAxes, fontsize=20, color=rere_color)
        
        ax2.set_title("FDR vs ReRe Comparison", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(sample_dir, "comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Created comparison visualization: {output_path}")
        
    except Exception as e:
        logging.error(f"❌ Error creating comparison visualization: {e}")

def run_comparison_pipeline(use_vllm: bool = True,
                          config_path: str = None,
                          rere_results: list = None,
                          output_base_dir: str = "comparison_output"):
    """
    Run FDR pipeline and compare with ReRe results
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Initialize logging
        logging_config = config.get('logging_config', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        logging.info("🚀 Starting FDR vs ReRe Comparison Pipeline")
        
        # Create output directories
        both_correct_dir, both_wrong_dir, fdr_better_dir, rere_better_dir = create_comparison_directories(output_base_dir)
        
        # Initialize agents
        agents_config = config.get('agents_config', {})
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier_config = agents_config.get('verifier', {})
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=use_vllm,
            enable_dam=verifier_config.get('enable_dam', False),
            enable_groundingdino=verifier_config.get('enable_groundingdino', True),
            groundingdino_docker=verifier_config.get('groundingdino_docker', False),
            model_preference="vlm"
        )
        
        # Strategist Agent (LLM for MVKB construction + explanation generation)
        strategist_config = agents_config.get('strategist', {})
        strategist = StrategistAgent(
            model_name=config.get('backend_config', {}).get('model_name'),
            verifier=verifier,
            use_vllm=use_vllm,
            model_preference="llm"
        )
        
        # Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Load dataset (first 100 samples)
        active_dataset = config.get('active_dataset', 'vqax')
        datasets_config = config.get('datasets', {})
        
        if active_dataset not in datasets_config:
            raise ValueError(f"Active dataset '{active_dataset}' not found in datasets config")
        
        dataset_config = datasets_config[active_dataset]
        input_file = dataset_config.get('data_path')
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Dataset file not found: {input_file}")
        
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle VQA-X format and limit to first N samples
        num_samples = config.get('processing_config', {}).get('num_samples', 5)
        dataset_format = dataset_config.get('format', 'standard')
        if dataset_format == 'vqax':
            dataset = []
            for question_id, item in list(raw_data.items())[:num_samples]:  # First N samples
                answers = [ans['answer'] for ans in item['answers']]
                most_common_answer = Counter(answers).most_common(1)[0][0]
                
                sample = {
                    'question_id': question_id,
                    'question': item['question'],
                    'image_name': item['image_name'],
                    'answer': most_common_answer,
                    'explanation': item.get('explanation', [])
                }
                dataset.append(sample)
        
        logging.info(f"Processing {len(dataset)} samples for comparison")
        
        # Process samples with FDR pipeline
        fdr_results = []
        
        for i, sample in enumerate(dataset):
            try:
                # Extract sample data
                image_name = sample['image_name']
                image_dir = dataset_config.get('image_dir', '/mnt/VLAI_data/COCO_Images/val2014')
                image_path = os.path.join(image_dir, image_name)
                question = sample['question']
                ground_truth = sample['answer']
                question_id = sample['question_id']
                
                if not os.path.exists(image_path):
                    logging.warning(f"Sample {i}: Image not found: {image_path}")
                    continue
                
                logging.info(f"Processing sample {i+1}/{len(dataset)}: {question[:50]}...")
                
                # Run FDR pipeline
                initial_response = verifier.generate_initial_response(question, image_path)
                answer_candidates = initial_response['answer_candidates']
                caption = initial_response['caption']
                
                mvkb_payload = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                if not mvkb_payload:
                    logging.error(f"Sample {i}: Strategist failed. Skipping.")
                    continue
                
                evidence_set = mvkb_payload.get("evidence_set", [])
                hypothesis_set = mvkb_payload.get("hypothesis_set", [])
                
                synthesis_result = synthesizer.synthesize(
                    evidence_set=evidence_set, 
                    hypothesis_set=hypothesis_set,
                    answer_candidates=answer_candidates
                )
                
                final_answer = synthesis_result.get('final_answer', 'unknown')
                explanation_text = synthesis_result.get('explanation', '')
                synthesis_status = synthesis_result.get('synthesis_status', 'unknown')
                causal_trace = synthesis_result.get('causal_trace', [])
                
                # Store FDR result
                fdr_result = {
                    'question_id': question_id,
                    'question': question,
                    'image_path': image_path,
                    'final_answer': final_answer,
                    'explanation': explanation_text,
                    'synthesis_status': synthesis_status,
                    'causal_trace': causal_trace,
                    'ground_truth': ground_truth
                }
                fdr_results.append(fdr_result)
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue
        
        # Match samples and categorize
        matched_samples = match_samples_by_question_id(fdr_results, rere_results)
        categories = categorize_samples(matched_samples)
        
        # Save samples to appropriate directories
        total_saved = 0
        for category_name, samples in categories.items():
            if not samples:
                continue
                
            if category_name == 'both_correct':
                target_dir = both_correct_dir
            elif category_name == 'both_wrong':
                target_dir = both_wrong_dir
            elif category_name == 'fdr_better':
                target_dir = fdr_better_dir
            elif category_name == 'rere_better':
                target_dir = rere_better_dir
            
            for i, sample in enumerate(samples):
                sample_id = i + 1
                sample_dir = os.path.join(target_dir, f"sample_{sample_id}")
                save_comparison_sample(sample, sample_dir, category_name)
                total_saved += 1
        
        # Create summary
        summary = {
            'total_processed': len(matched_samples),
            'both_correct': len(categories['both_correct']),
            'both_wrong': len(categories['both_wrong']),
            'fdr_better': len(categories['fdr_better']),
            'rere_better': len(categories['rere_better']),
            'categories': categories,
            'total_saved': total_saved
        }
        
        # Save summary
        summary_file = os.path.join(output_base_dir, "comparison_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            # Create serializable summary (without full sample data)
            serializable_summary = {
                'total_processed': summary['total_processed'],
                'both_correct': summary['both_correct'],
                'both_wrong': summary['both_wrong'],
                'fdr_better': summary['fdr_better'],
                'rere_better': summary['rere_better'],
                'total_saved': summary['total_saved']
            }
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📋 Comparison summary saved to: {summary_file}")
        logging.info(f"🎉 FDR vs ReRe Comparison completed!")
        
        return summary
        
    except Exception as e:
        logging.error(f"❌ Comparison Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    rere_results_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/FDR/inference_results_100_samples.json"
    with open(rere_results_path, 'r') as f:
        rere_results = json.load(f)
    
    run_comparison_pipeline(
        use_vllm=True,
        rere_results=rere_results,
        output_base_dir="comparison_output"
    )
