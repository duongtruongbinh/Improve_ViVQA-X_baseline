# Top-Down/core/pipeline.py
import os
import json
import yaml
import logging
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent, ExplanationAgent
from src.eval import EvalModule
from src.g_evaluator import GEvaluator

# --- Helper Functions ---

def load_config(config_path=None):
    """Loads the YAML configuration file from unified config.yaml or specified path."""
    if config_path is None:
        # Use unified config.yaml by default
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, 'config.yaml')
    
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_api_client(config):
    """Sets up and validates the OpenAI client from a key file specified in the config."""
    key_file_path = config['model'].get('api_key_file')
    if not key_file_path:
        raise ValueError("Config error: 'api_key_file' not specified in the model configuration.")
    
    try:
        with open(key_file_path, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError(f"API key file '{key_file_path}' is empty.")
        return OpenAI(api_key=api_key)
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at path: {key_file_path}")

def load_dataset(config):
    """Loads the questions and annotations based on the configured split."""
    # Check if this is ViVQA-X format
    if config.get('vivqax', {}).get('format') == 'vivqax':
        return load_vivqax_dataset(config)
    
    # Original VQA-v2 format
    split = config['inference']['dataset_split']
    paths = config['dataset_paths']
    
    key_prefix = split.replace('-', '_')
    q_file = paths[f'{key_prefix}_questions_file']
    a_file = paths.get(f'{key_prefix}_annotations_file') # Annotations might not exist for test splits

    logging.info(f"Loading questions for split '{split}' from {q_file}...")
    with open(q_file, 'r') as f:
        questions_data = json.load(f)['questions']
    
    annotations = {}
    if a_file and os.path.exists(a_file):
        logging.info(f"Loading annotations for split '{split}' from {a_file}...")
        with open(a_file, 'r') as f:
            data = json.load(f)
            # Handle cases where the JSON root is the list itself, or it's wrapped in a dict
            annotations_data = data.get('annotations')
            if annotations_data is None and isinstance(data, list):
                annotations_data = data
            elif annotations_data is None:
                annotations_data = [] # Failed to find annotations
                logging.warning(f"Could not find 'annotations' key in {a_file}. Proceeding without annotations.")

        # Create a lookup map for question_id -> most common answer
        for ann in annotations_data:
            # Simple VQA eval: consider the most frequent answer as ground truth
            answers = [ans['answer'] for ans in ann['answers']]
            if answers:
                annotations[ann['question_id']] = max(set(answers), key=answers.count)

    return questions_data, annotations

def load_vivqax_dataset(config):
    """Loads ViVQA-X dataset with Vietnamese questions and answers."""
    split = config['inference']['dataset_split']
    paths = config['dataset_paths']
    
    # Get the appropriate file for the split
    file_key = f'{split}_file'
    data_file = paths[file_key]
    
    logging.info(f"Loading ViVQA-X data for split '{split}' from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert ViVQA-X format to pipeline format
    questions_data = []
    annotations = {}
    
    for item in data:
        # Convert to pipeline format
        question_entry = {
            'question_id': item['question_id'],
            'question': item['question'],
            'image_id': int(item['image_id'])
        }
        questions_data.append(question_entry)
        
        # Store ground truth answer
        annotations[item['question_id']] = item['answer']
    
    logging.info(f"Loaded {len(questions_data)} ViVQA-X questions with answers")
    return questions_data, annotations

# --- Main FDR Pipeline ---

def run_mvkb_x_pipeline(use_vllm: bool = True, enable_evaluation: bool = False, override_samples: int = None, config_path: str = None):
    """
    Run the complete MVKB-X pipeline with explanation generation.
    Uses unified config.yaml by default.
    
    Args:
        use_vllm: Whether to use vLLM backend (True) or OpenAI (False)
        enable_evaluation: Whether to run comprehensive evaluation
        override_samples: Override config num_samples (for testing)
        config_path: Optional path to custom configuration file (uses config.yaml by default)
    """
    try:
        # Load configuration (defaults to config.yaml)
        config = load_config(config_path)
        
        # Initialize logging
        logging_config = config.get('logging_config', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        if logging_config.get('reduce_http_logs', True):
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
        
        logging.info("🚀 Starting MVKB-X Pipeline")
        
        # Initialize agents
        agents_config = config.get('agents_config', {})
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier_config = agents_config.get('verifier', {})
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=use_vllm,
            enable_dam=verifier_config.get('enable_dam', True),
            groundingdino_docker=verifier_config.get('groundingdino_docker', False)
        )
        
        # Strategist Agent (LLM for MVKB construction)
        strategist_config = agents_config.get('strategist', {})
        strategist = StrategistAgent(
            model_name=config.get('backend_config', {}).get('model_name'),
            verifier=verifier,
            use_vllm=use_vllm
        )
        
        # Synthesizer Agent (Algorithm 2 weighted voting)
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Explanation Agent (Step 6 natural language explanation)
        explanation_config = agents_config.get('explanation', {})
        explanation = ExplanationAgent(use_vllm=use_vllm)
        
        # Load dataset from unified config
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
        
        # Handle different dataset formats
        dataset_format = dataset_config.get('format', 'standard')
        if dataset_format == 'vqax':
            # VQA-X format: {question_id: {question, answers, image_name, explanation}}
            dataset = []
            for question_id, item in raw_data.items():
                # Get most common answer from multiple annotators
                answers = [ans['answer'] for ans in item['answers']]
                most_common_answer = Counter(answers).most_common(1)[0][0]
                
                sample = {
                    'question_id': question_id,
                    'question': item['question'],
                    'image_name': item['image_name'],
                    'image_id': item.get('image_id', question_id.split('00')[0] if '00' in question_id else question_id),
                    'answer': most_common_answer,
                    'all_answers': answers,
                    'explanation': item.get('explanation', [])
                }
                dataset.append(sample)
            
            logging.info(f"Loaded VQA-X format: {len(dataset)} samples")
        else:
            # Standard format: [{question, image_path, answer}]
            if isinstance(raw_data, dict) and 'data' in raw_data:
                dataset = raw_data['data']
            else:
                dataset = raw_data
            
            logging.info(f"Loaded standard format: {len(dataset)} samples")
        
        # Processing configuration
        processing_config = config.get('processing_config', {})
        num_samples = processing_config.get('num_samples', len(dataset))
        if override_samples:
            num_samples = override_samples
        if num_samples > 0 and num_samples < len(dataset):
            dataset = dataset[:num_samples]
            logging.info(f"Limited to {num_samples} samples for processing")
        
        # Results storage
        results = []
        
        # Process each sample
        logging.info(f"Processing {len(dataset)} samples...")
        for i, sample in enumerate(tqdm(dataset, desc="MVKB-X Processing")):
            try:
                # Extract sample data based on format
                if dataset_format == 'vqax' or dataset_format == 'vivqax':
                    # VQA-X or ViVQA-X format
                    image_name = sample['image_name']
                    image_dir = dataset_config.get('image_dir', '/mnt/VLAI_data/COCO_Images/val2014')
                    image_path = os.path.join(image_dir, image_name)
                    question = sample['question']
                    ground_truth = sample['answer']
                    question_id = sample['question_id']
                else:
                    # Standard format
                    if 'image_path' in sample:
                        image_path = sample['image_path']
                    elif 'image' in sample:
                        image_path = sample['image']
                    else:
                        logging.error(f"Sample {i}: No image path found")
                        continue
                    
                    question = sample.get('question', sample.get('question_text', ''))
                    ground_truth = sample.get('answer', sample.get('ground_truth', None))
                    question_id = sample.get('question_id', f"sample_{i}")
                
                if not question:
                    logging.error(f"Sample {i}: No question found")
                    continue
                
                # Ensure absolute image path
                if not os.path.isabs(image_path):
                    dataset_dir = os.path.dirname(input_file)
                    image_path = os.path.join(dataset_dir, image_path)
                
                if not os.path.exists(image_path):
                    logging.warning(f"Sample {i}: Image not found: {image_path}")
                    continue
                
                logging.info(f"Processing sample {i+1}/{len(dataset)}: {question[:50]}...")
                
                # Step 1: Verifier - Initial analysis
                initial_response = verifier.generate_initial_response(question, image_path)
                answer_candidates = initial_response['answer_candidates']
                caption = initial_response['caption']
                
                # Step 2: Strategist - MVKB construction
                mvkb = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                # Step 3: Synthesizer - Weighted voting (Algorithm 2)
                voting_result = synthesizer.conduct_weighted_voting(
                    question, image_path, answer_candidates, mvkb
                )
                
                final_answer = voting_result['final_answer']
                confidence_breakdown = voting_result['confidence_breakdown']
                
                # Step 4: Explanation - Generate natural language explanation
                explanation_text = explanation.generate_explanation(
                    question=question,
                    final_answer=final_answer,
                    caption=caption,
                    mvkb_entries=mvkb,
                    confidence_breakdown=confidence_breakdown
                )
                
                # Store result
                result = {
                    'question_id': question_id,
                    'sample_id': i,
                    'question': question,
                    'image_path': image_path,
                    'initial_candidates': answer_candidates,
                    'caption': caption,
                    'mvkb_entries': len(mvkb),
                    'final_answer': final_answer,
                    'explanation': explanation_text,
                    'confidence_breakdown': confidence_breakdown,
                    'ground_truth': ground_truth
                }
                
                # Add VQA-X specific fields if available
                if dataset_format == 'vqax':
                    result.update({
                        'ground_truth_explanations': sample.get('explanation', [])
                    })
                
                results.append(result)
                
                logging.info(f"✅ Sample {i+1} completed: {final_answer}")
                
            except Exception as e:
                logging.error(f"❌ Sample {i} failed: {e}")
                continue
        
        # Save results
        output_config = config.get('output_config', {})
        output_dir = output_config.get('output_dir', 'output')
        output_filename = output_config.get('output_file', 'mvkb_x_results.json')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine directory and filename
        output_file = os.path.join(output_dir, output_filename)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"💾 Results saved to: {output_file}")
        
        # Quick accuracy calculation (always run)
        accuracy_stats = calculate_quick_accuracy(results)
        
        # Run evaluation if enabled
        if enable_evaluation and results:
            eval_module = EvalModule()
            # evaluation_results = eval_module.evaluate_results(results)
            
                # 🔧 Extract ground_truth_explanations from results
            ground_truth_explanations = {}
            for result in results:
                question_id = result.get('question_id')
                gt_explanations = result.get('ground_truth_explanations', [])
                if question_id and gt_explanations:
                    ground_truth_explanations[question_id] = gt_explanations
            
            # 🔧 Pass ground_truth_explanations to evaluation
            if ground_truth_explanations:
                logging.info(f"📚 Found ground truth explanations for {len(ground_truth_explanations)} questions")
                evaluation_results = eval_module.evaluate_results(results, ground_truth_explanations)
            else:
                logging.warning("⚠️ No ground truth explanations found in results")
                evaluation_results = eval_module.evaluate_results(results)
            
            eval_output_file = output_file.replace('.json', '_evaluation.json')
            with open(eval_output_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logging.info(f"📊 Evaluation results saved to: {eval_output_file}")
            
            # Print summary with full evaluation
            print(f"\n🎯 MVKB-X Pipeline Summary:")
            print(f"Processed samples: {len(results)}")
            
            # Use quick accuracy stats instead of evaluation_results for accuracy
            print(f"VQA Accuracy: {accuracy_stats['accuracy']:.3f}")
            
            # Check for explanation metrics
            exp_metrics = evaluation_results.get('explanation_metrics', {})
            if exp_metrics.get('evaluated_explanations', 0) > 0:
                if 'bleu_4' in exp_metrics:
                    print(f"BLEU-4 Score: {exp_metrics['bleu_4']:.3f}")
                if 'semantic_similarity' in exp_metrics:
                    print(f"Semantic Similarity: {exp_metrics['semantic_similarity']:.3f}")
            
            # Confidence metrics
            conf_metrics = evaluation_results.get('confidence_metrics', {})
            if conf_metrics:
                print(f"Average Confidence: {conf_metrics.get('avg_confidence', 0.0):.3f}")
        else:
            # Print summary with quick accuracy
            print(f"\n🎯 MVKB-X Pipeline Summary:")
            print(f"Processed samples: {len(results)}")
            print(f"VQA Accuracy: {accuracy_stats['accuracy']:.1%}")
            print(f"Correct answers: {accuracy_stats['correct']}/{accuracy_stats['total']}")
            print(f"MVKB voting effectiveness: {accuracy_stats['voting_effectiveness']:.1%}")
        
        logging.info("🎉 MVKB-X Pipeline completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ MVKB-X Pipeline failed: {e}")
        raise

def calculate_quick_accuracy(results):
    """Calculate quick accuracy statistics from results"""
    total = 0
    correct = 0
    voting_worked = 0
    
    for result in results:
        ground_truth = result.get('ground_truth')
        if ground_truth is not None and ground_truth != "":  # Check for None and empty string
            total += 1
            final_answer = str(result.get('final_answer', '')).lower().strip()
            ground_truth_str = str(ground_truth).lower().strip()
            
            # Debug logging
            print(f"DEBUG: Comparing final_answer='{final_answer}' vs ground_truth='{ground_truth_str}'")
            
            # Simple string matching for accuracy
            if final_answer == ground_truth_str:
                correct += 1
                print(f"DEBUG: ✅ Match found!")
            else:
                print(f"DEBUG: ❌ No match")
            
            # Check if voting mechanism worked
            confidence_breakdown = result.get('confidence_breakdown', {})
            total_votes = confidence_breakdown.get('total_votes', 0)
            if total_votes > 0:
                voting_worked += 1
    
    accuracy = correct / total if total > 0 else 0
    voting_effectiveness = voting_worked / total if total > 0 else 0
    
    print(f"DEBUG: Final stats - total: {total}, correct: {correct}, accuracy: {accuracy}")
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'voting_effectiveness': voting_effectiveness
    }

# Legacy function name for backward compatibility
def run_fdr_pipeline(use_vllm: bool = True, enable_evaluation: bool = False, config_path: str = None):
    """
    Legacy function name for backward compatibility.
    """
    logging.warning("run_fdr_pipeline is deprecated. Use run_mvkb_x_pipeline instead.")
    return run_mvkb_x_pipeline(use_vllm, enable_evaluation, config_path=config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MVKB-X Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI instead of vLLM")
    parser.add_argument("--enable_evaluation", action="store_true", help="Run evaluation after processing")
    
    args = parser.parse_args()
    
    run_mvkb_x_pipeline(
        config_path=args.config,
        use_vllm=not args.use_openai,
        enable_evaluation=args.enable_evaluation
    )