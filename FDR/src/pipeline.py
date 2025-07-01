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

def convert_mvkb_to_synthesizer_format(mvkb, initial_response, question):
    """
    ADAPTER LAYER: Convert old MVKB format to evidence_set + hypothesis_set for new Synthesizer Logic Engine
    
    Args:
        mvkb: Old format from StrategistAgent
        initial_response: From VerifierAgent 
        question: Original question
    
    Returns:
        evidence_set, hypothesis_set: New format for SynthesizerEngine
    """
    logging.info("🔄 Converting MVKB to Synthesizer Logic Engine format...")
    
    evidence_set = []
    hypothesis_set = []
    evidence_id_counter = 1
    
    # Step 1: Create evidence from verifier's initial analysis
    answer_candidates = initial_response.get('answer_candidates', [])
    caption = initial_response.get('caption', '')
    
    # Evidence E01: Image caption provides context
    if caption:
        evidence_set.append({
            "evidence_id": f"E{evidence_id_counter:02d}",
            "issue_text": "Image analysis and context understanding",
            "answer": "Available",
            "confidence": 0.85,
            "source": "VerifierAgent",
            "details": caption
        })
        evidence_id_counter += 1
    
    # Step 2: Create evidence from MVKB entries
    mvkb_evidence_map = {}  # Map answer_candidate -> evidence_id
    
    for entry in mvkb:
        answer_candidate = entry.get('answer_candidate', '')
        confidence_score = entry.get('confidence_score', 0.5)
        hypothesis = entry.get('hypothesis', '')
        
        if answer_candidate and hypothesis:
            evidence_id = f"E{evidence_id_counter:02d}"
            
            # Create evidence based on hypothesis confidence
            evidence_answer = "Strong" if confidence_score > 0.7 else "Moderate" if confidence_score > 0.4 else "Weak"
            
            evidence_set.append({
                "evidence_id": evidence_id,
                "issue_text": f"Evidence supporting '{answer_candidate}': {hypothesis}",
                "answer": evidence_answer,
                "confidence": confidence_score,
                "source": "StrategistAgent",
                "answer_candidate": answer_candidate
            })
            
            mvkb_evidence_map[answer_candidate] = evidence_id
            evidence_id_counter += 1
    
    # Step 3: Create hypotheses with logical rules
    for answer_candidate, evidence_id in mvkb_evidence_map.items():
        # Find the corresponding MVKB entry
        mvkb_entry = next((entry for entry in mvkb if entry.get('answer_candidate') == answer_candidate), None)
        if mvkb_entry:
            confidence = mvkb_entry.get('confidence_score', 0.5)
            
            # Create hypothesis based on confidence level
            if confidence > 0.4:  # Only create hypothesis for reasonable confidence
                conditions = []
                
                # Condition 1: Image context is available
                if caption:
                    conditions.append({"evidence_id": "E01", "answer_is": "Available"})
                
                # Condition 2: This answer has sufficient evidence
                evidence_threshold = "Strong" if confidence > 0.7 else "Moderate"
                conditions.append({"evidence_id": evidence_id, "answer_is": evidence_threshold})
                
                hypothesis_set.append({
                    "hypothesis_id": f"H_{answer_candidate.replace(' ', '_')}",
                    "IF": conditions,
                    "THEN": {"final_answer": answer_candidate},
                    "confidence_source": confidence,
                    "reasoning": mvkb_entry.get('hypothesis', f"Evidence supports {answer_candidate}")
                })
    
    # Step 4: Add fallback hypothesis if no strong candidates
    strong_candidates = [h for h in hypothesis_set if any(entry.get('confidence_score', 0) > 0.6 for entry in mvkb if entry.get('answer_candidate') == h['THEN']['final_answer'])]
    
    if not strong_candidates and answer_candidates:
        # Create fallback hypothesis for most likely candidate
        fallback_candidate = answer_candidates[0]
        evidence_id = f"E{evidence_id_counter:02d}"
        
        evidence_set.append({
            "evidence_id": evidence_id,
            "issue_text": "Fallback analysis when no strong evidence is available",
            "answer": "Uncertain",
            "confidence": 0.3,
            "source": "FallbackLogic"
        })
        
        hypothesis_set.append({
            "hypothesis_id": "H_Fallback",
            "IF": [{"evidence_id": evidence_id, "answer_is": "Uncertain"}],
            "THEN": {"final_answer": fallback_candidate},
            "confidence_source": 0.3,
            "reasoning": f"Fallback to most likely candidate: {fallback_candidate}"
        })
    
    logging.info(f"✅ Converted to {len(evidence_set)} evidence items and {len(hypothesis_set)} hypotheses")
    logging.debug(f"Evidence set: {[e['evidence_id'] + ': ' + e['issue_text'] for e in evidence_set]}")
    logging.debug(f"Hypothesis set: {[h['hypothesis_id'] + ' -> ' + h['THEN']['final_answer'] for h in hypothesis_set]}")
    
    return evidence_set, hypothesis_set

# --- Main FDR Pipeline with Synthesizer Logic Engine ---

def run_mvkb_x_pipeline(use_vllm: bool = True, enable_evaluation: bool = False, override_samples: int = None, config_path: str = None):
    """
    Run the complete MVKB-X pipeline with NEW Synthesizer Logic Engine.
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
        
        logging.info("🚀 Starting MVKB-X Pipeline with Synthesizer Logic Engine")
        
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
        
        # NEW Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Enhanced Explanation Agent
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
                
                # Step 2: Strategist - MVKB construction (old format)
                mvkb = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                # Step 3a: ADAPTER - Convert to new format
                evidence_set, hypothesis_set = convert_mvkb_to_synthesizer_format(mvkb, initial_response, question)
                
                # Step 3b: NEW Synthesizer Logic Engine
                synthesis_result = synthesizer.synthesize(evidence_set, hypothesis_set)
                
                final_answer = synthesis_result.get('answer', answer_candidates[0] if answer_candidates else "Unknown")
                synthesis_status = synthesis_result.get('status', 'UNKNOWN')
                causal_trace = synthesis_result.get('causal_trace', [])
                
                # Step 4: Enhanced Explanation with Causal Trace
                explanation_text = explanation.generate_explanation_from_synthesis(
                    question=question,
                    synthesis_result=synthesis_result,
                    caption=caption,
                    evidence_set=evidence_set
                )
                
                # Store result - CLEAN OUTPUT: answer + explanation
                result = {
                    'question_id': question_id,
                    'sample_id': i,
                    'question': question,
                    'image_path': image_path,
                    
                    # MAIN OUTPUT
                    'final_answer': final_answer,
                    'explanation': explanation_text,
                    
                    # METADATA
                    'synthesis_status': synthesis_status,
                    'causal_trace': causal_trace,
                    'evidence_count': len(evidence_set),
                    'hypothesis_count': len(hypothesis_set),
                    'ground_truth': ground_truth,
                    
                    # DEBUG INFO (optional)
                    'initial_candidates': answer_candidates,
                    'caption': caption,
                    'mvkb_entries_count': len(mvkb)
                }
                
                # Add VQA-X specific fields if available
                if dataset_format == 'vqax':
                    result.update({
                        'ground_truth_explanations': sample.get('explanation', [])
                    })
                
                results.append(result)
                
                logging.info(f"✅ Sample {i+1} completed: '{final_answer}' ({synthesis_status})")
                
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
        accuracy_stats = calculate_quick_accuracy_with_synthesis(results)
        
        # Run evaluation if enabled
        if enable_evaluation and results:
            eval_module = EvalModule()
            evaluation_results = eval_module.evaluate_pipeline_results(results)
            
            eval_output_file = output_file.replace('.json', '_evaluation.json')
            with open(eval_output_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logging.info(f"📊 Evaluation results saved to: {eval_output_file}")
            
            # Print summary with full evaluation
            print(f"\n🎯 MVKB-X Pipeline with Synthesizer Logic Engine Summary:")
            print(f"Processed samples: {len(results)}")
            print(f"VQA Accuracy: {evaluation_results['vqa_accuracy']:.3f}")
            if 'explanation_quality' in evaluation_results:
                print(f"Explanation Quality: {evaluation_results['explanation_quality']:.3f}")
        else:
            # Print summary with quick accuracy
            print(f"\n🎯 MVKB-X Pipeline with Synthesizer Logic Engine Summary:")
            print(f"Processed samples: {len(results)}")
            print(f"VQA Accuracy: {accuracy_stats['accuracy']:.1%}")
            print(f"Correct answers: {accuracy_stats['correct']}/{accuracy_stats['total']}")
            print(f"Synthesis effectiveness: {accuracy_stats['synthesis_effectiveness']:.1%}")
            print(f"Status distribution: {accuracy_stats['status_distribution']}")
        
        logging.info("🎉 MVKB-X Pipeline with Synthesizer Logic Engine completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ MVKB-X Pipeline failed: {e}")
        raise

def calculate_quick_accuracy_with_synthesis(results):
    """Calculate accuracy statistics including synthesis status"""
    total = 0
    correct = 0
    synthesis_worked = 0
    status_counts = {}
    
    for result in results:
        if result.get('ground_truth'):
            total += 1
            final_answer = result.get('final_answer', '').lower().strip()
            ground_truth = result.get('ground_truth', '').lower().strip()
            
            # Simple string matching for accuracy
            if final_answer == ground_truth or ground_truth in final_answer or final_answer in ground_truth:
                correct += 1
            
            # Track synthesis status
            status = result.get('synthesis_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Check if synthesis mechanism worked
            if status in ['CONCLUSIVE', 'CONTRADICTORY']:
                synthesis_worked += 1
    
    accuracy = correct / total if total > 0 else 0
    synthesis_effectiveness = synthesis_worked / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'synthesis_effectiveness': synthesis_effectiveness,
        'status_distribution': status_counts
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