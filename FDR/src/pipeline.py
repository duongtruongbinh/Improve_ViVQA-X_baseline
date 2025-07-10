# Top-Down/core/pipeline.py
import os
import json
import yaml
import logging
from collections import Counter
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path to allow for utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent
from src.eval import EvalModule, GEvaluator

# Import BackendManager with error handling
try:
    from utils.backend_manager import BackendManager
except ImportError as e:
    logging.error(f"Failed to import BackendManager: {e}")
    BackendManager = None

# --- Helper Functions ---

def _normalize_answer(answer: str) -> str:
    """Normalize answer format for better evaluation accuracy."""
    if not answer:
        return answer

    # Remove trailing punctuation and extra whitespace
    normalized = answer.strip().rstrip('.!?').strip()

    # Handle common format variations
    if normalized.lower() in ['yes', 'no']:
        return normalized.lower()

    # For other answers, keep original case but remove trailing punctuation
    return normalized

def load_config(config_path=None):
    """Loads the YAML configuration file from unified config.yaml or specified path."""
    if config_path is None:
        # Use unified config.yaml by default
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, 'config.yaml')
    
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Main FDR Pipeline with Synthesizer Logic Engine ---

def run_fdr_pipeline(use_vllm: bool = True,
                     enable_evaluation: bool = False,
                     override_samples: int = None,
                     config_path: str = None,
                     active_dataset_override: str = None,
                     model_preference: str = "auto"):
    """
    Run the complete FDR pipeline with NEW Synthesizer Logic Engine.
    Uses unified config.yaml by default.
    
    Args:
        use_vllm: Whether to use vLLM backend (True) or OpenAI (False)
        enable_evaluation: Whether to run comprehensive evaluation
        override_samples: Override config num_samples (for testing)
        config_path: Optional path to custom configuration file (uses config.yaml by default)
        active_dataset_override: Override the active dataset from config
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
        
        logging.info("🚀 Starting FDR Pipeline with Synthesizer Logic Engine")
        
        # Setup API client for G-Eval if using OpenAI backend
        openai_client = None
        if not use_vllm:
            try:
                # Use BackendManager to ensure consistent client initialization
                if BackendManager is None:
                    logging.warning("BackendManager not available. G-Eval will be skipped.")
                else:
                    backend_manager = BackendManager(backend_type="openai")
                    if backend_manager.is_available():
                        openai_client = backend_manager.client
                        logging.info("✅ OpenAI client for G-Eval initialized via BackendManager.")
                    else:
                        logging.warning("Could not initialize OpenAI client via BackendManager. G-Eval will be skipped.")
            except Exception as e:
                logging.warning(f"Could not initialize OpenAI client for G-Eval. It will be skipped. Reason: {e}")
        
        # Initialize agents
        agents_config = config.get('agents_config', {})
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier_config = agents_config.get('verifier', {})
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=use_vllm,
            enable_dam=verifier_config.get('enable_dam', True),
            groundingdino_docker=verifier_config.get('groundingdino_docker', False),
            model_preference="vlm"  # Always use VLM for vision tasks
        )

        # Strategist Agent (LLM for MVKB construction + explanation generation)
        strategist_config = agents_config.get('strategist', {})
        strategist = StrategistAgent(
            model_name=config.get('backend_config', {}).get('model_name'),
            verifier=verifier,
            use_vllm=use_vllm,
            model_preference="llm"  # Optimized: Use LLM for text-only reasoning tasks
        )
        
        # Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Load dataset from unified config
        active_dataset = active_dataset_override or config.get('active_dataset', 'vqax')
        datasets_config = config.get('datasets', {})
        
        if active_dataset not in datasets_config:
            raise ValueError(f"Active dataset '{active_dataset}' not found in datasets config. Available: {list(datasets_config.keys())}")
        
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
        for i, sample in enumerate(tqdm(dataset, desc="FDR Processing")):
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
                
                # Step 1: Verifier - Generate initial context (caption)
                initial_response = verifier.generate_initial_response(question, image_path)
                answer_candidates = initial_response['answer_candidates']
                caption = initial_response['caption']
                
                # Step 2: Strategist - Decompose question and create reasoning plan (issues + hypothesis)
                # The mvkb variable now holds a dict: {"evidence_set": [], "hypothesis_set": []}
                mvkb_payload = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                if not mvkb_payload:
                    logging.error(f"Sample {i}: Strategist failed to build MVKB. Skipping.")
                    continue

                evidence_set = mvkb_payload.get("evidence_set", [])
                hypothesis_set = mvkb_payload.get("hypothesis_set", [])
                
                # Step 3: Synthesizer - Execute the reasoning plan
                synthesis_result = synthesizer.synthesize(
                    evidence_set=evidence_set, 
                    hypothesis_set=hypothesis_set,
                    answer_candidates=answer_candidates
                )
                
                final_answer = synthesis_result.get('answer')
                synthesis_status = synthesis_result.get('status', 'UNKNOWN')
                causal_trace = synthesis_result.get('causal_trace', [])

                # Normalize answer format for better evaluation accuracy
                if final_answer:
                    final_answer = _normalize_answer(final_answer)
                
                # Step 4: Enhanced Explanation with Causal Trace
                explanation_text = strategist.generate_explanation(
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
                    'mvkb_entries_count': len(mvkb_payload)
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
        output_filename = output_config.get('output_file', 'fdr_results.json')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine directory and filename
        output_file = os.path.join(output_dir, output_filename)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"💾 Results saved to: {output_file}")
        
        # Quick accuracy calculation (always run)
        accuracy_stats = calculate_quick_accuracy_with_synthesis(results)
        
        # Run evaluation if enabled - DEBUG ADDED
        logging.info(f"🔍 Evaluation check: enable_evaluation={enable_evaluation}, results_count={len(results) if results else 0}")
        
        if enable_evaluation and results:
            logging.info("📊 Starting comprehensive evaluation...")
            
            try:
                # Always try to get an OpenAI client for G-Eval, regardless of the main backend
                g_eval_client = None
                try:
                    # Use BackendManager to robustly initialize an OpenAI client
                    # This leverages the key loading logic from backend_manager
                    if BackendManager is None:
                        logging.warning("⚠️ BackendManager not available. G-Eval will be skipped.")
                    else:
                        openai_backend_for_eval = BackendManager(backend_type="openai")
                        if openai_backend_for_eval.is_available():
                            g_eval_client = openai_backend_for_eval.client
                            logging.info("✅ OpenAI client for G-Eval is available.")
                        else:
                            logging.warning("⚠️ OpenAI client for G-Eval is not available. G-Eval will be skipped.")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to initialize OpenAI client for G-Eval: {e}. G-Eval will be skipped.")

                eval_module = EvalModule(device="cpu")
                logging.info("✅ EvalModule initialized successfully")
                
                g_evaluator = GEvaluator(client=g_eval_client) if g_eval_client else None
                logging.info(f"G-Evaluator: {'✅ Initialized' if g_evaluator else '❌ Skipped (no OpenAI client)'}")

                # Prepare ground truth explanations from the dataset
                ground_truth_explanations = {}
                if dataset_format == 'vqax':
                    for sample in dataset:
                        if 'explanation' in sample and sample.get('explanation'):
                            # Ensure explanations are strings, not lists of strings
                            gt_exps = sample['explanation']
                            if isinstance(gt_exps, list) and all(isinstance(e, str) for e in gt_exps):
                                ground_truth_explanations[sample['question_id']] = gt_exps
                            elif isinstance(gt_exps, str):
                                ground_truth_explanations[sample['question_id']] = [gt_exps]

                logging.info(f"📝 Ground truth explanations prepared: {len(ground_truth_explanations)} samples")

                # Run the new comprehensive evaluation
                logging.info("🔬 Running comprehensive evaluation...")
                evaluation_results = eval_module.evaluate_comprehensive(
                    results=results,
                    ground_truth_explanations=ground_truth_explanations if ground_truth_explanations else None,
                    include_g_eval=g_evaluator is not None,
                    g_evaluator=g_evaluator
                )

                logging.info("✅ Comprehensive evaluation completed!")

                # Save the comprehensive report
                eval_output_file = output_file.replace('.json', '_evaluation.json')
                eval_module.save_evaluation_report(evaluation_results, eval_output_file)
                logging.info(f"📊 Comprehensive evaluation report saved to: {eval_output_file}")

                # The detailed summary is now logged inside _log_evaluation_summary, no need to log here.
                
            except Exception as eval_error:
                logging.error(f"❌ Evaluation failed: {eval_error}")
                import traceback
                logging.error(f"Evaluation traceback: {traceback.format_exc()}")

        else:
            # If evaluation is disabled, print a simpler summary based on quick accuracy.
            print(f"\n🎯 FDR Pipeline Summary (Evaluation Disabled):")
            print(f"  - Processed samples: {len(results)}")
            print(f"  - Quick VQA Accuracy: {accuracy_stats['accuracy']:.1%}")
            print(f"  - Status distribution: {accuracy_stats['status_distribution']}")
        
        # This is the new centralized completion message.
        logging.info("🎉 FDR Pipeline with Synthesizer Logic Engine completed.")
        
        # Display final evaluation table if evaluation was enabled
        if enable_evaluation and 'evaluation_results' in locals():
            _print_evaluation_table(evaluation_results)

        return results
        
    except Exception as e:
        logging.error(f"❌ FDR Pipeline failed: {e}")
        raise

def _print_evaluation_table(results: dict):
    """Prints a formatted summary table of the evaluation results with all major metrics."""
    summary = results.get("summary", {})
    vqa = results.get("vqa_metrics", {})
    exp = results.get("explanation_metrics", {})
    consistency = results.get("consistency_metrics", {})
    geval = results.get("g_eval_metrics", {})

    title = "📊 FDR Evaluation Summary"
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"{title:^100}")
    print(f"{bar}\n")

    # Overall Section
    print(f"  OVERALL ASSESSMENT")
    print(f"  {'-'*35}")
    print(f"  {'Total Questions:':<30} {summary.get('total_questions', 'N/A')}")
    print(f"  {'VQA Accuracy:':<30} {vqa.get('accuracy', 0.0):.2%}")
    if geval:
        avg_geval = (geval.get('avg_relevance', 0) + geval.get('avg_coherence', 0) + geval.get('avg_faithfulness', 0)) / 3
        print(f"  {'G-Eval Score (Avg):':<30} {avg_geval:.2f} / 10.0")
    print("-" * 100)

    # Metrics Table Header
    print(f"{'Metric':<12} | {'BLEU-1':^7} | {'BLEU-2':^7} | {'BLEU-3':^7} | {'BLEU-4':^7} | {'METEOR':^7} | {'ROUGE-L P':^9} | {'ROUGE-L R':^9} | {'ROUGE-L F1':^9} | {'CIDEr':^7} | {'SPICE':^7} | {'BERT-P':^7} | {'BERT-R':^7} | {'BERT-F1':^7} | {'SemScore':^7}")
    print("-" * 100)

    # Metrics Table Row
    print(f"{'Explanation':<12} | "
          f"{exp.get('bleu_1', 0.0):7.3f} | "
          f"{exp.get('bleu_2', 0.0):7.3f} | "
          f"{exp.get('bleu_3', 0.0):7.3f} | "
          f"{exp.get('bleu_4', 0.0):7.3f} | "
          f"{exp.get('meteor', 0.0):7.3f} | "
          f"{exp.get('rouge_l_precision', 0.0):9.3f} | "
          f"{exp.get('rouge_l_recall', 0.0):9.3f} | "
          f"{exp.get('rouge_l_f1', 0.0):9.3f} | "
          f"{exp.get('cider', 0.0):7.3f} | "
          f"{exp.get('spice', 0.0):7.3f} | "
          f"{exp.get('bert_precision', 0.0):7.3f} | "
          f"{exp.get('bert_recall', 0.0):7.3f} | "
          f"{exp.get('bert_f1', 0.0):7.3f} | "
          f"{exp.get('semantic_similarity', 0.0):7.3f}")
    print("-" * 100)

    # G-Eval Metrics (if available)
    if geval:
        print(f"\n  G-Eval Breakdown (10-point scale):")
        print(f"    {'Relevance:':<22} {geval.get('avg_relevance', 0):.2f} / 10.0")
        print(f"    {'Coherence:':<22} {geval.get('avg_coherence', 0):.2f} / 10.0")
        print(f"    {'Faithfulness:':<22} {geval.get('avg_faithfulness', 0):.2f} / 10.0")

    print(f"\n{bar}")

def calculate_quick_accuracy_with_synthesis(results):
    """Calculate accuracy statistics including synthesis status"""
    total = 0
    correct = 0
    status_counts = {}
    
    for result in results:
        if result.get('ground_truth'):
            total += 1
            
            # Robustly handle None for final_answer
            final_answer_value = result.get('final_answer')
            final_answer = (final_answer_value or "").lower().strip()
            
            ground_truth = (result.get('ground_truth') or "").lower().strip()
            
            # Simple string matching for accuracy, handles empty final_answer
            if final_answer and (final_answer == ground_truth or ground_truth in final_answer or final_answer in ground_truth):
                correct += 1
            
            # Track synthesis status
            status = result.get('synthesis_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'status_distribution': status_counts
    }