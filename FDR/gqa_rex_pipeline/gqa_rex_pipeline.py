"""
GQA-REX Pipeline Integration for FDR Framework
Extended pipeline to handle GQA + GQA-REX combined datasets
"""

import os
import sys
import json
import yaml
import logging
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

# Import original FDR components
from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent
from src.pipeline import _normalize_answer
from src.eval import EvalModule, GEvaluator

# Import GQA-REX specific components
from gqa_rex_pipeline.gqa_rex_loader import GQAREXLoader, create_gqa_rex_loader
from gqa_rex_pipeline.gqa_rex_config import validate_gqa_rex_paths

# Import BackendManager with error handling
try:
    from utils.backend_manager import BackendManager
except ImportError as e:
    logging.error(f"Failed to import BackendManager: {e}")
    BackendManager = None


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


def load_gqa_rex_config(config_path=None):
    """Load GQA-REX configuration file."""
    if config_path is None:
        # Use GQA-REX config by default
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config_gqa_rex.yaml')
    
    logging.info(f"Loading GQA-REX configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_gqa_rex_pipeline(use_vllm: bool = True,
                        enable_evaluation: bool = False,
                        override_samples: int = None,
                        config_path: str = None,
                        active_dataset_override: str = None,
                        model_preference: str = "auto"):
    """
    Run FDR pipeline with GQA-REX dataset support.
    
    Args:
        use_vllm: Whether to use vLLM backend
        enable_evaluation: Whether to run comprehensive evaluation
        override_samples: Override config num_samples
        config_path: Path to GQA-REX config file
        active_dataset_override: Override active dataset
    """
    try:
        # Load GQA-REX configuration
        config = load_gqa_rex_config(config_path)
        
        # Initialize logging
        logging_config = config.get('logging_config', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        
        # Setup file logging if enabled
        handlers = [logging.StreamHandler()]
        if logging_config.get('enable_file_logging', False):
            log_file = logging_config.get('log_file', 'logs/gqa_rex_pipeline.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers
        )
        
        # Reduce HTTP logs if configured
        if logging_config.get('reduce_http_logs', True):
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        logging.info("🚀 Starting GQA-REX FDR Pipeline")
        logging.info(f"📋 Experiment: {config.get('experiment_name', 'gqa_rex_pipeline')}")
        
        # Initialize backend manager
        backend_config = config.get('backend_config', {})
        if BackendManager is None:
            logging.warning("⚠️ BackendManager not available, using basic backend")
            backend_manager = None
        else:
            backend_type = "vllm" if use_vllm else "openai"
            backend_manager = BackendManager(
                backend_type=backend_type,
                model_preference=backend_config.get('model_preference', 'auto')
            )
        
        # Initialize agents with GQA-REX optimized settings (following original pipeline pattern)
        agents_config = config.get('agents_config', {})
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier_config = agents_config.get('verifier', {})
        verifier = VerifierAgent(
            model_name=backend_config.get('model_name'),
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
            model_name=backend_config.get('model_name'),
            verifier=verifier,
            use_vllm=use_vllm,
            model_preference="llm"  # Optimized: Use LLM for text-only reasoning tasks
        )
        
        # Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Load GQA-REX dataset
        active_dataset = active_dataset_override or config.get('active_dataset', 'gqa_rex_val')
        datasets_config = config.get('datasets', {})
        
        if active_dataset not in datasets_config:
            raise ValueError(f"Active dataset '{active_dataset}' not found in datasets config. Available: {list(datasets_config.keys())}")
        
        dataset_config = datasets_config[active_dataset]
        
        # Validate all required paths
        if not validate_gqa_rex_paths(dataset_config):
            raise ValueError("GQA-REX dataset path validation failed")
        
        logging.info(f"📚 Loading dataset: {dataset_config.get('name', active_dataset)}")
        
        # Create GQA-REX loader
        loader = GQAREXLoader(
            gqa_data_path=dataset_config['data_path'],
            rex_data_path=dataset_config['rex_path'],
            image_dir=dataset_config['image_dir'],
            scene_graph_path=dataset_config.get('scene_graph_path')
        )
        
        # Get dataset statistics
        stats = loader.get_dataset_stats()
        logging.info(f"📊 Dataset Statistics:")
        logging.info(f"   • Total linked samples: {stats['total_samples']}")
        logging.info(f"   • GQA questions: {stats['gqa_questions']}")
        logging.info(f"   • REX explanations: {stats['rex_explanations']}")
        if stats['scene_graphs']:
            logging.info(f"   • Scene graphs: {stats['scene_graphs']}")
        
        # Show top question types
        if stats.get('top_semantic_types'):
            logging.info(f"   • Top semantic types: {stats['top_semantic_types'][:3]}")
        
        # Processing configuration
        processing_config = config.get('processing_config', {})
        num_samples = processing_config.get('num_samples', 0)
        
        if override_samples is not None:
            num_samples = override_samples
            
        # Get samples from loader
        if num_samples > 0:
            dataset = loader.get_samples(limit=num_samples)
            logging.info(f"🎯 Processing {num_samples} samples (limited)")
        else:
            dataset = loader.get_samples()
            logging.info(f"🎯 Processing all {len(dataset)} samples")
        
        # Results storage
        results = []
        results_config = config.get('output_config', {})
        results_dir = results_config.get('results_dir', 'output/gqa_rex_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each sample
        logging.info(f"🔄 Starting GQA-REX pipeline processing...")
        
        for i, sample in enumerate(tqdm(dataset, desc="GQA-REX FDR Processing")):
            try:
                # Extract sample data
                question = sample['question']
                ground_truth = sample['answer']
                image_path = sample['image_path']
                explanation = sample['explanation']
                question_id = sample['question_id']
                
                logging.debug(f"Processing sample {i+1}: {question_id}")
                logging.debug(f"Question: {question}")
                logging.debug(f"Ground truth: {ground_truth}")
                logging.debug(f"REX explanation: {explanation}")
                
                # Validate image exists
                if not os.path.exists(image_path):
                    logging.warning(f"⚠️ Image not found: {image_path}")
                    continue
                
                # Step 1: Verifier - Generate initial context (caption and candidates)
                initial_response = verifier.generate_initial_response(question, image_path)
                answer_candidates = initial_response['answer_candidates']
                caption = initial_response['caption']
                
                # Step 2: Strategist - Decompose question and create reasoning plan (MVKB)
                mvkb_payload = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                if not mvkb_payload:
                    logging.error(f"Sample {i}: Strategist failed to build MVKB. Skipping.")
                    continue
                
                # Step 3: Synthesizer - Execute the reasoning plan (following original pattern)
                evidence_set = mvkb_payload.get("evidence_set", [])
                hypothesis_set = mvkb_payload.get("hypothesis_set", [])
                
                synthesis_result = synthesizer.synthesize(
                    evidence_set=evidence_set, 
                    hypothesis_set=hypothesis_set,
                    answer_candidates=answer_candidates
                )
                
                final_answer = synthesis_result.get('answer')
                synthesis_status = synthesis_result.get('status', 'UNKNOWN')
                
                # Step 4: Enhanced Explanation with Causal Trace
                explanation_generated = strategist.generate_explanation(
                    question=question,
                    synthesis_result=synthesis_result,
                    caption=caption,
                    evidence_set=evidence_set
                )
                
                # Prepare result
                predicted_answer = _normalize_answer(final_answer)
                normalized_ground_truth = _normalize_answer(ground_truth)
                
                result = {
                    'question_id': question_id,
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'ground_truth': normalized_ground_truth,
                    'final_answer': final_answer,
                    'synthesis_status': synthesis_status,
                    'is_correct': predicted_answer.lower() == normalized_ground_truth.lower(),
                    'explanation': explanation_generated,
                    'rex_explanation': explanation,
                    'initial_candidates': answer_candidates,
                    'caption': caption,
                    'mvkb_entries_count': len(mvkb_payload),
                    'image_path': image_path,
                    'question_type': sample.get('semantic_type', 'unknown'),
                    'structural_type': sample.get('structural_type', 'unknown'),
                    'processing_time': synthesis_result.get('processing_time', 0),
                    'confidence': synthesis_result.get('confidence', 0.0)
                }
                
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0 or i == 0:
                    accuracy = sum(r['is_correct'] for r in results) / len(results) * 100
                    logging.info(f"🎯 Progress: {i+1}/{len(dataset)} samples, Accuracy: {accuracy:.1f}%")
                
                # Save individual result if configured
                if results_config.get('save_individual_results', False):
                    individual_file = os.path.join(results_dir, f'result_{question_id}.json')
                    with open(individual_file, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"❌ Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final evaluation
        if results:
            total_samples = len(results)
            correct_answers = sum(1 for r in results if r['is_correct'])
            accuracy = correct_answers / total_samples * 100
            
            # Calculate per-type accuracy
            type_stats = {}
            for result in results:
                q_type = result['question_type']
                if q_type not in type_stats:
                    type_stats[q_type] = {'total': 0, 'correct': 0}
                type_stats[q_type]['total'] += 1
                if result['is_correct']:
                    type_stats[q_type]['correct'] += 1
            
            # Sort by frequency
            sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['total'], reverse=True)
            
            logging.info("=" * 60)
            logging.info("🎉 GQA-REX FDR Pipeline Results")
            logging.info("=" * 60)
            logging.info(f"📊 Overall Accuracy: {accuracy:.2f}% ({correct_answers}/{total_samples})")
            logging.info("")
            logging.info("📋 Per-Type Accuracy:")
            
            for q_type, stats in sorted_types[:10]:  # Top 10 types
                type_accuracy = stats['correct'] / stats['total'] * 100
                logging.info(f"   • {q_type}: {type_accuracy:.1f}% ({stats['correct']}/{stats['total']})")
            
            # Save summary results
            summary = {
                'experiment_name': config.get('experiment_name'),
                'dataset': active_dataset,
                'total_samples': total_samples,
                'correct_answers': correct_answers,
                'accuracy': accuracy,
                'type_stats': dict(type_stats),
                'config_used': config,
                'results': results if results_config.get('save_detailed_results', True) else []
            }
            
            summary_file = os.path.join(results_dir, 'summary_results.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logging.info(f"💾 Results saved to: {results_dir}")
            logging.info(f"📄 Summary: {summary_file}")
            
            # Run comprehensive evaluation if enabled
            if enable_evaluation and results:
                logging.info("🔍 Running comprehensive evaluation...")
                try:
                    eval_module = EvalModule()
                    eval_results = eval_module.evaluate_batch(results)
                    logging.info("📊 Evaluation completed")
                    
                    # Save evaluation results
                    eval_file = os.path.join(results_dir, 'evaluation_results.json')
                    with open(eval_file, 'w') as f:
                        json.dump(eval_results, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    logging.error(f"❌ Evaluation failed: {e}")
        
        logging.info("✅ GQA-REX FDR Pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        logging.error(f"❌ GQA-REX Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Simple test run
    import argparse
    
    parser = argparse.ArgumentParser(description="GQA-REX FDR Pipeline")
    parser.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--dataset", choices=["gqa_rex_train", "gqa_rex_val"], default=None)
    
    args = parser.parse_args()
    
    run_gqa_rex_pipeline(
        use_vllm=(args.backend == "vllm"),
        enable_evaluation=args.evaluate,
        override_samples=args.samples,
        active_dataset_override=args.dataset
    )
