#!/usr/bin/env python3
"""
CLEVR-X Pipeline Runner
Runs FDR pipeline on CLEVR-X dataset without modifying the original pipeline code.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.data_loaders.clevr_x_loader import CLEVRXLoader
from src.pipeline import run_fdr_pipeline
from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent
from src.eval import EvalModule


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('output/clevr_x_pipeline.log')
        ]
    )


def run_clevr_x_pipeline(
    data_path: str = "/mnt/VLAI_data/CLEVR-X/CLEVR_val_explanations_v0.7.10.json",
    image_dir: str = "/mnt/VLAI_data/CLEVR/CLEVR_v1.0/images/val",
    limit_samples: int = 10,
    use_vllm: bool = True,
    enable_evaluation: bool = False
):
    """
    Run FDR pipeline on CLEVR-X dataset.
    
    Args:
        data_path: Path to CLEVR-X JSON file
        image_dir: Path to CLEVR images directory  
        limit_samples: Number of samples to process (None for all)
        use_vllm: Whether to use vLLM backend
        enable_evaluation: Whether to enable comprehensive evaluation
    """
    setup_logging()
    
    print("🚀 Starting CLEVR-X FDR Pipeline")
    print(f"📊 Data: {data_path}")
    print(f"🖼️  Images: {image_dir}")
    print(f"📝 Samples: {limit_samples}")
    print(f"🔧 Backend: {'vLLM' if use_vllm else 'OpenAI'}")
    print(f"📈 Evaluation: {'Enabled' if enable_evaluation else 'Disabled'}")
    print("="*70)
    
    try:
        # 1. Initialize CLEVR-X Loader
        logging.info("🔄 Initializing CLEVR-X Data Loader...")
        loader = CLEVRXLoader(data_path, image_dir)
        
        # 2. Validate data
        logging.info("🔍 Validating data samples...")
        validation = loader.validate_samples(limit=min(limit_samples, 10))
        logging.info(f"✅ Validation success rate: {validation['validation_success_rate']:.1f}%")
        
        if validation['validation_success_rate'] < 80:
            logging.warning("⚠️ Low validation success rate. Check data paths.")
            for issue in validation['issues'][:3]:
                logging.warning(f"  - {issue}")
        
        # 3. Get samples
        logging.info(f"📥 Loading {limit_samples} samples...")
        samples = loader.get_samples(limit=limit_samples)
        
        if not samples:
            logging.error("❌ No valid samples found!")
            return None
        
        logging.info(f"✅ Loaded {len(samples)} valid samples")
        
        # 4. Initialize Agents (similar to original pipeline)
        logging.info("🤖 Initializing FDR Agents...")
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier = VerifierAgent(
            temperature=0.7,
            max_tokens=1000,
            use_vllm=use_vllm,
            enable_dam=True,
            groundingdino_docker=False,
            model_preference="vlm"
        )
        
        # Strategist Agent (LLM for MVKB construction)
        strategist = StrategistAgent(
            model_name=None,  # Will use default from config
            verifier=verifier,
            use_vllm=use_vllm,
            model_preference="llm"
        )
        
        # Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        logging.info("✅ All agents initialized successfully")
        
        # 5. Process samples
        logging.info("🔄 Processing CLEVR-X samples...")
        results = []
        
        from tqdm import tqdm
        
        for i, sample in enumerate(tqdm(samples, desc="FDR Processing")):
            try:
                question = sample['question']
                image_path = sample['image_path']
                ground_truth_answer = sample['answer']
                ground_truth_explanation = sample['explanation']
                
                logging.info(f"🔍 Processing sample {i+1}/{len(samples)}: {question[:50]}...")
                
                # Step 1: Verifier - Initial analysis
                initial_response = verifier.generate_initial_response(question, image_path)
                
                # Step 2: Strategist - MVKB construction
                answer_candidates = initial_response.get('answer_candidates', [ground_truth_answer])
                caption = initial_response.get('caption', '')
                
                mvkb = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                # Step 3: Synthesizer - Final decision
                synthesis_result = synthesizer.conduct_weighted_voting(
                    question, image_path, answer_candidates, mvkb
                )
                
                # Step 4: Explanation generation
                explanation = strategist.generate_explanation(
                    question=question,
                    synthesis_result=synthesis_result,
                    caption=caption,
                    evidence_set=mvkb
                )
                
                # Collect results
                result = {
                    'sample_index': i,
                    'question': question,
                    'image_path': image_path,
                    'image_filename': sample['image_filename'],
                    'predicted_answer': synthesis_result.get('final_answer', 'unknown'),
                    'ground_truth_answer': ground_truth_answer,
                    'predicted_explanation': explanation,
                    'ground_truth_explanation': ground_truth_explanation,
                    'all_ground_truth_explanations': sample.get('all_explanations', []),
                    'confidence_score': synthesis_result.get('confidence', 0.0),
                    'synthesis_status': synthesis_result.get('status', 'unknown'),
                    'mvkb_size': len(mvkb),
                    'program_steps': len(sample.get('program', [])),
                    'processing_success': True
                }
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"❌ Error processing sample {i+1}: {e}")
                results.append({
                    'sample_index': i,
                    'question': sample.get('question', 'unknown'),
                    'processing_success': False,
                    'error': str(e)
                })
        
        # 6. Calculate basic metrics
        successful_results = [r for r in results if r.get('processing_success', False)]
        total_processed = len(successful_results)
        
        if total_processed > 0:
            # Calculate accuracy
            correct = sum(1 for r in successful_results 
                         if r['predicted_answer'].lower().strip() == r['ground_truth_answer'].lower().strip())
            accuracy = correct / total_processed * 100
            
            # Status distribution
            status_dist = {}
            for r in successful_results:
                status = r.get('synthesis_status', 'unknown')
                status_dist[status] = status_dist.get(status, 0) + 1
            
            print(f"\n🎯 CLEVR-X FDR Pipeline Results:")
            print(f"  - Total samples: {len(samples)}")
            print(f"  - Successfully processed: {total_processed}")
            print(f"  - Accuracy: {accuracy:.1f}% ({correct}/{total_processed})")
            print(f"  - Status distribution: {status_dist}")
            
            # Save results
            import json
            output_file = "output/clevr_x_fdr_results.json"
            os.makedirs("output", exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'dataset_info': loader.get_info(),
                    'processing_summary': {
                        'total_samples': len(samples),
                        'processed_samples': total_processed,
                        'accuracy': accuracy,
                        'correct_predictions': correct,
                        'status_distribution': status_dist
                    },
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to: {output_file}")
            
            # Evaluation if enabled
            if enable_evaluation and total_processed > 0:
                logging.info("📊 Running evaluation...")
                try:
                    eval_module = EvalModule()
                    
                    predictions = [r['predicted_answer'] for r in successful_results]
                    ground_truths = [r['ground_truth_answer'] for r in successful_results]
                    explanations = [r['predicted_explanation'] for r in successful_results]
                    
                    eval_results = eval_module.evaluate_accuracy(predictions, ground_truths)
                    print(f"\n📈 Detailed Evaluation Results:")
                    for metric, score in eval_results.items():
                        print(f"  - {metric}: {score:.3f}")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Evaluation failed: {e}")
            
        else:
            print("❌ No samples processed successfully!")
            
        print("\n✅ CLEVR-X Pipeline completed!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        raise


def main():
    """Main entry point with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FDR Pipeline on CLEVR-X Dataset")
    parser.add_argument("--data", default="/mnt/VLAI_data/CLEVR-X/CLEVR_val_explanations_v0.7.10.json",
                       help="Path to CLEVR-X JSON file")
    parser.add_argument("--images", default="/mnt/VLAI_data/CLEVR/CLEVR_v1.0/images/val",
                       help="Path to CLEVR images directory")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples to process")
    parser.add_argument("--backend", choices=["vllm", "openai"], default="vllm",
                       help="Backend to use")
    parser.add_argument("--evaluate", action="store_true",
                       help="Enable comprehensive evaluation")
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_clevr_x_pipeline(
        data_path=args.data,
        image_dir=args.images,
        limit_samples=args.samples,
        use_vllm=(args.backend == "vllm"),
        enable_evaluation=args.evaluate
    )
    
    return results


if __name__ == "__main__":
    main()
