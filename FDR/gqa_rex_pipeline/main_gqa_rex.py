#!/usr/bin/env python3
"""
GQA-REX FDR Pipeline Main Entry Point
Easy-to-use script for running FDR pipeline with GQA-REX dataset
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

# Import GQA-REX pipeline
from gqa_rex_pipeline.gqa_rex_pipeline import run_gqa_rex_pipeline
from gqa_rex_pipeline.gqa_rex_config import get_gqa_rex_config, validate_gqa_rex_paths


def main():
    """
    Main entry point for GQA-REX FDR Pipeline
    Multi-Agent Visual Question Answering with GQA-REX Explanations
    """
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        description="GQA-REX FDR Pipeline: Multi-Agent Visual Question Answering with Reasoning Explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 samples
  python main_gqa_rex.py --test 5

  # Run on validation set with evaluation
  python main_gqa_rex.py --dataset gqa_rex_val --evaluate --samples 50

  # Run on training set with vLLM backend
  python main_gqa_rex.py --dataset gqa_rex_train --backend vllm --samples 100

  # Full validation run
  python main_gqa_rex.py --dataset gqa_rex_val --backend vllm --evaluate
        """
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "openai"],
        default="vllm",
        help="Choose backend: 'vllm' for local vLLM server or 'openai' for OpenAI API (default: vllm)"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable comprehensive evaluation including explanation quality metrics"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process (overrides config). Use small numbers for testing."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gqa_rex_train", "gqa_rex_val"],
        default=None,
        help="Override active dataset from config (gqa_rex_train or gqa_rex_val)"
    )

    parser.add_argument(
        "--test",
        nargs='?',
        type=int,
        const=5,  # Default if --test is used without a value
        default=None,  # Default if --test is not used
        help="Run in test mode with limited samples. Optionally specify number (e.g., --test 10). Defaults to 5."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (uses default GQA-REX config if not specified)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset paths and configuration without running pipeline"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics and exit"
    )

    args = parser.parse_args()

    # Determine the number of samples to run
    is_test_mode = args.test is not None
    num_samples = args.samples
    if is_test_mode:
        num_samples = args.test  # --test overrides --samples

    use_vllm = args.backend == "vllm"

    # Display pipeline information
    print("🚀 GQA-REX FDR Pipeline")
    print("=" * 50)
    print(f"🔥 Backend: {'vLLM (local models)' if use_vllm else 'OpenAI API'}")
    
    if args.dataset:
        print(f"💾 Dataset: {args.dataset}")
    
    if is_test_mode:
        print(f"🧪 Test mode: {num_samples} samples")
    elif num_samples:
        print(f"📊 Processing: {num_samples} samples")
    else:
        print(f"📊 Processing: All available samples")
    
    if args.evaluate:
        print("📈 Comprehensive evaluation: ENABLED")

    print()

    # Validate configuration and dataset paths
    if args.validate or args.stats:
        try:
            from gqa_rex_pipeline.gqa_rex_loader import create_gqa_rex_loader
            
            split = "val" if (args.dataset == "gqa_rex_val" or args.dataset is None) else "train"
            logging.info(f"🔍 Validating GQA-REX {split} dataset...")
            
            # Create loader to validate paths and get stats
            loader = create_gqa_rex_loader(split=split)
            stats = loader.get_dataset_stats()
            
            print("✅ Dataset validation successful!")
            print(f"📊 Dataset Statistics for GQA-REX {split}:")
            print(f"   • Total linked samples: {stats['total_samples']}")
            print(f"   • GQA questions: {stats['gqa_questions']}")
            print(f"   • REX explanations: {stats['rex_explanations']}")
            
            if stats.get('scene_graphs'):
                print(f"   • Scene graphs: {stats['scene_graphs']}")
            
            if stats.get('top_semantic_types'):
                print("   • Top question types:")
                for q_type, count in stats['top_semantic_types'][:5]:
                    print(f"     - {q_type}: {count} samples")
            
            if args.validate:
                print("✅ Validation completed successfully!")
                return
            
            if args.stats:
                print("📊 Statistics displayed successfully!")
                return
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return

    # Check vLLM servers if using vLLM backend
    if use_vllm:
        print("🔍 Checking vLLM servers...")
        try:
            import requests
            
            # Check VL model server
            try:
                response = requests.get("http://localhost:9100/v1/models", timeout=5)
                if response.status_code == 200:
                    print("✅ VL model server (port 9100): Online")
                else:
                    print("⚠️ VL model server (port 9100): Responding but may have issues")
            except:
                print("❌ VL model server (port 9100): Offline or unreachable")
                print("   Please start with: CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen2.5-VL-7B-Instruct --port 9100")
            
            # Check LLM server
            try:
                response = requests.get("http://localhost:9200/v1/models", timeout=5)
                if response.status_code == 200:
                    print("✅ LLM server (port 9200): Online")
                else:
                    print("⚠️ LLM server (port 9200): Responding but may have issues")
            except:
                print("❌ LLM server (port 9200): Offline or unreachable")
                print("   Please start with: CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/Qwen2.5-7B-Instruct --port 9200")
                
        except ImportError:
            print("⚠️ Could not check vLLM servers (requests module not available)")
        
        print()

    # Run the pipeline
    try:
        logging.info("🎯 Starting GQA-REX FDR Pipeline...")
        
        results = run_gqa_rex_pipeline(
            use_vllm=use_vllm,
            enable_evaluation=args.evaluate,
            override_samples=num_samples,
            config_path=args.config,
            active_dataset_override=args.dataset
        )
        
        if results:
            accuracy = sum(1 for r in results if r.get('is_correct', False)) / len(results) * 100
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Final Accuracy: {accuracy:.2f}% ({sum(1 for r in results if r.get('is_correct', False))}/{len(results)})")
        else:
            print("\n⚠️ Pipeline completed but no results generated")
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
