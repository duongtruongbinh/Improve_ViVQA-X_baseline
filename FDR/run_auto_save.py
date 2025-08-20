#!/usr/bin/env python3
# FDR/run_auto_save.py - Main script to run Auto Save Pipeline
import argparse
import logging
import os
import sys
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

from src.auto_save_pipeline import run_auto_save_pipeline

def main():
    """
    Main entry point for Auto Save FDR Pipeline
    Automatically saves correct and wrong samples to separate directories
    """
    parser = argparse.ArgumentParser(
        description="Auto Save FDR: Multi-Agent Language Visual Question Answering with Automatic Correct/Wrong Classification"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "openai"],
        default="vllm",
        help="Choose backend: 'vllm' for local vLLM server or 'openai' for OpenAI API"
    )
    
    parser.add_argument(
        "--correct",
        type=int,
        default=3,
        help="Number of correct samples to collect (default: 3)"
    )
    
    parser.add_argument(
        "--wrong", 
        type=int,
        default=3,
        help="Number of wrong samples to collect (default: 3)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="auto_output",
        help="Base output directory (default: auto_output)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["vqax", "vivqax"],
        default=None,
        help="Override active dataset from config (e.g., VQAX, VIVQAX)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (uses config.yaml by default)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logging.info("🚀 Starting Auto Save FDR Pipeline")
    logging.info(f"📊 Target: {args.correct} correct, {args.wrong} wrong samples")
    logging.info(f"🔧 Backend: {args.backend}")
    logging.info(f"📁 Output: {args.output_dir}")
    
    try:
        # Run Auto Save Pipeline
        summary = run_auto_save_pipeline(
            use_vllm=(args.backend == "vllm"),
            config_path=args.config,
            active_dataset_override=args.dataset,
            max_correct=args.correct,
            max_wrong=args.wrong,
            output_base_dir=args.output_dir
        )
        
        if summary:
            print(f"\n🎉 Auto Save Pipeline completed successfully!")
            print(f"📊 Summary:")
            print(f"  - Total processed: {summary['total_processed']}")
            print(f"  - Correct samples: {summary['correct_samples']}")
            print(f"  - Wrong samples: {summary['wrong_samples']}")
            print(f"  - Target reached: {summary['target_reached']}")
            print(f"📁 Output directory: {args.output_dir}")
            
            # Show directory structure
            print(f"\n📂 Directory structure:")
            print(f"  {args.output_dir}/")
            print(f"  ├── correct/")
            for i in range(1, summary['correct_samples'] + 1):
                print(f"  │   └── sample_{i}/")
                print(f"  │       ├── info.json")
                print(f"  │       ├── input_image.jpg")
                print(f"  │       └── output_visualization.png")
            print(f"  ├── wrong/")
            for i in range(1, summary['wrong_samples'] + 1):
                print(f"  │   └── sample_{i}/")
                print(f"  │       ├── info.json")
                print(f"  │       ├── input_image.jpg")
                print(f"  │       └── output_visualization.png")
            print(f"  └── summary.json")
            
        return summary
        
    except Exception as e:
        logging.error(f"❌ Auto Save Pipeline failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main() 