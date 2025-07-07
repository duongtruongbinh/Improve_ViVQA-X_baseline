# FDR/main.py - FDR Central Control Point
import argparse
import logging
import os
import json
import warnings
import sys
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

from src.pipeline import run_fdr_pipeline

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def main():
    """
    Main entry point for FDR (Faithful Decomposed Reasoning) Framework
    Multi-Agent Vietnamese Visual Question Answering with Faithful Explanations
    """
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Log to console
        ]
    )

    parser = argparse.ArgumentParser(
        description="FDR: Multi-Agent Language Visual Question Answering with Faithful Explanations"
    )
    
    # Configuration is now unified in config.yaml - no --config argument needed
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "openai"],
        default="vllm",
        help="Choose backend: 'vllm' for local vLLM server or 'openai' for OpenAI API"
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
        help="Number of samples to process (overrides config). Is overridden by --test."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["vqax", "vivqax"],
        default=None,
        help="Override active dataset from config (e.g., VQAX, VIVQAX)"
    )

    parser.add_argument(
        "--test",
        nargs='?',
        type=int,
        const=2,  # Default if --test is used without a value
        default=None, # Default if --test is not used
        help="Run in test mode. Overrides --samples. Optionally specify number of samples (e.g., --test 5). Defaults to 2."
    )

    args = parser.parse_args()
    
    # Determine the number of samples to run
    is_test_mode = args.test is not None
    num_samples = args.samples
    if is_test_mode:
        num_samples = args.test # --test overrides --samples
        
    use_vllm = args.backend == "vllm"
    
    # Display pipeline information
    if use_vllm:
        logging.info("🔥 Using vLLM backend for better performance")
    else:
        logging.info("🌐 Using OpenAI backend")

    if args.evaluate:
        logging.info("📊 Comprehensive evaluation enabled")
    
    if args.dataset:
        logging.info(f"💾 Overriding active dataset to: {args.dataset}")
    
    if is_test_mode:
        logging.info("🧪 Test mode: Running with test configuration")

    try:
        logging.info("🚀 Starting FDR pipeline...")
        
        # Run FDR Pipeline with unified config
        results = run_fdr_pipeline(
            use_vllm=use_vllm, 
            enable_evaluation=args.evaluate,
            override_samples=num_samples,
            active_dataset_override=args.dataset
        )
        
        logging.info(f"✅ Pipeline returned {len(results) if results else 0} results")
        
        # Display completion summary
        logging.info(f"✅ Pipeline completed successfully!")
        logging.info(f"📊 Processed {len(results)} samples")
        
        # Display final summary (this was missing!)
        if results:
            logging.info("📋 Final Results Summary:")
            for i, result in enumerate(results[-2:]):  # Show last 2 results
                logging.info(f"  Sample {i+1}: Q='{result.get('question', '')[:50]}...'")
                logging.info(f"             A='{result.get('final_answer', 'N/A')}'")
                logging.info(f"             Status={result.get('synthesis_status', 'N/A')}")
        
        return results
        
    except FileNotFoundError as e:
        logging.error(f"FATAL: Required file not found. Check your config paths.")
        logging.error(f"Details: {e}")
        return None
    except ValueError as e:
        logging.error(f"FATAL: Configuration error: {e}")
        return None
    except Exception as e:
        logging.error(f"FATAL: Unexpected error during pipeline execution: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    main()
    # The final summary is now handled by the pipeline's table view.
    # This provides a clean exit point.
    print(f"\n✅ Script finished.") 