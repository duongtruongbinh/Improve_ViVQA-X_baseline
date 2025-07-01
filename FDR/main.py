# FDR/main.py - FDR Central Control Point
import argparse
import logging
import os
import json
import warnings
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
        "--test",
        action="store_true",
        help="Test mode: Run pipeline with limited samples for testing"
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
        help="Number of samples to process (overrides config)"
    )

    args = parser.parse_args()
    
    use_vllm = args.backend == "vllm"
    
    # Display pipeline information
    if use_vllm:
        logging.info("🔥 Using vLLM backend for better performance")
    else:
        logging.info("🌐 Using OpenAI backend")

    if args.evaluate:
        logging.info("📊 Comprehensive evaluation enabled")
    
    if args.test:
        logging.info("🧪 Test mode: Running with test configuration")

    try:
        # Run FDR Pipeline with unified config
        results = run_fdr_pipeline(
            use_vllm=use_vllm, 
            enable_evaluation=args.evaluate,
            override_samples=args.samples
        )
        
        # Display completion summary
        logging.info(f"✅ Pipeline completed successfully!")
        logging.info(f"📊 Processed {len(results)} samples")
        
        return results
        
    except FileNotFoundError as e:
        logging.error(f"FATAL: Required file not found. Check your config paths.")
        logging.error(f"Details: {e}")
        return None
    except ValueError as e:
        logging.error(f"FATAL: Configuration error: {e}")
        return None
    except Exception as e:
        logging.error(f"FATAL: Unexpected error during pipeline execution: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main() 