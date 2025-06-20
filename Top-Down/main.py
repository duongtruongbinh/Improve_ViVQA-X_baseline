# Top-Down/main.py
import argparse
import logging
import os
import json
import base64
import warnings
from core.pipeline import run_siri_pipeline

# Suppress warnings for clean test output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def main():
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Log to console
        ]
    )

    parser = argparse.ArgumentParser(
        description="Run the SIRI framework for VQA using a Top-Down Reasoning approach."
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="Top-Down/configs/vivqax_config.yaml", 
        help="Path to the configuration YAML file for the SIRI pipeline."
    )
    
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
        help="Test mode: Run refactored pipeline with 10 questions"
    )

    args = parser.parse_args()
    
    use_vllm = args.backend == "vllm"
    
    if use_vllm:
        logging.info("🚀 Using local vLLM backend")
    else:
        logging.info("🌐 Using OpenAI backend")

    try:
        if args.test:
            # Test mode: Run refactored pipeline with 10 questions
            logging.info("🧪 Test mode: Running refactored pipeline with 10 questions")
            run_siri_pipeline(args.config, use_vllm=use_vllm)
        else:
            # Full pipeline mode
            run_siri_pipeline(args.config, use_vllm=use_vllm)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found. Please check your paths in the config file.")
        logging.error(f"Details: {e}")
    except ValueError as e:
        logging.error(f"FATAL: A configuration error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the pipeline execution: {e}", exc_info=True)


if __name__ == "__main__":
    main() 