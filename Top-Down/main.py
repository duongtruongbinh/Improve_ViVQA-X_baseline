# Top-Down/main.py
import argparse
import logging
import os
from core.pipeline import run_siri_pipeline

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
        default="Top-Down/configs/vivqa_config.yaml", 
        help="Path to the configuration YAML file for the SIRI pipeline."
    )

    args = parser.parse_args()

    try:
        run_siri_pipeline(args.config)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found. Please check your paths in the config file.")
        logging.error(f"Details: {e}")
    except ValueError as e:
        logging.error(f"FATAL: A configuration error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the pipeline execution: {e}", exc_info=True)


if __name__ == "__main__":
    main() 