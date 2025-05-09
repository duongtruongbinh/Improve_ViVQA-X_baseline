# main.py (Final Version - Integrates utils.py/Grader)

import asyncio
import argparse
import os
import torch # For DataLoader and dataset selection logic
from torch.utils.data import Subset, DataLoader
import json # Needed for final stats saving logic potentially

# --- AutoGen VQA System Imports ---
# Assuming all .py files are in the same directory or accessible via PYTHONPATH
try:
    from config_loader import app_config # app_config is loaded when config_loader is imported
    from main_vqa_flow import run_vqa_pipeline # Import the core pipeline function
    # Import actual dataloaders provided by user
    from dataloader import VQAv2Dataset, GQADataset
    # Try importing utilities needed at the end
    from utils import Grader, write_response_to_json, record_final_accuracy, Colors
except ImportError as e:
    print(f"Import Error in main.py: {e}. Please ensure all required .py files "
          "(config_loader, main_vqa_flow, dataloader, agents, utils, etc.) "
          "are in the correct path and required classes/functions exist.")
    exit(1)
# --- End Imports ---


async def main_logic_entry_point():
    """
    Main logic for argument parsing, dataset loading, and orchestrating VQA runs.
    Integrates Grader from utils.py for final accuracy calculation.
    """
    current_config = app_config # Use the globally loaded config

    print(f"Torch version: {torch.__version__}")
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision not found or importable.")

    # --- Pre-check critical config sections ---
    if not isinstance(current_config, dict) or not current_config:
        print("CRITICAL ERROR: Configuration loaded incorrectly or is empty. Check config.yaml and config_loader.py.")
        return
    if "datasets" not in current_config or "inference_settings" not in current_config or "vllm_details" not in current_config:
        print("CRITICAL ERROR: 'datasets', 'inference_settings', or 'vllm_details' section missing in loaded config.yaml.")
        return

    dataset_s_config = current_config["datasets"]
    inference_s_config = current_config["inference_settings"]
    vllm_details_conf = current_config["vllm_details"]

    # --- Get confirmed defaults AFTER loading config ---
    default_dataset_name = dataset_s_config.get("dataset_name")
    if default_dataset_name is None:
         print("CRITICAL ERROR: 'dataset_name' key is missing within 'datasets' in config.yaml.")
         return
    if default_dataset_name not in ['vqa-v2', 'gqa']:
         print(f"CRITICAL ERROR: Invalid 'dataset_name' ('{default_dataset_name}') found in config.yaml. Must be 'vqa-v2' or 'gqa'.")
         return

    default_verbose = inference_s_config.get("verbose", False)
    default_use_num = dataset_s_config.get("use_num_test_data", False)
    default_num_test = dataset_s_config.get("num_test_data", 10)
    default_split_gqa = dataset_s_config.get("gqa_dataset_split", "val")
    default_split_vqa = dataset_s_config.get("vqa_v2_dataset_split", "rest-val")

    # --- Argument Parsing (using confirmed defaults) ---
    parser = argparse.ArgumentParser(description="AutoGen VQA Pipeline Runner - Main Entry Point")
    parser.add_argument("--dataset_name", type=str, default=default_dataset_name,
                        choices=['vqa-v2', 'gqa'], help="Dataset name.")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="Specific dataset split. Overrides config.yaml.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=default_verbose,
                        help="Enable/disable verbose logging.")
    parser.add_argument("--use_num_test_data", action=argparse.BooleanOptionalAction, default=default_use_num,
                        help="Use 'num_test_data' for subset size.")
    parser.add_argument("--num_test_data", type=int, default=default_num_test,
                        help="Number of test items if --use_num_test_data.")

    cli_args = parser.parse_args()

    # --- Determine final effective values ---
    effective_dataset_name = cli_args.dataset_name
    effective_verbose = cli_args.verbose
    effective_use_num = cli_args.use_num_test_data
    effective_num_test = cli_args.num_test_data

    effective_split_name = cli_args.dataset_split
    if not effective_split_name:
        if effective_dataset_name == 'gqa':
            effective_split_name = default_split_gqa
        elif effective_dataset_name == 'vqa-v2':
            effective_split_name = default_split_vqa
        else:
            effective_split_name = "unknown"

    # --- Update global config with effective values ---
    # This ensures consistency if any module re-imports app_config, although it's generally
    # better practice to pass specific needed values down if possible.
    current_config["datasets"]["dataset_name"] = effective_dataset_name
    current_config["inference_settings"]["verbose"] = effective_verbose
    current_config["datasets"]["use_num_test_data"] = effective_use_num
    current_config["datasets"]["num_test_data"] = effective_num_test
    if effective_dataset_name == 'gqa':
        current_config["datasets"]["gqa_dataset_split"] = effective_split_name
    elif effective_dataset_name == 'vqa-v2':
        current_config["datasets"]["vqa_v2_dataset_split"] = effective_split_name

    # --- Print Effective Configuration ---
    print(f"\n--- Effective Configuration for this Run ---")
    print(f"Dataset: {effective_dataset_name}, Split: {effective_split_name}")
    print(f"Verbose Logging: {effective_verbose}")
    print(f"Using Fixed Number of Test Data: {effective_use_num}")
    if effective_use_num:
        print(f"Number of Test Data Items: {effective_num_test}")
    print(f"VLM for AutoGen (via vLLM): {vllm_details_conf.get('vlm_model_name')}")
    print(f"LLM for AutoGen (via vLLM): {vllm_details_conf.get('llm_model_name')}")
    print(f"vLLM Base URL: {vllm_details_conf.get('base_url')}")
    print(f"----------------------------------------\n")

    # --- Device Info ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Script recognized device: {device}')
    if torch.cuda.is_available():
        print(f'Torch.distributed.is_available: {torch.distributed.is_available()}')
        print(f'Using {torch.cuda.device_count()} GPU(s).')

    # --- Prepare Datasets (using user's dataloader.py) ---
    print("Loading dataset using classes from dataloader.py...")
    try:
        if effective_dataset_name == 'gqa':
            full_dataset = GQADataset(current_config, transform=None)
        elif effective_dataset_name == 'vqa-v2':
            full_dataset = VQAv2Dataset(current_config, transform=None)
        else:
            raise ValueError(f"Internal Error: Unsupported dataset name '{effective_dataset_name}' determined.")
    except FileNotFoundError as e_data:
         print(f"CRITICAL ERROR: FileNotFoundError during dataset loading: {e_data}. Check paths in config.yaml.")
         return
    except Exception as e_load:
        print(f"CRITICAL ERROR: Failed to instantiate Dataset class for '{effective_dataset_name}'. Check dataloader.py and config.")
        print(f"Error details: {e_load}")
        return

    if not full_dataset or len(full_dataset) == 0:
        print(f"CRITICAL ERROR: Dataset loaded via dataloader.py for {effective_dataset_name} - {effective_split_name} is empty. Exiting.")
        return

    # --- Subset Selection ---
    torch.manual_seed(0)
    datasets = current_config.get("datasets", {})
    if effective_use_num: # Use fixed number
        num_to_use = int(effective_num_test)
        if num_to_use <= 0: num_to_use = 1 if len(full_dataset) > 0 else 0
        actual_num_to_use = min(num_to_use, len(full_dataset))
        if actual_num_to_use < num_to_use : print(f"Warning: Requested num_test_data ({num_to_use}) > available ({len(full_dataset)}). Using {actual_num_to_use}.")
        if actual_num_to_use == 0: print("Error: num_test_data results in 0 items. Exiting."); return
        print(f"Selecting {actual_num_to_use} items randomly using 'num_test_data' setting.")
        test_subset_idx = torch.randperm(len(full_dataset))[:actual_num_to_use]
        test_subset = Subset(full_dataset, test_subset_idx.tolist())
    else: # Use percentage
        percent_test = float(datasets.get("percent_test", 1.0))
        num_items_by_percent = int(percent_test * len(full_dataset))
        if num_items_by_percent <= 0: num_items_by_percent = 1 if len(full_dataset) > 0 else 0
        if num_items_by_percent == 0: print("Error: 'percent_test' results in 0 items. Exiting."); return
        print(f"Selecting {num_items_by_percent} items ({percent_test*100:.1f}%) randomly using 'percent_test' setting.")
        test_subset_idx = torch.randperm(len(full_dataset))[:num_items_by_percent]
        test_subset = Subset(full_dataset, test_subset_idx.tolist())

    if len(test_subset) == 0:
        print("CRITICAL ERROR: Resulting data subset has 0 items after selection.")
        return

    # Using DataLoader settings from user's original main.py
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    print(f"Created DataLoader with {len(test_subset)} samples to process (num batches: {len(test_loader)}).")

    # --- Initialize Grader and Results Storage ---
    grader_instance = Grader() # Instantiate the grader from utils.py
    all_results_dict = {} # Store results keyed by question_id for final JSON saving
    output_filename = inference_settings.get("output_response_filename", "outputs/autogen_vqa_default.jsonl")
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if inference_settings.get("save_output_response") and output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Warning: Could not create output directory {output_dir}: {e}. Individual results might not be saved correctly by external utils if needed.")
            # Saving the final JSON might still work if the directory can be created later.

    # --- Start AutoGen VQA Inference Loop ---
    print(f"\nStarting AutoGen VQA processing loop...")
    processed_count = 0
    error_count = 0
    for i, data_batch_from_loader in enumerate(test_loader):
        # Unpack batch data
        image_path_val = data_batch_from_loader['image_path'][0]
        question_val = data_batch_from_loader['question'][0]
        question_id_val = data_batch_from_loader['question_id'][0]
        target_answer_val = data_batch_from_loader['answer'][0]
        current_question_id = str(question_id_val.item()) if hasattr(question_id_val, 'item') else str(question_id_val)

        # Log progress
        if effective_verbose:
            print(f"\n[Main Loop] Processing item {i+1}/{len(test_loader)}: QID {current_question_id}")
        elif (i + 1) % inference_settings.get("print_every", 10) == 0 or i == len(test_loader) - 1 :
            print(f"[Main Loop] Processing item {i + 1}/{len(test_loader)}: QID {current_question_id}")

        # Check image existence
        if not os.path.exists(image_path_val):
            print(f"ERROR: Image not found at '{image_path_val}' for QID {current_question_id}. Skipping.")
            # Store basic error info, more details not available as pipeline didn't run
            all_results_dict[current_question_id] = {"error": "ImageFileNotFound", "final_answer": "[Answer Failed]", "question": question_val, "target_answer": target_answer_val}
            error_count += 1
            continue

        # Call the AutoGen pipeline
        try:
            pipeline_result_data = await run_vqa_pipeline(
                image_path=image_path_val,
                question=question_val,
                question_id=current_question_id,
                target_answer=target_answer_val
            )

            # Accumulate grades using the Grader instance
            # Need to extract grades and match_baseline_failed from the result
            grades = pipeline_result_data.get("grades", [])
            match_failed = pipeline_result_data.get("match_baseline_failed", False) # Default to False if missing

            # Call accumulate_grades which also calculates majority vote
            # It requires the config ('args' in its definition)
            if grades: # Only accumulate if grading was actually performed
                 majority_vote_str = grader_instance.accumulate_grades(current_config, grades, match_failed)
                 pipeline_result_data["majority_vote"] = majority_vote_str # Add majority vote to results
            else:
                 pipeline_result_data["majority_vote"] = "Grading Skipped"


            # Store the full result data
            all_results_dict[current_question_id] = pipeline_result_data
            processed_count += 1
            if pipeline_result_data.get("error"):
                 error_count += 1

        except Exception as e_pipeline:
            print(f"CRITICAL ERROR during run_vqa_pipeline for QID {current_question_id}: {e_pipeline}")
            import traceback
            traceback.print_exc()
            # Store error information
            error_result_data = {
                "question_id": current_question_id, "image_path": image_path_val, "question": question_val,
                "target_answer": target_answer_val, "initial_answer": "", "final_answer": "[Answer Failed - Pipeline Error]",
                "match_baseline_failed": False, "is_numeric_reattempt": False, "analysis_output": "",
                "object_attributes_queried": "", "reattempt_answer": "", "grades": [],
                "processing_time_seconds": 0.0, "error": f"PipelineRuntimeError: {str(e_pipeline)}"
            }
            all_results_dict[current_question_id] = error_result_data
            error_count += 1

    # --- End of Loop ---
    print(f"\n--- AutoGen VQA Processing Finished ---")
    print(f"Total items processed: {processed_count}")
    if error_count > 0:
         print(f"Number of items with errors during processing: {error_count}")

    # --- Final Accuracy Calculation & Saving ---
    print("\nCalculating final accuracy and saving results...")
    try:
        # Use the Grader instance which has accumulated results throughout the loop
        baseline_accuracy, final_accuracy, stats = grader_instance.average_score()

        print(f"Final Accuracy (Overall): {final_accuracy * 100:.2f}%")
        print(f"Baseline Accuracy (if applicable): {baseline_accuracy * 100:.2f}%")
        print(f"Stats: {stats}")

        # Save the final dictionary with all results to JSON
        # Add accuracy stats to the dictionary before saving
        all_results_dict['overall_baseline_accuracy'] = baseline_accuracy
        all_results_dict['overall_final_accuracy'] = final_accuracy
        all_results_dict['overall_stats'] = stats

        if inference_settings.get("save_output_response"):
            # Use the write_response_to_json logic (or adapt your function)
            # This function writes the entire dict keyed by qid
            try:
                 # Ensure directory exists one last time
                 output_dir = os.path.dirname(output_filename)
                 if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                 
                 with open(output_filename, 'w', encoding='utf-8') as f:
                     json.dump(all_results_dict, f, indent=4) # Use indent for readability
                 print(f"Final results saved to: {output_filename}")
            except Exception as e_save:
                 print(f"ERROR saving final JSON results to {output_filename}: {e_save}")
        else:
             print("Skipping saving final results file as per configuration.")

    except NameError as e: # Catch if Grader wasn't defined due to utils import failure
         if 'Grader' in str(e): print("Could not calculate final accuracy because 'Grader' class from 'utils.py' was not available.")
         else: print(f"An unexpected NameError occurred during final processing: {e}")
    except Exception as e_final:
        print(f"Error during final accuracy calculation or saving: {e_final}")

if __name__ == "__main__":
    # Removed Gemini/VertexAI specific setup
    try:
        asyncio.run(main_logic_entry_point())
    except FileNotFoundError as e:
        print(f"\nCRITICAL FileNotFoundError: {e}. Check config.yaml and paths.")
    except ImportError as e:
        print(f"\nCRITICAL ImportError: {e}. Check libraries and project structure.")
    except Exception as e_main:
        print(f"\nCRITICAL unexpected error in main execution: {e_main}")
        import traceback
        traceback.print_exc()