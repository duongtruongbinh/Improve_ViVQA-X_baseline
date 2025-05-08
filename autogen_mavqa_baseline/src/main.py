# main.py

import asyncio
import argparse
import os
import torch # For DataLoader and dataset selection logic
from torch.utils.data import Subset, DataLoader

# --- AutoGen VQA System Imports ---
# Assuming all .py files are in the same directory or accessible via PYTHONPATH
try:
    from config_loader import app_config # app_config is loaded when config_loader is imported
    from main_vqa_flow import run_vqa_pipeline, write_response_to_jsonl
    # Import actual dataloaders provided by user
    from dataloader import VQAv2Dataset, GQADataset
except ImportError as e:
    print(f"Import Error in main.py: {e}. Please ensure all required .py files (config_loader, main_vqa_flow, dataloader, agents, etc.) are in the correct path.")
    exit(1)
# --- End AutoGen Imports ---


async def main_logic_entry_point():
    """
    Main logic for argument parsing, dataset loading, and orchestrating VQA runs.
    Derived from user's original main script, adapted for AutoGen.
    """
    current_config = app_config # Use the globally loaded config

    print(f"Torch version: {torch.__version__}")
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision not found or importable.")

    # --- Argument Parsing (mirrors user's original main.py structure) ---
    parser = argparse.ArgumentParser(description="AutoGen VQA Pipeline Runner - Main Entry Point")
    
    dataset_s_config = current_config.get("dataset_settings", {})
    inference_s_config = current_config.get("inference_settings", {})

    parser.add_argument("--dataset_name", type=str,
                        default=dataset_s_config.get("dataset_name"),
                        choices=['vqa-v2', 'gqa'],
                        help="Dataset name. Overrides config.yaml.")
    parser.add_argument("--dataset_split", type=str,
                        help="Specific dataset split. Overrides config.yaml. E.g., for vqa-v2: 'rest-val', 'val'; for gqa: 'val', 'val-subset'.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=inference_s_config.get("verbose"),
                        help="Enable/disable verbose logging. Overrides config.yaml.")
    parser.add_argument("--use_num_test_data", action=argparse.BooleanOptionalAction,
                        default=dataset_s_config.get("use_num_test_data"),
                        help="Use 'num_test_data' for subset size. Overrides config.yaml.")
    parser.add_argument("--num_test_data", type=int,
                        default=dataset_s_config.get("num_test_data"),
                        help="Number of test items if --use_num_test_data. Overrides config.yaml.")

    cli_args = parser.parse_args()

    # --- Update global app_config with CLI overrides ---
    current_config.setdefault("dataset_settings", {})["dataset_name"] = cli_args.dataset_name
    current_config.setdefault("inference_settings", {})["verbose"] = cli_args.verbose
    current_config.setdefault("dataset_settings", {})["use_num_test_data"] = cli_args.use_num_test_data
    if cli_args.num_test_data is not None and \
       (cli_args.num_test_data != parser.get_default("num_test_data") or current_config["dataset_settings"]["use_num_test_data"]):
        current_config.setdefault("dataset_settings", {})["num_test_data"] = cli_args.num_test_data

    active_dataset_name = current_config["dataset_settings"]["dataset_name"]
    active_split_name = cli_args.dataset_split
    if not active_split_name:
        if active_dataset_name == 'gqa':
            active_split_name = current_config["dataset_settings"].get("gqa_dataset_split", "val")
        elif active_dataset_name == 'vqa-v2':
            active_split_name = current_config["dataset_settings"].get("vqa_v2_dataset_split", "rest-val")
        else:
             print(f"Warning: Invalid dataset_name '{active_dataset_name}'. Check config and arguments.")
             active_split_name = "unknown"

    if active_dataset_name == 'gqa':
        current_config["dataset_settings"]["gqa_dataset_split"] = active_split_name
    elif active_dataset_name == 'vqa-v2':
        current_config["dataset_settings"]["vqa_v2_dataset_split"] = active_split_name

    print(f"\n--- Effective Configuration for this Run ---")
    print(f"Dataset: {active_dataset_name}, Split: {active_split_name}")
    print(f"Verbose Logging: {current_config['inference_settings']['verbose']}")
    print(f"Using Fixed Number of Test Data: {current_config['dataset_settings']['use_num_test_data']}")
    if current_config['dataset_settings']['use_num_test_data']:
        print(f"Number of Test Data Items: {current_config['dataset_settings']['num_test_data']}")
    vllm_details_conf = current_config.get("vllm_details", {})
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
        if active_dataset_name == 'gqa':
            # Pass the whole config dict as 'args' to match original dataloader constructor
            full_dataset = GQADataset(current_config, transform=None)
        elif active_dataset_name == 'vqa-v2':
            full_dataset = VQAv2Dataset(current_config, transform=None)
        else:
            raise ValueError(f"Unsupported dataset name '{active_dataset_name}' specified.")
    except FileNotFoundError as e_data:
         print(f"CRITICAL ERROR: FileNotFoundError during dataset loading: {e_data}. Check paths in config.yaml.")
         return
    except Exception as e_load:
        print(f"CRITICAL ERROR: Failed to instantiate Dataset class for '{active_dataset_name}'. Check dataloader.py and config.")
        print(f"Error details: {e_load}")
        return

    if not full_dataset or len(full_dataset) == 0:
        print(f"CRITICAL ERROR: Dataset loaded via dataloader.py for {active_dataset_name} - {active_split_name} is empty. "
              "Please check dataloader logic and config.yaml paths. Exiting.")
        return

    # --- Subset Selection ---
    torch.manual_seed(0)
    dataset_settings = current_config.get("dataset_settings", {}) # Re-fetch for safety
    if dataset_settings.get("use_num_test_data"):
        num_to_use = int(dataset_settings.get("num_test_data", 0))
        if num_to_use <= 0: num_to_use = 1 if len(full_dataset) > 0 else 0
        actual_num_to_use = min(num_to_use, len(full_dataset))
        if actual_num_to_use < num_to_use : print(f"Warning: Requested num_test_data ({num_to_use}) > available ({len(full_dataset)}). Using {actual_num_to_use}.")
        if actual_num_to_use == 0: print("Error: num_test_data results in 0 items. Exiting."); return
        print(f"Selecting {actual_num_to_use} items randomly using 'num_test_data' setting.")
        test_subset_idx = torch.randperm(len(full_dataset))[:actual_num_to_use]
        test_subset = Subset(full_dataset, test_subset_idx.tolist())
    else:
        percent_test = float(dataset_settings.get("percent_test", 1.0))
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

    # --- Start AutoGen VQA Inference Loop ---
    all_run_results_summary = []
    inference_settings = current_config.get("inference_settings", {})
    output_filename = inference_settings.get("output_response_filename", "outputs/autogen_vqa_default.jsonl")
    
    output_dir = os.path.dirname(output_filename)
    if inference_settings.get("save_output_response") and output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Warning: Could not create output directory {output_dir}: {e}. Saving disabled.")
            inference_settings["save_output_response"] = False

    print(f"\nStarting AutoGen VQA processing loop...")
    for i, data_batch_from_loader in enumerate(test_loader):
        image_path_val = data_batch_from_loader['image_path'][0]
        question_val = data_batch_from_loader['question'][0]
        question_id_val = data_batch_from_loader['question_id'][0]
        target_answer_val = data_batch_from_loader['answer'][0]
        current_question_id = str(question_id_val.item()) if hasattr(question_id_val, 'item') else str(question_id_val)

        # Verbose logging decision for item level
        is_verbose = inference_settings.get("verbose", False)
        if not is_verbose and (i + 1) % inference_settings.get("print_every", 10) == 0 or i == len(test_loader) - 1:
            print(f"[Main Loop] Processing item {i + 1}/{len(test_loader)}: QID {current_question_id}")

        if not os.path.exists(image_path_val):
            print(f"ERROR: Image not found at '{image_path_val}' for QID {current_question_id}. Skipping.")
            error_result = {"question_id": current_question_id, "error": "ImageFileNotFound", "final_answer": "[Answer Failed]"}
            all_run_results_summary.append({"question_id": current_question_id, "status": "Error - Image Not Found"})
            if inference_settings.get("save_output_response"):
                write_response_to_jsonl(error_result, output_filename)
            continue
        
        try:
            # Core call to the AutoGen pipeline
            pipeline_result_data = await run_vqa_pipeline(
                image_path=image_path_val,
                question=question_val,
                question_id=current_question_id,
                target_answer=target_answer_val
            )
            all_run_results_summary.append({ # Append summary for final stats
                "question_id": current_question_id,
                "final_answer": pipeline_result_data.get("final_answer"),
                "status": "Processed" if not pipeline_result_data.get("error") else "ErrorInPipeline",
                "grades": pipeline_result_data.get("grades", [])
            })
        except Exception as e_pipeline:
            print(f"CRITICAL ERROR during run_vqa_pipeline for QID {current_question_id}: {e_pipeline}")
            import traceback
            traceback.print_exc()
            # Structure matches output from run_vqa_pipeline on error
            error_result = {
                "question_id": current_question_id, "image_path": image_path_val, "question": question_val,
                "target_answer": target_answer_val, "initial_answer": "", "final_answer": "[Answer Failed - Pipeline Error]",
                "match_baseline_failed": False, "is_numeric_reattempt": False,
                "analysis_output": "", "object_attributes_queried": "", "reattempt_answer": "",
                "grades": [], "processing_time_seconds": 0.0, "error": f"PipelineRuntimeError: {str(e_pipeline)}"
            }
            all_run_results_summary.append({"question_id": current_question_id, "status": f"Error - Pipeline Exception"})
            if inference_settings.get("save_output_response"):
                 write_response_to_jsonl(error_result, output_filename)

    print(f"\n--- AutoGen VQA Processing Finished ---")
    print(f"Total items processed: {len(all_run_results_summary)}")
    
    # --- Final Accuracy Calculation (Placeholder using Grader logic) ---
    # This section assumes you have a Grader class potentially from utils.py
    # If not, it provides a very basic accuracy calculation based on grades.
    print("\nCalculating final accuracy...")
    try:
        # Try importing the Grader class - USER NEEDS TO ENSURE utils.py IS AVAILABLE
        from utils import Grader # Assuming Grader is in utils.py in the path
        
        grader_instance = Grader()
        # This requires reading the output file or using detailed results stored during the run
        # Option 1: Re-read the output file (simpler if file saving was enabled)
        if inference_settings.get("save_output_response") and os.path.exists(output_filename):
            print(f"Calculating accuracy from output file: {output_filename}")
            full_results_list = []
            with open(output_filename, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        full_results_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {output_filename}")

            if full_results_list:
                 # Assuming Grader class has methods like accumulate_grades_from_list and average_score
                 # You might need to adapt this call based on your Grader class's exact methods
                 # grader_instance.accumulate_from_list(full_results_list) # Hypothetical method
                 
                 # Manual accumulation based on the eval script's logic if Grader class is simple
                 for result_item in full_results_list:
                      # Ensure grades exist and it's not an error state without grades
                     if result_item.get("grades") and not result_item.get("error"):
                         # Pass app_config if accumulate_grades_simple needs it
                         grader_instance.accumulate_grades_simple(current_config, result_item["grades"]) 
                 
                 accuracy, stats = grader_instance.average_score_simple() # Assuming this method exists
                 print(f"Final Accuracy calculated by Grader: {accuracy * 100:.2f}%")
                 # print(f"Stats from Grader: {stats}") # If stats are returned
            else:
                 print("Output file is empty or unreadable, cannot calculate accuracy.")

        else:
             print("Output file saving was disabled or file not found. Calculating basic accuracy from memory.")
             # Option 2: Calculate basic accuracy from all_run_results_summary (less detailed)
             correct_count = 0
             total_graded = 0
             for result in all_run_results_summary:
                if result.get("status") == "Processed" and result.get("grades"):
                    correct_votes = sum(1 for g in result["grades"] if "[Correct]" in g)
                    if correct_votes >= (len(result["grades"]) / 2.0): # Simple majority
                        correct_count += 1
                    total_graded += 1
             if total_graded > 0:
                 final_accuracy_basic = (correct_count / total_graded) * 100
                 print(f"Placeholder Accuracy (simple majority): {final_accuracy_basic:.2f}% ({correct_count}/{total_graded})")
             else:
                 print("No items were successfully graded to calculate basic accuracy.")

    except ImportError:
        print("Warning: 'utils.py' or 'Grader' class not found. Skipping detailed accuracy calculation.")
        print("Provide 'utils.py' with a 'Grader' class for accuracy stats.")
    except Exception as e_grade:
        print(f"Error during final accuracy calculation: {e_grade}")
        print("Skipping detailed accuracy calculation.")

if __name__ == "__main__":
    try:
        asyncio.run(main_logic_entry_point())
    except FileNotFoundError as e:
        print(f"\nCRITICAL FileNotFoundError: {e}. Check config.yaml and dataset paths.")
    except ImportError as e:
        print(f"\nCRITICAL ImportError: {e}. Check library installations (pyautogen, torch, pyyaml, etc.) and project structure.")
    except Exception as e_main:
        print(f"\nCRITICAL unexpected error in main execution: {e_main}")
        import traceback
        traceback.print_exc()