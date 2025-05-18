# main.py
import asyncio
import argparse
import os
import torch
from torch.utils.data import Subset, DataLoader
import json
import traceback
import logging
import datetime
import re
import sys

try:
    from config_loader import app_config
    from main_vqa_flow import run_vqa_pipeline
    from dataloader import VQAv2Dataset, GQADataset
    from utils import Grader
except ImportError as e:
    print(f"Import Error in main.py: {e}. Please ensure all required .py files "
          "(config_loader, main_vqa_flow, dataloader, utils, etc.) "
          "are in the correct path and modules exist.")
    exit(1)

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'

class RemoveColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape_pattern.sub('', message)

APP_LOGGER_NAME = "VQA_AutoGen_App"

def setup_logging(cli_args, current_config_dict):
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(logging.DEBUG)
    if app_logger.hasHandlers():
        app_logger.handlers.clear()
    app_logger.propagate = False

    outputs_base_dir = "outputs"
    if current_config_dict and "inference_settings" in current_config_dict and \
       current_config_dict["inference_settings"].get("output_base_dir"):
        outputs_base_dir = current_config_dict["inference_settings"]["output_base_dir"]
    os.makedirs(outputs_base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_folder = cli_args.dataset_name.replace('-', '_') if cli_args.dataset_name else "unknown_dataset"
    split_name_for_folder = cli_args.dataset_split.replace('-', '_') if cli_args.dataset_split else "unknown_split"

    if cli_args.use_num_test_data:
        num_data_str = f"num{cli_args.num_test_data}"
    else:
        percent_test_val = current_config_dict.get("datasets", {}).get("percent_test", 1.0)
        num_data_str = f"pct{int(percent_test_val*100)}"

    run_folder_name = f"run_{timestamp}_{dataset_name_for_folder}_{split_name_for_folder}_{num_data_str}_seed{cli_args.random_seed}"
    current_run_output_dir = os.path.join(outputs_base_dir, run_folder_name)
    os.makedirs(current_run_output_dir, exist_ok=True)
    log_file_path = os.path.join(current_run_output_dir, "execution.log")

    ch = logging.StreamHandler(sys.stdout)
    if cli_args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(console_formatter)
    app_logger.addHandler(ch)

    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.DEBUG)
    file_formatter = RemoveColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(file_formatter)
    app_logger.addHandler(fh)

    noisy_libraries = ["httpx", "openai", "aiohttp", "asyncio", "urllib3", "httpcore", "uvicorn", "PIL"]
    for lib_name in noisy_libraries:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)

    app_logger.info(f"Application logging initialized. Console Level: {'DEBUG' if cli_args.verbose else 'INFO'}. File Level: DEBUG.")
    app_logger.info(f"Log file: {log_file_path}")
    app_logger.info(f"All outputs for this run will be stored in: {current_run_output_dir}")
    return app_logger, current_run_output_dir

# Added: Function to determine question type
def get_question_type(question_text: str) -> str:
    """Classifies a question into 'yes/no', 'number', or 'other'."""
    q_lower = question_text.lower().strip()
    
    # Keywords for yes/no questions (check if question starts with these)
    yes_no_starters = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", "am ",
        "can ", "could ", "will ", "would ", "should ", 
        "has ", "have ", "had ", "may ", "might ", "must ",
        "is there ", "are there ", "was there ", "were there ",
        "can there ", "will there ", "is it ", "are they "
    ]
    # Keywords for number questions
    number_starters = [
        "how many", "what is the number of", "count the number of", "what number"
    ]

    if any(q_lower.startswith(s) for s in yes_no_starters):
        return "yes/no"
    if any(q_lower.startswith(s) for s in number_starters):
        return "number"
    return "other"

async def main_logic_entry_point():
    current_config = app_config
    
    default_dataset_name_cfg = current_config.get("datasets", {}).get("dataset_name")
    default_verbose_cfg = current_config.get("inference_settings", {}).get("verbose", False)
    default_use_num_cfg = current_config.get("datasets", {}).get("use_num_test_data", False)
    default_num_test_cfg = current_config.get("datasets", {}).get("num_test_data", 10)
    default_split_gqa_cfg = current_config.get("datasets", {}).get("gqa_dataset_split", "val")
    default_split_vqa_cfg = current_config.get("datasets", {}).get("vqa_v2_dataset_split", "rest-val")
    default_random_seed_cfg = current_config.get("datasets", {}).get("random_seed", 0)

    parser = argparse.ArgumentParser(description="AutoGen VQA Pipeline Runner")
    parser.add_argument("--dataset_name", type=str, default=default_dataset_name_cfg,
                        choices=['vqa-v2', 'gqa'], help="Dataset name.")
    parser.add_argument("--dataset_split", type=str, default=None, 
                        help="Specific dataset split. Overrides config.yaml if provided.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=default_verbose_cfg,
                        help="Enable/disable verbose logging for console (app DEBUG vs INFO).")
    parser.add_argument("--use_num_test_data", action=argparse.BooleanOptionalAction, default=default_use_num_cfg,
                        help="Use 'num_test_data' for subset size instead of 'percent_test'.")
    parser.add_argument("--num_test_data", type=int, default=default_num_test_cfg,
                        help="Number of test items if --use_num_test_data is active.")
    parser.add_argument("--random_seed", type=int, default=default_random_seed_cfg,
                        help="Random seed for subset selection and other stochastic processes.")
    cli_args = parser.parse_args()

    if cli_args.dataset_split is None:
        cli_args.dataset_split = default_split_gqa_cfg if cli_args.dataset_name == 'gqa' else default_split_vqa_cfg
    
    logger, current_run_output_dir = setup_logging(cli_args, current_config)

    logger.info(f"Torch version: {torch.__version__}")
    try:
        import torchvision
        logger.info(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        logger.warning(f"{Colors.YELLOW}Torchvision not found or importable.{Colors.ENDC}")

    if not isinstance(current_config, dict) or not current_config:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Configuration loaded incorrectly or is empty. Check config.yaml and config_loader.py.{Colors.ENDC}")
        return
    required_config_sections = ["datasets", "inference_settings", "vllm_details"]
    for section in required_config_sections:
        if section not in current_config:
            logger.critical(f"{Colors.RED}CRITICAL ERROR: '{section}' section missing in loaded config.yaml.{Colors.ENDC}")
            return

    dataset_s_config = current_config["datasets"]
    inference_s_config = current_config["inference_settings"]
    vllm_details_conf = current_config["vllm_details"]

    if not cli_args.dataset_name:
         logger.critical(f"{Colors.RED}CRITICAL ERROR: 'dataset_name' key is missing or empty in config.yaml or CLI args.{Colors.ENDC}")
         return
    if cli_args.dataset_name not in ['vqa-v2', 'gqa']:
         logger.critical(f"{Colors.RED}CRITICAL ERROR: Invalid 'dataset_name' ('{cli_args.dataset_name}'). Must be 'vqa-v2' or 'gqa'.{Colors.ENDC}")
         return

    effective_dataset_name = cli_args.dataset_name
    effective_verbose = cli_args.verbose 
    effective_use_num = cli_args.use_num_test_data
    effective_num_test = cli_args.num_test_data
    effective_random_seed = cli_args.random_seed
    effective_split_name = cli_args.dataset_split

    dataset_s_config["dataset_name"] = effective_dataset_name
    inference_s_config["verbose"] = effective_verbose
    dataset_s_config["use_num_test_data"] = effective_use_num
    dataset_s_config["num_test_data"] = effective_num_test
    dataset_s_config["random_seed"] = effective_random_seed
    if effective_dataset_name == 'gqa':
        dataset_s_config["gqa_dataset_split"] = effective_split_name
    elif effective_dataset_name == 'vqa-v2':
        dataset_s_config["vqa_v2_dataset_split"] = effective_split_name
    
    logger.info("\n--- Effective Configuration for this Run ---")
    logger.info(f"Dataset: {effective_dataset_name}, Split: {effective_split_name}")
    logger.info(f"Verbose Console Logging (app DEBUG level): {effective_verbose}")
    logger.info(f"Random Seed: {effective_random_seed}")
    if effective_use_num:
        logger.info(f"Using Fixed Number of Test Data: {effective_num_test}")
    else:
        percent_to_display = dataset_s_config.get('percent_test', 1.0) * 100
        logger.info(f"Using Percentage of Test Data: {percent_to_display:.1f}%")
    logger.info(f"VLM for AutoGen: {vllm_details_conf.get('vlm_model_name', 'N/A')}")
    logger.info(f"LLM for AutoGen: {vllm_details_conf.get('llm_model_name', 'N/A')}")
    logger.info(f"vLLM Base URL (or OpenAI API): {vllm_details_conf.get('base_url', vllm_details_conf.get('openai_base_url', 'N/A'))}")
    logger.info(f"Run-specific output directory: {current_run_output_dir}")
    logger.info("----------------------------------------\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Script recognized device: {device}')
    if torch.cuda.is_available():
        logger.info(f'PyTorch CUDA version: {torch.version.cuda}')
        logger.info(f'Torch.distributed.is_available: {torch.distributed.is_available()}')
        logger.info(f'Using {torch.cuda.device_count()} GPU(s). Name: {torch.cuda.get_device_name(0)}')

    torch.manual_seed(effective_random_seed)
    logger.info(f"PyTorch random seed set to: {effective_random_seed}")

    logger.info(f"Loading dataset: {effective_dataset_name}, split: {effective_split_name}...")
    try:
        dataset_class = GQADataset if effective_dataset_name == 'gqa' else VQAv2Dataset
        full_dataset = dataset_class(current_config, transform=None)
    except FileNotFoundError as e_data:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: FileNotFoundError during dataset loading: {e_data}. Check paths in config.yaml.{Colors.ENDC}", exc_info=True)
        return
    except Exception as e_load:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Failed to instantiate Dataset for '{effective_dataset_name}'. Check dataloader.py and config.{Colors.ENDC} Error: {e_load}", exc_info=True)
        return

    if not full_dataset or len(full_dataset) == 0:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Dataset {effective_dataset_name} (split: {effective_split_name}) is empty. Exiting.{Colors.ENDC}")
        return
    logger.info(f"Full dataset loaded with {len(full_dataset)} items.")

    if effective_use_num:
        num_to_use = int(effective_num_test)
        if num_to_use <= 0:
            logger.warning(f"{Colors.YELLOW}Warning: 'num_test_data' is {num_to_use}. Will use all available items ({len(full_dataset)}) if positive, or 1 if dataset has items.{Colors.ENDC}")
            actual_num_to_use = len(full_dataset) if len(full_dataset) > 0 else 0
            if actual_num_to_use == 0 and len(full_dataset) > 0 : actual_num_to_use = 1
        else:
            actual_num_to_use = min(num_to_use, len(full_dataset))
            if actual_num_to_use < num_to_use :
                logger.warning(f"{Colors.YELLOW}Warning: Requested num_test_data ({num_to_use}) > available ({len(full_dataset)}). Using {actual_num_to_use}.{Colors.ENDC}")
    else:
        percent_test = float(dataset_s_config.get("percent_test", 1.0))
        if not (0.0 < percent_test <= 1.0):
            logger.warning(f"{Colors.YELLOW}Warning: 'percent_test' ({percent_test*100:.1f}%) is outside (0, 100%]. Adjusting to 100%.{Colors.ENDC}")
            percent_test = 1.0
        actual_num_to_use = int(percent_test * len(full_dataset))

    if actual_num_to_use == 0 :
        if len(full_dataset) > 0:
            logger.warning(f"{Colors.YELLOW}Warning: Configuration for subset size results in 0 items. Processing 1 item instead to verify pipeline.{Colors.ENDC}")
            actual_num_to_use = 1
        else:
            logger.error(f"{Colors.RED}Error: Dataset is empty and subset selection results in 0 items. Exiting.{Colors.ENDC}")
            return

    logger.info(f"Selecting {actual_num_to_use} items randomly for this run.")
    indices = torch.randperm(len(full_dataset))[:actual_num_to_use]
    test_subset = Subset(full_dataset, indices.tolist())

    if len(test_subset) == 0:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Resulting data subset has 0 items. This should not happen with prior checks. Exiting.{Colors.ENDC}")
        return

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    logger.info(f"Created DataLoader with {len(test_subset)} samples (num batches: {len(test_loader)}).")

    grader_instance = Grader()
    run_config_summary = {
        key: vars(cli_args).get(key, dataset_s_config.get(key, inference_s_config.get(key)))
        for key in ["dataset_name", "dataset_split", "verbose", "use_num_test_data", "num_test_data", "random_seed"]
    }
    run_config_summary.update({
        "actual_items_in_loader": len(test_loader),
        "vlm_model": vllm_details_conf.get('vlm_model_name'),
        "llm_model": vllm_details_conf.get('llm_model_name'),
        "percent_test_configured": dataset_s_config.get('percent_test') if not effective_use_num else None
    })
    all_results_data = {
        "run_configuration": run_config_summary,
        "results_by_question_id": {}
    }
    
    default_json_filename = "autogen_vqa_results.json" # Or .jsonl if preferred as default
    config_json_filename = inference_s_config.get("output_response_filename", default_json_filename)
    base_json_filename = os.path.basename(config_json_filename if config_json_filename else default_json_filename)
    output_filename = os.path.join(current_run_output_dir, base_json_filename)

    if inference_s_config.get("save_output_response", True):
        logger.info(f"Output JSON results will be saved to: {output_filename}")
    
    logger.info("\nStarting AutoGen VQA processing loop...")
    processed_successfully_count = 0
    error_in_pipeline_count = 0
    image_not_found_count = 0

    def create_error_result_dict(qid, img_path, q_text, target_ans, error_message_str, error_type="UnknownError"):
        # Added: Include question_type in error dict
        q_type = get_question_type(q_text)
        return {
            "question_id": qid, "image_path": img_path, "question": q_text, "question_type": q_type,
            "target_answer": target_ans, "initial_answer": "", "final_answer": "[Process Failed]",
            "match_baseline_failed": False, "is_numeric_reattempt": False, "analysis_output": "",
            "object_attributes_queried": "", "reattempt_answer": "", "grades": [], "majority_vote": "Error",
            "processing_time_seconds": 0.0, "error": f"{error_type}: {error_message_str}"
        }

    for i, data_batch in enumerate(test_loader):
        image_path = data_batch['image_path'][0]
        question_text = data_batch['question'][0]
        question_id_tensor = data_batch['question_id'][0]
        target_answer_text = data_batch['answer'][0]
        current_qid = str(question_id_tensor.item()) if hasattr(question_id_tensor, 'item') else str(question_id_tensor)

        # Added: Get question type
        current_question_type = get_question_type(question_text)

        is_print_turn = (i + 1) % inference_s_config.get("print_every", 10) == 0 or i == len(test_loader) - 1
        if effective_verbose or is_print_turn:
            logger.info(f"[Main Loop] Processing item {i + 1}/{len(test_loader)} (QID: {current_qid}, Type: {current_question_type})")
        else:
            logger.debug(f"[Main Loop] Processing item {i + 1}/{len(test_loader)} (QID: {current_qid}, Type: {current_question_type})")

        if not os.path.exists(image_path):
            logger.error(f"{Colors.RED}ERROR:{Colors.ENDC} Image not found at '{image_path}' for QID {current_qid}. Skipping.")
            # create_error_result_dict now includes question_type
            error_data = create_error_result_dict(current_qid, image_path, question_text, target_answer_text, "Image file not found.", "ImageFileNotFound")
            all_results_data["results_by_question_id"][current_qid] = error_data
            image_not_found_count += 1
            continue

        try:
            pipeline_result_data = await run_vqa_pipeline(
                image_path=image_path,
                question=question_text,
                question_id=current_qid,
                target_answer=target_answer_text
            )
            # Added: Store question type with results
            pipeline_result_data["question_type"] = current_question_type

            grades = pipeline_result_data.get("grades", [])
            match_failed = pipeline_result_data.get("match_baseline_failed", False)

            if grades:
                majority_vote_str = grader_instance.accumulate_grades(current_config, grades, match_failed)
                pipeline_result_data["majority_vote"] = majority_vote_str
            elif "error" not in pipeline_result_data:
                pipeline_result_data["majority_vote"] = "Grading Not Applicable or No Grades"

            all_results_data["results_by_question_id"][current_qid] = pipeline_result_data
            if "error" in pipeline_result_data and pipeline_result_data["error"]:
                error_in_pipeline_count +=1
                logger.warning(f"Pipeline returned an error for QID {current_qid}: {pipeline_result_data['error']}")
            else:
                processed_successfully_count += 1
                logger.debug(f"Successfully processed QID {current_qid} through pipeline.")

        except Exception as e_pipeline:
            logger.error(f"{Colors.RED}CRITICAL ERROR{Colors.ENDC} during run_vqa_pipeline for QID {current_qid}: {e_pipeline}", exc_info=True)
            # create_error_result_dict now includes question_type
            error_data = create_error_result_dict(current_qid, image_path, question_text, target_answer_text, str(e_pipeline), "PipelineRuntimeError")
            all_results_data["results_by_question_id"][current_qid] = error_data
            error_in_pipeline_count += 1

    logger.info("\n--- AutoGen VQA Processing Finished ---")
    logger.info(f"Total items attempted (from DataLoader): {len(test_loader)}")
    logger.info(f"Successfully processed by pipeline: {Colors.GREEN}{processed_successfully_count}{Colors.ENDC}")
    if image_not_found_count > 0:
        logger.warning(f"Skipped due to image not found: {Colors.YELLOW}{image_not_found_count}{Colors.ENDC}")
    if error_in_pipeline_count > 0:
        logger.error(f"Errors during VQA pipeline processing: {Colors.RED}{error_in_pipeline_count}{Colors.ENDC}")

    # --- Overall Metrics Calculation ---
    if processed_successfully_count > 0 or error_in_pipeline_count > 0 or image_not_found_count > 0 :
        logger.info("\nCalculating final accuracy...")
        try:
            baseline_accuracy, final_accuracy, stats = grader_instance.average_score()
            logger.info(f"Final Accuracy (Overall): {Colors.GREEN}{final_accuracy * 100:.2f}%{Colors.ENDC}")
            logger.info(f"Baseline Accuracy (Initial VLM): {baseline_accuracy * 100:.2f}%")
            logger.info(f"Detailed Overall Stats from Grader: {stats}")

            all_results_data['overall_metrics'] = {
                'baseline_accuracy': baseline_accuracy,
                'final_accuracy': final_accuracy,
                'detailed_grader_stats': stats, # Renamed for clarity
                'total_processed_successfully': processed_successfully_count,
                'total_image_not_found': image_not_found_count,
                'total_pipeline_errors': error_in_pipeline_count,
                'total_attempted_in_loader': len(test_loader)
            }
        except Exception as e_final_score:
            logger.error(f"{Colors.RED}Error during final accuracy calculation: {e_final_score}{Colors.ENDC}", exc_info=True)
            all_results_data['overall_metrics'] = {"error_calculating_scores": str(e_final_score)}
    else:
        logger.info("\nNo items were processed by the VQA pipeline or encountered errors; skipping final accuracy calculation.")
        all_results_data['overall_metrics'] = {"message": "No pipeline results to score or items attempted."}

    # Added: Question Type Statistics Aggregation
    question_type_counts = {"total": 0, "yes/no": 0, "number": 0, "other": 0}
    question_type_correct_counts = {"yes/no": 0, "number": 0, "other": 0}

    for qid_key, result_item in all_results_data["results_by_question_id"].items():
        q_type = result_item.get("question_type", "other") # Default to "other" if not found
        
        question_type_counts["total"] += 1
        if q_type in question_type_counts: # Ensure q_type is one of the predefined keys
            question_type_counts[q_type] += 1
        else: # Should ideally not happen if get_question_type is comprehensive
            question_type_counts["other"] += 1 
            logger.warning(f"Encountered an unexpected question type '{q_type}' for QID {qid_key}. Categorizing as 'other'.")


        # Check for correctness based on 'majority_vote'
        # The user's log shows: "Majority vote is [Correct] with a score of 1/1"
        # And pipeline_result_data["majority_vote"] = majority_vote_str
        # So majority_vote_str likely contains "[Correct]" or "[Incorrect]"
        is_correct = False
        majority_vote_value = result_item.get("majority_vote")
        if majority_vote_value and isinstance(majority_vote_value, str):
            if "[correct]" in majority_vote_value.lower():
                is_correct = True
        
        if is_correct:
            if q_type in question_type_correct_counts: # Ensure q_type is one of the predefined keys
                question_type_correct_counts[q_type] += 1
            # No else needed here, as we only count correct for defined types

    logger.info("\n--- Question Type Statistics ---")
    logger.info(f"Total questions analyzed for type stats: {question_type_counts['total']}")
    
    accuracies_by_type_report = {}
    for q_type_key in ["yes/no", "number", "other"]:
        total_for_type = question_type_counts[q_type_key]
        correct_for_type = question_type_correct_counts[q_type_key]
        logger.info(f"Type '{q_type_key.capitalize()}': Count={total_for_type}")
        if total_for_type > 0:
            accuracy_val = (correct_for_type / total_for_type) * 100
            logger.info(f"  Accuracy for '{q_type_key.capitalize()}': {accuracy_val:.2f}% ({correct_for_type}/{total_for_type})")
            accuracies_by_type_report[q_type_key] = {
                "count": total_for_type,
                "correct": correct_for_type,
                "accuracy_percent": accuracy_val
            }
        else:
            logger.info(f"  Accuracy for '{q_type_key.capitalize()}': N/A (0 questions)")
            accuracies_by_type_report[q_type_key] = {
                "count": total_for_type,
                "correct": correct_for_type,
                "accuracy_percent": "N/A"
            }
    
    if 'overall_metrics' not in all_results_data: # Should exist from accuracy calculation part
        all_results_data['overall_metrics'] = {}
    all_results_data['overall_metrics']['question_type_summary'] = {
        "counts": question_type_counts, # total, yes/no, number, other
        "accuracies_report": accuracies_by_type_report # detailed report for each type
    }
    # End Added Section

    if inference_s_config.get("save_output_response", True):
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Final results, configuration, and metrics saved to: {output_filename}")
        except Exception as e_save:
            logger.error(f"{Colors.RED}ERROR saving final JSON results to {output_filename}: {e_save}{Colors.ENDC}", exc_info=True)
    else:
        logger.info("Skipping saving final results file as per 'save_output_response' configuration.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}Warning: CUDA is not available. Running on CPU will be significantly slower.{Colors.ENDC}")
    try:
        asyncio.run(main_logic_entry_point())
    except FileNotFoundError as e:
        print(f"\n{Colors.RED}CRITICAL FileNotFoundError in main execution{Colors.ENDC}: {e}")
        traceback.print_exc() 
    except ImportError as e:
        print(f"\n{Colors.RED}CRITICAL ImportError in main execution{Colors.ENDC}: {e}")
        traceback.print_exc() 
    except KeyboardInterrupt: # Added: Graceful exit on Ctrl+C
        print(f"\n{Colors.YELLOW}Execution interrupted by user (Ctrl+C).{Colors.ENDC}")
    except Exception as e_main_global:
        print(f"\n{Colors.RED}CRITICAL UNEXPECTED ERROR in main execution{Colors.ENDC}: {e_main_global}")
        traceback.print_exc() 
    finally:
        print("\nMain script execution finished.")