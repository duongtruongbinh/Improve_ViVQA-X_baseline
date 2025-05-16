# main.py
import asyncio
import argparse
import os
import torch
from torch.utils.data import Subset, DataLoader
import json
import traceback

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

async def main_logic_entry_point():
    current_config = app_config

    print(f"Torch version: {torch.__version__}")
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print(f"{Colors.YELLOW}Torchvision not found or importable.{Colors.ENDC}")

    if not isinstance(current_config, dict) or not current_config:
        print(f"{Colors.RED}CRITICAL ERROR: Configuration loaded incorrectly or is empty. Check config.yaml and config_loader.py.{Colors.ENDC}")
        return
    required_config_sections = ["datasets", "inference_settings", "vllm_details"]
    for section in required_config_sections:
        if section not in current_config:
            print(f"{Colors.RED}CRITICAL ERROR: '{section}' section missing in loaded config.yaml.{Colors.ENDC}")
            return

    dataset_s_config = current_config["datasets"]
    inference_s_config = current_config["inference_settings"]
    vllm_details_conf = current_config["vllm_details"]

    default_dataset_name = dataset_s_config.get("dataset_name")
    if not default_dataset_name:
        print(f"{Colors.RED}CRITICAL ERROR: 'dataset_name' key is missing or empty within 'datasets' in config.yaml.{Colors.ENDC}")
        return
    if default_dataset_name not in ['vqa-v2', 'gqa']:
        print(f"{Colors.RED}CRITICAL ERROR: Invalid 'dataset_name' ('{default_dataset_name}') in config.yaml. Must be 'vqa-v2' or 'gqa'.{Colors.ENDC}")
        return

    default_verbose = inference_s_config.get("verbose", False)
    default_use_num = dataset_s_config.get("use_num_test_data", False)
    default_num_test = dataset_s_config.get("num_test_data", 10)
    default_split_gqa = dataset_s_config.get("gqa_dataset_split", "val")
    default_split_vqa = dataset_s_config.get("vqa_v2_dataset_split", "rest-val")
    default_random_seed = dataset_s_config.get("random_seed", 0)

    parser = argparse.ArgumentParser(description="AutoGen VQA Pipeline Runner")
    parser.add_argument("--dataset_name", type=str, default=default_dataset_name,
                        choices=['vqa-v2', 'gqa'], help="Dataset name.")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="Specific dataset split. Overrides config.yaml if provided.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=default_verbose,
                        help="Enable/disable verbose logging.")
    parser.add_argument("--use_num_test_data", action=argparse.BooleanOptionalAction, default=default_use_num,
                        help="Use 'num_test_data' for subset size instead of 'percent_test'.")
    parser.add_argument("--num_test_data", type=int, default=default_num_test,
                        help="Number of test items if --use_num_test_data is active.")
    parser.add_argument("--random_seed", type=int, default=default_random_seed,
                        help="Random seed for subset selection and other stochastic processes.")

    cli_args = parser.parse_args()

    effective_dataset_name = cli_args.dataset_name
    effective_verbose = cli_args.verbose
    effective_use_num = cli_args.use_num_test_data
    effective_num_test = cli_args.num_test_data
    effective_random_seed = cli_args.random_seed

    effective_split_name = cli_args.dataset_split
    if not effective_split_name:
        effective_split_name = default_split_gqa if effective_dataset_name == 'gqa' else default_split_vqa

    dataset_s_config["dataset_name"] = effective_dataset_name
    inference_s_config["verbose"] = effective_verbose
    dataset_s_config["use_num_test_data"] = effective_use_num
    dataset_s_config["num_test_data"] = effective_num_test
    dataset_s_config["random_seed"] = effective_random_seed
    if effective_dataset_name == 'gqa':
        dataset_s_config["gqa_dataset_split"] = effective_split_name
    elif effective_dataset_name == 'vqa-v2':
        dataset_s_config["vqa_v2_dataset_split"] = effective_split_name

    print("\n--- Effective Configuration for this Run ---")
    print(f"Dataset: {effective_dataset_name}, Split: {effective_split_name}")
    print(f"Verbose Logging: {effective_verbose}")
    print(f"Random Seed: {effective_random_seed}")
    if effective_use_num:
        print(f"Using Fixed Number of Test Data: {effective_num_test}")
    else:
        percent_to_display = dataset_s_config.get('percent_test', 1.0) * 100
        print(f"Using Percentage of Test Data: {percent_to_display:.1f}%")
    print(f"VLM for AutoGen: {vllm_details_conf.get('vlm_model_name', 'N/A')}")
    print(f"LLM for AutoGen: {vllm_details_conf.get('llm_model_name', 'N/A')}")
    print(f"vLLM Base URL: {vllm_details_conf.get('base_url', 'N/A')}")
    print("----------------------------------------\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Script recognized device: {device}')
    if torch.cuda.is_available():
        print(f'PyTorch CUDA version: {torch.version.cuda}')
        print(f'Torch.distributed.is_available: {torch.distributed.is_available()}')
        print(f'Using {torch.cuda.device_count()} GPU(s). Name: {torch.cuda.get_device_name(0)}')

    torch.manual_seed(effective_random_seed)

    print(f"Loading dataset: {effective_dataset_name}, split: {effective_split_name}...")
    try:
        dataset_class = GQADataset if effective_dataset_name == 'gqa' else VQAv2Dataset
        full_dataset = dataset_class(current_config, transform=None)
    except FileNotFoundError as e_data:
        print(f"{Colors.RED}CRITICAL ERROR: FileNotFoundError during dataset loading: {e_data}. Check paths in config.yaml.{Colors.ENDC}")
        return
    except Exception as e_load:
        print(f"{Colors.RED}CRITICAL ERROR: Failed to instantiate Dataset for '{effective_dataset_name}'. Check dataloader.py and config.{Colors.ENDC}")
        print(f"Error details: {e_load}")
        traceback.print_exc()
        return

    if not full_dataset or len(full_dataset) == 0:
        print(f"{Colors.RED}CRITICAL ERROR: Dataset {effective_dataset_name} (split: {effective_split_name}) is empty. Exiting.{Colors.ENDC}")
        return
    print(f"Full dataset loaded with {len(full_dataset)} items.")

    if effective_use_num:
        num_to_use = int(effective_num_test)
        if num_to_use <= 0:
            print(f"{Colors.YELLOW}Warning: 'num_test_data' is {num_to_use}. Will use all available items ({len(full_dataset)}) if positive, or 1 if dataset has items.{Colors.ENDC}")
            actual_num_to_use = len(full_dataset) if len(full_dataset) > 0 else 0
            if actual_num_to_use == 0 and len(full_dataset) > 0 : actual_num_to_use = 1
        else:
            actual_num_to_use = min(num_to_use, len(full_dataset))
            if actual_num_to_use < num_to_use :
                print(f"{Colors.YELLOW}Warning: Requested num_test_data ({num_to_use}) > available ({len(full_dataset)}). Using {actual_num_to_use}.{Colors.ENDC}")
    else:
        percent_test = float(dataset_s_config.get("percent_test", 1.0))
        if not (0.0 < percent_test <= 1.0):
            print(f"{Colors.YELLOW}Warning: 'percent_test' ({percent_test*100:.1f}%) is outside (0, 100%]. Adjusting to 100%.{Colors.ENDC}")
            percent_test = 1.0
        actual_num_to_use = int(percent_test * len(full_dataset))

    if actual_num_to_use == 0 :
        if len(full_dataset) > 0:
            print(f"{Colors.YELLOW}Warning: Configuration for subset size results in 0 items. Processing 1 item instead to verify pipeline.{Colors.ENDC}")
            actual_num_to_use = 1
        else:
            print(f"{Colors.RED}Error: Dataset is empty and subset selection results in 0 items. Exiting.{Colors.ENDC}")
            return

    print(f"Selecting {actual_num_to_use} items randomly for this run.")
    test_subset_idx = torch.randperm(len(full_dataset))[:actual_num_to_use]
    test_subset = Subset(full_dataset, test_subset_idx.tolist())

    if len(test_subset) == 0:
        print(f"{Colors.RED}CRITICAL ERROR: Resulting data subset has 0 items. This should not happen with prior checks. Exiting.{Colors.ENDC}")
        return

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    print(f"Created DataLoader with {len(test_subset)} samples (num batches: {len(test_loader)}).")

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
    output_filename = inference_s_config.get("output_response_filename", "outputs/autogen_vqa_results.json")
    output_dir = os.path.dirname(output_filename)

    if inference_s_config.get("save_output_response", True) and output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output will be saved to directory: {output_dir}")
        except OSError as e:
            print(f"{Colors.YELLOW}Warning: Could not create output directory {output_dir}: {e}. Results might not be saved.{Colors.ENDC}")

    print("\nStarting AutoGen VQA processing loop...")
    processed_successfully_count = 0
    error_in_pipeline_count = 0
    image_not_found_count = 0

    def create_error_result_dict(qid, img_path, q_text, target_ans, error_message_str, error_type="UnknownError"):
        return {
            "question_id": qid, "image_path": img_path, "question": q_text,
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

        if effective_verbose or (i + 1) % inference_s_config.get("print_every", 10) == 0 or i == len(test_loader) - 1:
            print(f"[Main Loop] Processing item {i + 1}/{len(test_loader)} (QID: {current_qid})")

        if not os.path.exists(image_path):
            print(f"{Colors.RED}ERROR:{Colors.ENDC} Image not found at '{image_path}' for QID {current_qid}. Skipping.")
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
            else:
                processed_successfully_count += 1

        except Exception as e_pipeline:
            print(f"{Colors.RED}CRITICAL ERROR{Colors.ENDC} during run_vqa_pipeline for QID {current_qid}: {e_pipeline}")
            traceback.print_exc()
            error_data = create_error_result_dict(current_qid, image_path, question_text, target_answer_text, str(e_pipeline), "PipelineRuntimeError")
            all_results_data["results_by_question_id"][current_qid] = error_data
            error_in_pipeline_count += 1

    print("\n--- AutoGen VQA Processing Finished ---")
    print(f"Total items attempted (from DataLoader): {len(test_loader)}")
    print(f"Successfully processed by pipeline: {Colors.GREEN}{processed_successfully_count}{Colors.ENDC}")
    if image_not_found_count > 0:
        print(f"Skipped due to image not found: {Colors.YELLOW}{image_not_found_count}{Colors.ENDC}")
    if error_in_pipeline_count > 0:
        print(f"Errors during VQA pipeline processing: {Colors.RED}{error_in_pipeline_count}{Colors.ENDC}")

    if processed_successfully_count > 0 or error_in_pipeline_count > 0:
        print("\nCalculating final accuracy and saving results...")
        try:
            baseline_accuracy, final_accuracy, stats = grader_instance.average_score()
            print(f"Final Accuracy (Overall): {Colors.GREEN}{final_accuracy * 100:.2f}%{Colors.ENDC}")
            print(f"Baseline Accuracy (Initial VLM): {baseline_accuracy * 100:.2f}%")
            print(f"Detailed Stats: {stats}")

            all_results_data['overall_metrics'] = {
                'baseline_accuracy': baseline_accuracy,
                'final_accuracy': final_accuracy,
                'detailed_stats': stats,
                'total_processed_successfully': processed_successfully_count,
                'total_image_not_found': image_not_found_count,
                'total_pipeline_errors': error_in_pipeline_count,
                'total_attempted_in_loader': len(test_loader)
            }
        except Exception as e_final_score:
            print(f"{Colors.RED}Error during final accuracy calculation: {e_final_score}{Colors.ENDC}")
            traceback.print_exc()
            all_results_data['overall_metrics'] = {"error_calculating_scores": str(e_final_score)}
    else:
        print("\nNo items were processed by the VQA pipeline; skipping final accuracy calculation.")
        all_results_data['overall_metrics'] = {"message": "No pipeline results to score."}

    if inference_s_config.get("save_output_response", True):
        try:
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results_data, f, indent=4, ensure_ascii=False)
            print(f"Final results, configuration, and metrics saved to: {output_filename}")
        except Exception as e_save:
            print(f"{Colors.RED}ERROR saving final JSON results to {output_filename}: {e_save}{Colors.ENDC}")
            traceback.print_exc()
    else:
        print("Skipping saving final results file as per 'save_output_response' configuration.")

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
    except Exception as e_main_global:
        print(f"\n{Colors.RED}CRITICAL UNEXPECTED ERROR in main execution{Colors.ENDC}: {e_main_global}")
        traceback.print_exc()
    finally:
        print("\nMain script execution finished.")