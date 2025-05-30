# main.py
import asyncio
import argparse
import os
import torch
from torch.utils.data import Subset, DataLoader
import json
import traceback
import logging
import sys
import time
from datetime import datetime, timedelta

try:
    from .config_loader import app_config
    from .workflows.specialized_flow import run_vqa_pipeline
    from .workflows.reflection_flow import run_simple_vqa_pipeline
    from .workflows.debate_flow import run_simplified_debate_vqa_pipeline
    from .dataloader import VQAv2Dataset, GQADataset
    from .utils import (
        Colors,
        setup_logging,
        get_question_type,
        create_error_result_dict,
        set_seed,
        Grader,
        extract_error_analysis_data,
        save_error_analysis
    )
    from .evaluation import perform_direct_accuracy_check
    from .agents.reflection_agents import VQAGeneratorAgent
    from .agents.debate_agents import VQADebateSolverAgent
    from .vllm_clients import vlm_client_vllm, llm_client_vllm
    from .vllm_clients import llm_config_vlm, llm_config_llm
except ImportError as e:
    print(f"Import Error in main.py: {e}. Please ensure all required .py files "
          "(config_loader, workflows, dataloader, utils, evaluation etc.) "
          "are in the correct path and modules exist.")
    exit(1)

async def main_logic_entry_point():
    script_start_time = time.time()
    current_config = app_config

    default_dataset_name_cfg = current_config.get("datasets", {}).get("dataset_name")
    default_verbose_cfg = current_config.get("inference_settings", {}).get("verbose", False)
    default_use_num_cfg = current_config.get("datasets", {}).get("use_num_test_data", False)
    default_num_test_cfg = current_config.get("datasets", {}).get("num_test_data", 10)
    default_split_gqa_cfg = current_config.get("datasets", {}).get("gqa_dataset_split", "val")
    default_split_vqa_cfg = current_config.get("datasets", {}).get("vqa_v2_dataset_split", "rest-val")
    default_random_seed_cfg = current_config.get("datasets", {}).get("random_seed", 0)
    default_vqa_flow_type_cfg = current_config.get("inference_settings", {}).get("vqa_flow_type", "specialized")

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
    parser.add_argument("--vqa_flow_type", type=str, default=default_vqa_flow_type_cfg,
                        choices=['specialized', 'reflection', 'debate'],
                        help="Type of VQA flow to run. Overrides config.yaml if provided.")
    cli_args = parser.parse_args()

    if cli_args.dataset_split is None:
        cli_args.dataset_split = default_split_gqa_cfg if cli_args.dataset_name == 'gqa' else default_split_vqa_cfg

    current_config["datasets"]["dataset_name"] = cli_args.dataset_name
    current_config["inference_settings"]["verbose"] = cli_args.verbose
    current_config["datasets"]["use_num_test_data"] = cli_args.use_num_test_data
    current_config["datasets"]["num_test_data"] = cli_args.num_test_data
    current_config["datasets"]["random_seed"] = cli_args.random_seed
    current_config["inference_settings"]["vqa_flow_type"] = cli_args.vqa_flow_type

    if cli_args.dataset_name == 'gqa':
        current_config["datasets"]["gqa_dataset_split"] = cli_args.dataset_split
    elif cli_args.dataset_name == 'vqa-v2':
        current_config["datasets"]["vqa_v2_dataset_split"] = cli_args.dataset_split

    logger, current_run_output_dir = setup_logging(cli_args, current_config)

    dataset_s_config = current_config["datasets"]
    inference_s_config = current_config["inference_settings"]
    vlm_details_conf = current_config.get("vlm_details", {})

    effective_dataset_name = dataset_s_config["dataset_name"]
    effective_verbose = inference_s_config["verbose"]
    effective_use_num = dataset_s_config["use_num_test_data"]
    effective_num_test = dataset_s_config["num_test_data"]
    effective_random_seed = dataset_s_config["random_seed"]
    effective_split_name = dataset_s_config.get("gqa_dataset_split") if effective_dataset_name == 'gqa' else dataset_s_config.get("vqa_v2_dataset_split")
    effective_vqa_flow_type = cli_args.vqa_flow_type.lower()

    logger.info(f"Torch version: {torch.__version__}")
    try:
        import torchvision
        logger.info(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        logger.warning(f"{Colors.YELLOW}Torchvision not found or importable.{Colors.ENDC}")

    if not isinstance(current_config, dict) or not current_config:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Configuration loaded incorrectly or is empty. Check config.yaml and config_loader.py.{Colors.ENDC}")
        return
    required_config_sections = ["datasets", "inference_settings"]
    if effective_vqa_flow_type != "simple_direct":
        required_config_sections.append("vlm_details")

    for section in required_config_sections:
        if section not in current_config:
            logger.critical(f"{Colors.RED}CRITICAL ERROR: '{section}' section missing in loaded config.yaml.{Colors.ENDC}")
            return

    if not effective_dataset_name:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: 'dataset_name' key is missing or empty.{Colors.ENDC}")
        return
    if effective_dataset_name not in ['vqa-v2', 'gqa']:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Invalid 'dataset_name' ('{effective_dataset_name}'). Must be 'vqa-v2' or 'gqa'.{Colors.ENDC}")
        return

    logger.info("--- Effective Configuration for this Run ---")
    logger.info(f"Dataset: {effective_dataset_name}, Split: {effective_split_name}")
    logger.info(f"VQA Flow Type: {effective_vqa_flow_type}")
    logger.info(f"Verbose Console Logging (app DEBUG level): {effective_verbose}")
    logger.info(f"Random Seed: {effective_random_seed}")
    if effective_use_num:
        logger.info(f"Using Fixed Number of Test Data: {effective_num_test}")
    else:
        percent_to_display = dataset_s_config.get('percent_test', 1.0) * 100
        logger.info(f"Using Percentage of Test Data: {percent_to_display:.1f}%")

    if effective_vqa_flow_type == "simple_direct":
        logger.warning(f"{Colors.YELLOW}Warning: 'simple_direct' flow type is not implemented. Using 'specialized' flow instead.{Colors.ENDC}")
        effective_vqa_flow_type = "specialized"
    else:
        logger.info(f"VLM for AutoGen: {vlm_details_conf.get('vlm_model_name', 'N/A')}")
        logger.info(f"LLM for AutoGen: {vlm_details_conf.get('llm_model_name', 'N/A')}")
        base_url_autogen = vlm_details_conf.get('base_url', vlm_details_conf.get('openai_base_url', 'N/A'))
        logger.info(f"Base URL (or OpenAI API for AutoGen): {base_url_autogen}")

    logger.info(f"Run-specific output directory: {current_run_output_dir}")
    logger.info("----------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Script recognized device: {device}')
    if torch.cuda.is_available():
        logger.info(f'PyTorch CUDA version: {torch.version.cuda}')
        logger.info(f'Torch.distributed.is_available: {torch.distributed.is_available()}')
        gpu_count = torch.cuda.device_count()
        logger.info(f'Using {gpu_count} GPU(s). Name: {torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"}')

    set_seed(effective_random_seed)
    logger.info(f"Global random seeds set to: {effective_random_seed}")

    dataset_load_start = time.time()
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
            actual_num_to_use = len(full_dataset) if len(full_dataset) > 0 else 0
            if actual_num_to_use == 0 and len(full_dataset) > 0 : actual_num_to_use = 1
            logger.warning(f"{Colors.YELLOW}Warning: 'num_test_data' is {num_to_use}. Using {actual_num_to_use} items.{Colors.ENDC}")
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
    generator = torch.Generator().manual_seed(effective_random_seed)
    indices = torch.randperm(len(full_dataset), generator=generator)[:actual_num_to_use]
    test_subset = Subset(full_dataset, indices.tolist())

    if len(test_subset) == 0:
        logger.critical(f"{Colors.RED}CRITICAL ERROR: Resulting data subset has 0 items. This should not happen with prior checks. Exiting.{Colors.ENDC}")
        return

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    dataset_load_end = time.time()
    dataset_load_time = dataset_load_end - dataset_load_start
    logger.info(f"Created DataLoader with {len(test_subset)} samples (num batches: {len(test_loader)}). Dataset loading took: {dataset_load_time:.2f} seconds")

    grader_instance = Grader() if effective_vqa_flow_type == "specialized" else None

    run_config_summary = {
        key: current_config["datasets"].get(key, current_config["inference_settings"].get(key, cli_args.__dict__.get(key)))
        for key in ["dataset_name", "verbose", "use_num_test_data", "num_test_data", "random_seed"]
    }
    run_config_summary["dataset_split"] = effective_split_name
    run_config_summary["vqa_flow_type"] = effective_vqa_flow_type
    run_config_summary.update({
        "actual_items_in_loader": len(test_loader),
        "vlm_model_autogen": vlm_details_conf.get('vlm_model_name') if effective_vqa_flow_type != "simple_direct" else "N/A",
        "llm_model_autogen": vlm_details_conf.get('llm_model_name') if effective_vqa_flow_type != "simple_direct" else "N/A",
        "direct_vlm_model_path": vlm_details_conf.get('simple_direct_model_path') if effective_vqa_flow_type == "simple_direct" else "N/A",
        "percent_test_configured": dataset_s_config.get('percent_test') if not effective_use_num else None
    })
    all_results_data = {
        "run_configuration": run_config_summary,
        "results_by_question_id": {}
    }

    default_json_filename = "autogen_vqa_results.jsonl"
    config_json_filename = inference_s_config.get("output_response_filename", default_json_filename)
    base_json_filename = os.path.basename(config_json_filename if config_json_filename else default_json_filename)
    output_filename = os.path.join(current_run_output_dir, base_json_filename)

    if inference_s_config.get("save_output_response", True):
        logger.info(f"Output JSON results will be saved to: {output_filename}")

    logger.info(f"Starting AutoGen VQA processing loop with flow: {effective_vqa_flow_type}...")
    processed_successfully_count = 0
    error_in_pipeline_count = 0
    image_not_found_count = 0
    processed_debate_qids = set() # ADDED: To track QIDs processed by debate flow
    skipped_debate_due_to_duplicate_count = 0 # ADDED: Counter for skipped debate items

    processing_start_time = time.time()

    for i, data_batch in enumerate(test_loader):
        image_path = data_batch['image_path'][0]
        question_text = data_batch['question'][0]
        question_id_tensor = data_batch['question_id'][0]
        raw_target_answer = data_batch['answer'][0]
        current_qid = str(question_id_tensor.item()) if hasattr(question_id_tensor, 'item') else str(question_id_tensor)

        main_determined_q_type = get_question_type(question_text)

        if i < 10 and effective_verbose:
            from .utils import yes_no_starters_global as ynsg_debug, number_starters_global as nsg_debug
            q_lower_for_debug = str(question_text or "").lower().strip()
            classified_as_for_debug = "other"
            debug_match_info = "No match"
            for s_debug_γν in ynsg_debug:
                if q_lower_for_debug.startswith(s_debug_γν):
                    classified_as_for_debug = "yes/no"
                    debug_match_info = f"Matched yes/no: '{s_debug_γν}'"
                    break
            if classified_as_for_debug == "other":
                for s_debug_num in nsg_debug:
                    if q_lower_for_debug.startswith(s_debug_num):
                        classified_as_for_debug = "number"
                        debug_match_info = f"Matched number: '{s_debug_num}'"
                        break
            logger.info(f"DEBUG_Q_TYPE: QID={current_qid}")
            logger.info(f"DEBUG_Q_TYPE: Original Text='{question_text}'")
            logger.info(f"DEBUG_Q_TYPE: Lowered/Stripped Text='{q_lower_for_debug}'")
            logger.info(f"DEBUG_Q_TYPE: Inline Debug Matcher Result='{classified_as_for_debug}', Info='{debug_match_info}'")
            logger.info(f"DEBUG_Q_TYPE: main.py get_question_type Result='{main_determined_q_type}'")

        is_print_turn = (i + 1) % inference_s_config.get("print_every", 10) == 0 or i == len(test_loader) - 1
        log_prefix = f"[Main Loop - Flow: {effective_vqa_flow_type}]"
        if effective_verbose or is_print_turn:
            logger.info(f"{log_prefix} Processing item {i + 1}/{len(test_loader)} (QID: {current_qid}, Main Type Det: {main_determined_q_type})")
        else:
            logger.debug(f"{log_prefix} Processing item {i + 1}/{len(test_loader)} (QID: {current_qid}, Main Type Det: {main_determined_q_type})")

        pipeline_result_data = None # Initialize for current iteration
        item_processed_by_pipeline_this_iteration = False # Flag: default to False

        if not os.path.exists(image_path):
            logger.error(f"{Colors.RED}ERROR:{Colors.ENDC} Image not found at '{image_path}' for QID {current_qid}. Skipping.")
            error_data = create_error_result_dict(current_qid, image_path, question_text, raw_target_answer,
                                                  "Image file not found.", "ImageFileNotFound", effective_vqa_flow_type)
            all_results_data["results_by_question_id"][current_qid] = error_data
            image_not_found_count += 1
            continue

        try:
            if effective_vqa_flow_type == "specialized":
                logger.debug(f"Running SPECIALIZED VQA pipeline for QID {current_qid}...")
                pipeline_result_data = await run_vqa_pipeline(
                    image_path=image_path,
                    question=question_text,
                    question_id=current_qid,
                    target_answer=raw_target_answer
                )
                item_processed_by_pipeline_this_iteration = True
            elif effective_vqa_flow_type == "reflection":
                logger.debug(f"Running REFLECTION (simple) VQA pipeline for QID {current_qid}...")
                pipeline_result_data = await run_simple_vqa_pipeline(
                    image_path=image_path,
                    question=question_text,
                    question_id=current_qid,
                    target_answer=raw_target_answer,
                    question_type=main_determined_q_type,
                    logger_instance=logger
                )
                item_processed_by_pipeline_this_iteration = True
            elif effective_vqa_flow_type == "debate":
                if current_qid in processed_debate_qids:
                    logger.info(f"DEBATE flow: QID {current_qid} has already had run_simplified_debate_vqa_pipeline called. "
                                f"Skipping redundant execution for item index {i}. Using existing result for QID {current_qid}.")
                    skipped_debate_due_to_duplicate_count += 1
                    item_processed_by_pipeline_this_iteration = False # Pipeline not run for THIS specific item
                else:
                    logger.debug(f"Running DEBATE (simplified) VQA pipeline for QID {current_qid}...")
                    pipeline_result_data = await run_simplified_debate_vqa_pipeline(
                        image_path=image_path,
                        question=question_text,
                        question_id=current_qid,
                        target_answer=raw_target_answer,
                        question_type=main_determined_q_type,
                        logger_instance=logger
                    )
                    processed_debate_qids.add(current_qid) # Mark QID as having its pipeline call attempted
                item_processed_by_pipeline_this_iteration = True
            else: # Handles unknown flow type, preserves original continue behavior
                logger.error(f"Unknown vqa_flow_type: '{effective_vqa_flow_type}'. Skipping QID {current_qid}.")
                error_data = create_error_result_dict(current_qid, image_path, question_text, raw_target_answer,
                                                      f"Unknown vqa_flow_type: {effective_vqa_flow_type}", "ConfigurationError", effective_vqa_flow_type)
                all_results_data["results_by_question_id"][current_qid] = error_data
                error_in_pipeline_count += 1
                continue # Skip further processing for this item in this iteration

            if item_processed_by_pipeline_this_iteration:
                if pipeline_result_data is None:
                    raise ValueError(f"Pipeline function for flow '{effective_vqa_flow_type}' (QID {current_qid}) returned None unexpectedly.")

                pipeline_result_data["flow_type_used"] = effective_vqa_flow_type
                if "question_type" not in pipeline_result_data or not pipeline_result_data["question_type"]:
                    logger.warning(f"QID {current_qid}, Flow {effective_vqa_flow_type}: 'question_type' was not in pipeline_result_data or empty. Using main.py's determination: {main_determined_q_type}")
                    pipeline_result_data["question_type"] = main_determined_q_type
                elif pipeline_result_data["question_type"] != main_determined_q_type and effective_verbose:
                    logger.info(f"QID {current_qid}: main.py type='{main_determined_q_type}', flow '{effective_vqa_flow_type}' type='{pipeline_result_data['question_type']}'")

                if effective_vqa_flow_type == "specialized":
                    if grader_instance:
                        grades = pipeline_result_data.get("grades", [])
                        match_baseline_failed = pipeline_result_data.get("match_baseline_failed", False)
                        if grades:
                            majority_vote_str = grader_instance.accumulate_grades(current_config, grades, match_baseline_failed)
                            pipeline_result_data["majority_vote"] = majority_vote_str
                        elif "error" not in pipeline_result_data:
                            pipeline_result_data["majority_vote"] = "Grading Not Applicable or No Grades"
                    else:
                        pipeline_result_data["majority_vote"] = "Specialized (Grader not initialized)"
                else:
                    accuracy_check_res = pipeline_result_data.get("direct_accuracy_check", {})
                    pipeline_result_data["majority_vote"] = accuracy_check_res.get("notes", "[Accuracy Check Incomplete]")

                all_results_data["results_by_question_id"][current_qid] = pipeline_result_data

                if effective_verbose and "direct_accuracy_check" in pipeline_result_data:
                    logger.debug(f"QID {current_qid} [main.py]: direct_accuracy_check content: {pipeline_result_data['direct_accuracy_check']}")

                if "error" in pipeline_result_data and pipeline_result_data["error"]:
                    error_in_pipeline_count +=1
                    logger.warning(f"Pipeline ({effective_vqa_flow_type}) returned an error for QID {current_qid}: {pipeline_result_data['error']}")
                else:
                    processed_successfully_count += 1
                    logger.debug(f"Successfully processed QID {current_qid} through {effective_vqa_flow_type} pipeline.")
            # If item_processed_by_pipeline_this_iteration is False (e.g. skipped debate duplicate),
            # this item does not update all_results_data[current_qid], and does not affect success/error counts here.

        except Exception as e_pipeline:
            logger.error(f"{Colors.RED}CRITICAL ERROR{Colors.ENDC} during {effective_vqa_flow_type} pipeline for QID {current_qid}: {e_pipeline}", exc_info=True)
            error_data = create_error_result_dict(current_qid, image_path, question_text, raw_target_answer,
                                                  str(e_pipeline), "PipelineRuntimeError", effective_vqa_flow_type)
            all_results_data["results_by_question_id"][current_qid] = error_data
            error_in_pipeline_count += 1
            # If this exception occurred during the *first attempt* for a debate flow QID,
            # ensure it's marked in processed_debate_qids to prevent retries.
            if effective_vqa_flow_type == "debate" and current_qid not in processed_debate_qids:
                processed_debate_qids.add(current_qid)

    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time
    execution_time_str = str(timedelta(seconds=processing_time))

    logger.info("--- AutoGen VQA Processing Finished ---")
    logger.info(f"Processing time: {Colors.CYAN}{execution_time_str} ({processing_time:.2f} seconds){Colors.ENDC}")
    logger.info(f"Total items from DataLoader: {len(test_loader)}") # MODIFIED Log Message
    logger.info(f"Items processed successfully by pipeline: {Colors.GREEN}{processed_successfully_count}{Colors.ENDC}") # MODIFIED Log Message
    if skipped_debate_due_to_duplicate_count > 0: # ADDED Log Block
        logger.info(f"Items skipped for DEBATE flow (QID already processed): {Colors.YELLOW}{skipped_debate_due_to_duplicate_count}{Colors.ENDC}")
    if image_not_found_count > 0:
        logger.warning(f"Items skipped due to image not found: {Colors.YELLOW}{image_not_found_count}{Colors.ENDC}") # MODIFIED Log Message
    if error_in_pipeline_count > 0:
        logger.error(f"Items with errors (pipeline or config issues): {Colors.RED}{error_in_pipeline_count}{Colors.ENDC}") # MODIFIED Log Message


    question_type_counts = {"total": 0, "yes/no": 0, "number": 0, "other": 0}
    question_type_correct_counts = {"yes/no": 0, "number": 0, "other": 0}

    # The condition for processing stats should be fine as is, because all_results_data will contain
    # the results for unique QIDs (or the first encountered result if a QID was processed multiple times by non-debate flows).
    # For debate flow, it will correctly use the single processed result for that QID.
    if len(all_results_data["results_by_question_id"]) > 0 : # check if there are any results to analyze
        for qid_key, result_item in all_results_data["results_by_question_id"].items():
            if not isinstance(result_item, dict):
                logger.warning(f"Skipping malformed result item for QID {qid_key} in stats aggregation.")
                continue

            q_type = result_item.get("question_type", "other")
            item_flow_type = result_item.get("flow_type_used", "unknown")

            question_type_counts["total"] += 1
            if q_type in question_type_counts:
                question_type_counts[q_type] += 1
            else:
                question_type_counts["other"] += 1
                logger.warning(f"Encountered an unexpected question type '{q_type}' for QID {qid_key} (flow: {item_flow_type}). Categorizing as 'other'.")

            is_correct = False
            if result_item.get("error"):
                is_correct = False
            elif item_flow_type == "specialized":
                if grader_instance :
                    majority_vote_value = result_item.get("majority_vote")
                    if majority_vote_value and isinstance(majority_vote_value, str):
                        if "[correct]" in majority_vote_value.lower():
                            is_correct = True
                else:
                    accuracy_check = result_item.get("direct_accuracy_check", {})
                    is_correct = accuracy_check.get("is_correct", False)
            elif item_flow_type in ["reflection", "debate"]:
                accuracy_check = result_item.get("direct_accuracy_check", {})
                if q_type == "yes/no":
                    is_correct = accuracy_check.get("is_correct_strict", accuracy_check.get("is_correct", False))
                elif q_type == "number":
                    is_correct = accuracy_check.get("is_correct_loose_numeric", accuracy_check.get("is_correct", False))
                elif q_type == "other":
                    is_correct = accuracy_check.get("is_correct_f1", accuracy_check.get("is_correct", False))
                else:
                    is_correct = accuracy_check.get("is_correct", False)

            if is_correct:
                if q_type in question_type_correct_counts:
                    question_type_correct_counts[q_type] += 1
                else:
                    logger.warning(f"Correct answer for unhandled q_type '{q_type}' QID {qid_key} but q_type not in standard counters.")


        logger.info("Calculating overall statistics...")
        if 'overall_metrics' not in all_results_data: all_results_data['overall_metrics'] = {}

        total_analyzed_for_types = question_type_counts['total']
        total_correct_overall = sum(question_type_correct_counts.values())

        overall_accuracy_value = 0.0
        overall_accuracy_str = "N/A"
        if total_analyzed_for_types > 0:
            overall_accuracy_value = (total_correct_overall / total_analyzed_for_types) * 100
            overall_accuracy_str = f"{overall_accuracy_value:.2f}% ({total_correct_overall}/{total_analyzed_for_types})"

        logger.info(f"Overall Accuracy (based on type aggregation): {Colors.GREEN}{overall_accuracy_str}{Colors.ENDC}")

        all_results_data['overall_metrics'].update({
            'total_processed_successfully': processed_successfully_count,
            'total_skipped_debate_duplicates': skipped_debate_due_to_duplicate_count, # ADDED
            'total_image_not_found': image_not_found_count,
            'total_pipeline_errors': error_in_pipeline_count, # This includes config errors like unknown flow
            'total_attempted_in_loader': len(test_loader),
            'overall_accuracy_from_type_aggregation': overall_accuracy_value if total_analyzed_for_types > 0 else None,
            'overall_correct_count': total_correct_overall,
            'overall_total_typed_questions': total_analyzed_for_types, # This is count of unique QIDs analyzed for stats
            'processing_time_seconds': processing_time,
            'processing_time_formatted': execution_time_str,
            'dataset_load_time_seconds': dataset_load_time
        })
        logger.info(f"Total items processed/analyzed for stats (unique QIDs in results): {total_analyzed_for_types}")
        # The following log lines are now part of the summary block earlier.
        # logger.info(f"Total items with image not found: {image_not_found_count}")
        # logger.info(f"Total items with pipeline errors: {error_in_pipeline_count}")

    else: # This 'else' matches 'if len(all_results_data["results_by_question_id"]) > 0'
        logger.info("No items were processed or resulted in entries in 'results_by_question_id'; skipping statistics calculation.")
        all_results_data['overall_metrics'] = {
            "message": "No pipeline results to score or items attempted, or results dictionary is empty.",
            'total_processed_successfully': processed_successfully_count,
            'total_skipped_debate_duplicates': skipped_debate_due_to_duplicate_count,
            'total_image_not_found': image_not_found_count,
            'total_pipeline_errors': error_in_pipeline_count,
            'total_attempted_in_loader': len(test_loader),
            'processing_time_seconds': processing_time,
            'processing_time_formatted': execution_time_str,
            'dataset_load_time_seconds': dataset_load_time
        }


    logger.info("--- Question Type Statistics (Aggregated from all flow types) ---")
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
                "count": total_for_type, "correct": correct_for_type, "accuracy_percent": accuracy_val
            }
        else:
            logger.info(f"  Accuracy for '{q_type_key.capitalize()}': N/A (0 questions)")
            accuracies_by_type_report[q_type_key] = {
                "count": total_for_type, "correct": correct_for_type, "accuracy_percent": "N/A"
            }

    if 'overall_metrics' not in all_results_data: all_results_data['overall_metrics'] = {} # Should be initialized by now
    all_results_data['overall_metrics']['question_type_summary'] = {
        "counts_by_type": question_type_counts,
        "accuracies_by_type": accuracies_by_type_report
    }

    if inference_s_config.get("save_output_response", True):
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Final results, configuration, and metrics saved to: {output_filename}")
            
            # ADDED: Save detailed error analysis for wrong answers
            save_error_analysis(all_results_data, current_run_output_dir, logger)
            
        except Exception as e_save:
            logger.error(f"{Colors.RED}ERROR saving final JSON results to {output_filename}: {e_save}{Colors.ENDC}", exc_info=True)
    else:
        logger.info("Skipping saving final results file as per 'save_output_response' configuration.")

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    total_script_time_str = str(timedelta(seconds=total_script_time))
    
    logger.info("=" * 60)
    logger.info(f"TOTAL SCRIPT EXECUTION TIME: {Colors.GREEN}{total_script_time_str} ({total_script_time:.2f} seconds){Colors.ENDC}")
    logger.info(f"  - Dataset loading: {dataset_load_time:.2f} seconds")
    logger.info(f"  - Processing pipeline: {processing_time:.2f} seconds")
    logger.info(f"  - Other operations: {(total_script_time - dataset_load_time - processing_time):.2f} seconds")
    logger.info("=" * 60)

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
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Execution interrupted by user (Ctrl+C).{Colors.ENDC}")
    except Exception as e_main_global:
        print(f"\n{Colors.RED}CRITICAL UNEXPECTED ERROR in main execution{Colors.ENDC}: {e_main_global}")
        traceback.print_exc()
    finally:
        print("\nMain script execution finished.")