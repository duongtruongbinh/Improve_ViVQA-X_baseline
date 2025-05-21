import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import asyncio
import argparse
import torch
from torch.utils.data import Subset, DataLoader
import json
import traceback
import time

try:
    from config_loader import app_config
    from image_utils import process_image_for_vlm_agent
    from dataloader import VQAv2Dataset, GQADataset
    from utils import Grader
    from vllm_clients import (
        vlm_client_vllm,
        VLM_MODEL_NAME_ACTUAL,
        API_PROVIDER
    )
except ImportError as e:
    print(f"Import Error in evaluate_single_model.py: {e}.")
    print("Ensure paths are correct and required modules (config_loader, vllm_clients, etc.) are in the same directory or configured in PYTHONPATH.")
    exit(1)
except Exception as e_init:
    print(f"Error during initial imports or loading from vllm_clients: {e_init}")
    exit(1)


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'

def call_vlm_with_autogen_client(
    autogen_client_instance,
    system_prompt: str,
    user_question: str,
    image_url: str,
    sampling_params: dict
):
    if autogen_client_instance is None:
        print(f"{Colors.RED}Error: AutoGen VLM client instance is None. Cannot make API call.{Colors.ENDC}")
        return "Error: VLM Client Not Initialized"
    
    messages_payload = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    try:
        response = autogen_client_instance.create(
            messages=messages_payload,
            temperature=sampling_params.get("temperature"),
            max_tokens=sampling_params.get("max_tokens")
        )
        
        if response and response.choices and response.choices[0].message and response.choices[0].message.content is not None:
            generated_text = response.choices[0].message.content
            return generated_text.strip()
        else:
            print(f"{Colors.RED}Error: Malformed response from AutoGen VLM client.{Colors.ENDC}")
            print(f"Response object: {response}")
            return "Error: Malformed Client Response"

    except Exception as e:
        print(f"{Colors.RED}Error calling VLM via AutoGen client: {e}{Colors.ENDC}")
        traceback.print_exc()
        return "Error: Client API Call Failed"


async def main_evaluation_logic():
    current_config = app_config
    if not current_config:
        print(f"{Colors.RED}CRITICAL ERROR: Configuration could not be loaded (app_config is None). Ensure config_loader.py works.{Colors.ENDC}")
        return

    dataset_s_config = current_config.get("datasets", {})
    inference_s_config = current_config.get("inference_settings", {})
    
    default_dataset_name = "vqa-v2"
    default_split_vqa = "val"
    default_split_gqa = "val" 
    default_verbose = True
    default_use_num = True 
    default_num_test = 20 
    default_random_seed = 42
    
    default_prompt_path_arg = os.path.join(_SCRIPT_DIR, "..", "prompts", "single_model", "basic_vqa.j2")
    default_output_file_arg = os.path.join(_SCRIPT_DIR, "..", "outputs", "single_model_results.jsonl")

    parser = argparse.ArgumentParser(description="Single Model VQA Evaluation")
    parser.add_argument("--dataset_name", type=str, default=default_dataset_name, choices=['vqa-v2', 'gqa'])
    parser.add_argument("--dataset_split", type=str, default=None, help=f"Default for vqa-v2: {default_split_vqa}, for gqa: {default_split_gqa}")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=default_verbose)
    parser.add_argument("--use_num_test_data", action=argparse.BooleanOptionalAction, default=default_use_num)
    parser.add_argument("--num_test_data", type=int, default=default_num_test)
    parser.add_argument("--random_seed", type=int, default=default_random_seed)
    parser.add_argument("--prompt_file", type=str, default=default_prompt_path_arg, help="Path to the system prompt file for the single model.")
    parser.add_argument("--output_file", type=str, default=default_output_file_arg, help="Path to save evaluation results.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for the VLM.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens for VLM generation.")


    cli_args = parser.parse_args()

    effective_dataset_name = cli_args.dataset_name
    effective_verbose = cli_args.verbose
    effective_use_num = cli_args.use_num_test_data # Sẽ luôn là True nếu default là True và không bị override
    effective_num_test = cli_args.num_test_data
    effective_random_seed = cli_args.random_seed
    single_model_prompt_file = cli_args.prompt_file
    output_results_file = cli_args.output_file
    eval_temperature = cli_args.temperature
    eval_max_tokens = cli_args.max_tokens


    effective_split_name = cli_args.dataset_split
    if not effective_split_name:
        effective_split_name = default_split_gqa if effective_dataset_name == 'gqa' else default_split_vqa

    print("\n--- Single Model Evaluation Configuration ---")
    print(f"API Provider from vllm_clients: {API_PROVIDER.upper()}")
    print(f"VLM Model for Evaluation: {VLM_MODEL_NAME_ACTUAL}")
    print(f"Dataset: {effective_dataset_name}, Split: {effective_split_name}")
    print(f"Verbose: {effective_verbose}")
    print(f"Random Seed: {effective_random_seed}")
    print(f"Prompt File: {single_model_prompt_file}")
    print(f"Output Results To: {output_results_file}")
    print(f"Sampling Temperature: {eval_temperature}")
    print(f"Sampling Max Tokens: {eval_max_tokens}")
    if effective_use_num:
        print(f"Using Fixed Number of Test Data: {effective_num_test}")
    else:
         print(f"{Colors.YELLOW}Warning: --use_num_test_data is False, this script primarily defaults to fixed num. behavior will be fixed num.{Colors.ENDC}")
         effective_use_num = True # Enforce fixed number for this script's typical use

    print("-------------------------------------------\n")
    
    if vlm_client_vllm is None:
        provider_name_for_error = API_PROVIDER.upper() if API_PROVIDER else "Unknown"
        print(f"{Colors.RED}CRITICAL ERROR: VLM client (for {provider_name_for_error}) is not initialized. Check vllm_clients.py.{Colors.ENDC}")
        return

    try:
        if not os.path.isabs(single_model_prompt_file) and not os.path.exists(single_model_prompt_file) :
             print(f"{Colors.YELLOW}Warning: Prompt file '{single_model_prompt_file}' might not be found if not absolute or relative to CWD/PYTHONPATH. Trying default relative path logic.{Colors.ENDC}")
        
        with open(single_model_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt_content = f.read()
    except FileNotFoundError:
        print(f"{Colors.RED}CRITICAL ERROR: Prompt file '{single_model_prompt_file}' not found.{Colors.ENDC}")
        return

    torch.manual_seed(effective_random_seed)
    try:
        dataset_class = GQADataset if effective_dataset_name == 'gqa' else VQAv2Dataset
        current_config["datasets"]["dataset_name"] = effective_dataset_name 
        if effective_dataset_name == 'gqa':
            current_config["datasets"]["gqa_dataset_split"] = effective_split_name
        else:
            current_config["datasets"]["vqa_v2_dataset_split"] = effective_split_name

        full_dataset = dataset_class(current_config, transform=None)
    except Exception as e_load:
        print(f"{Colors.RED}CRITICAL ERROR: Failed to load Dataset '{effective_dataset_name}'.{Colors.ENDC}")
        print(f"Error details: {e_load}")
        traceback.print_exc()
        return

    if not full_dataset or len(full_dataset) == 0:
        print(f"{Colors.RED}CRITICAL ERROR: Dataset is empty. Exiting.{Colors.ENDC}")
        return
    print(f"Full dataset '{effective_dataset_name}' (split: '{effective_split_name}') loaded with {len(full_dataset)} items.")

    actual_num_to_use = min(effective_num_test, len(full_dataset))
    if actual_num_to_use <= 0 :
        print(f"{Colors.YELLOW}Warning: num_test_data ({effective_num_test}) results in {actual_num_to_use} items. Using 1 if possible.{Colors.ENDC}")
        actual_num_to_use = 1 if len(full_dataset) > 0 else 0
    
    if actual_num_to_use == 0:
        print(f"{Colors.RED}CRITICAL ERROR: Resulting number of items to test is 0. Exiting.{Colors.ENDC}")
        return

    print(f"Selecting {actual_num_to_use} items randomly for this run.")
    test_subset_idx = torch.randperm(len(full_dataset))[:actual_num_to_use]
    test_subset = Subset(full_dataset, test_subset_idx.tolist())
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    grader_instance = Grader()
    all_run_results = []

    print(f"\nStarting Single Model VQA Evaluation (on {len(test_loader)} items)...")
    start_run_time = time.time()

    vlm_call_sampling_params = {"temperature": eval_temperature, "max_tokens": eval_max_tokens}

    for i, data_batch in enumerate(test_loader):
        image_path = data_batch['image_path'][0]
        question_text = data_batch['question'][0]
        question_id_tensor = data_batch['question_id'][0]
        target_answer_text = data_batch['answer'][0]
        current_qid = str(question_id_tensor.item()) if hasattr(question_id_tensor, 'item') else str(question_id_tensor)

        if effective_verbose or (i + 1) % 1 == 0:
            print(f"[Eval Loop] Item {i + 1}/{len(test_loader)} (QID: {current_qid})")

        if not os.path.exists(image_path):
            print(f"{Colors.YELLOW}Warning: Image not found at '{image_path}' for QID {current_qid}. Skipping.{Colors.ENDC}")
            all_run_results.append({
                "question_id": current_qid, "question": question_text, "image_path": image_path,
                "target_answer": target_answer_text, "model_answer": "Error: Image Not Found", "grade": "Error",
                "processing_time_seconds": 0
            })
            continue
        
        item_start_time = time.time()
        model_answer_text = "Error: Processing Failed"
        current_grade = "[Error]"
        try:
            image_url_for_vlm = process_image_for_vlm_agent(image_path, current_config)
            if not image_url_for_vlm or not (image_url_for_vlm.startswith("data:image") or image_url_for_vlm.startswith("http")):
                raise ValueError(f"Invalid image URL from process_image_for_vlm_agent: {str(image_url_for_vlm)[:100]}")

            model_answer_text = call_vlm_with_autogen_client(
                autogen_client_instance=vlm_client_vllm,
                system_prompt=system_prompt_content,
                user_question=question_text,
                image_url=image_url_for_vlm,
                sampling_params=vlm_call_sampling_params
            )

            if effective_verbose:
                print(f"  Q: {question_text}")
                print(f"  Raw Model Ans: {model_answer_text}")
            
            if model_answer_text.lower() == target_answer_text.lower():
                current_grade = "[Correct]"
            elif target_answer_text.lower() == "yes" and model_answer_text.lower() == "yes":
                current_grade = "[Correct]"
            elif target_answer_text.lower() == "no" and model_answer_text.lower() == "no":
                current_grade = "[Correct]"
            elif model_answer_text == "Unable to answer":
                current_grade = "[Incorrect] (Unable to answer)"
            elif "Error:" in model_answer_text or "Client Not Initialized" in model_answer_text or "Malformed Client Response" in model_answer_text :
                current_grade = "[Error]"
            else:
                current_grade = "[Incorrect]"
            
            if effective_verbose:
                print(f"  Target Ans: {target_answer_text}")
                print(f"  Grade: {current_grade}")

        except Exception as e_item:
            print(f"{Colors.RED}Error processing item QID {current_qid}: {e_item}{Colors.ENDC}")
            traceback.print_exc()
            current_grade = "[Error]"
            if "model_answer_text" not in locals() or model_answer_text == "Error: Processing Failed":
                model_answer_text = f"Error: Exception during processing - {str(e_item)[:100]}"
        
        item_processing_time = time.time() - item_start_time
        all_run_results.append({
            "question_id": current_qid,
            "question": question_text,
            "image_path": image_path,
            "target_answer": target_answer_text,
            "model_answer": model_answer_text,
            "grade": current_grade,
            "processing_time_seconds": round(item_processing_time, 2)
        })

    total_evaluated = len(all_run_results)
    correct_final = sum(1 for r in all_run_results if r["grade"] == "[Correct]")
    
    final_accuracy = (correct_final / total_evaluated * 100) if total_evaluated > 0 else 0

    print("\n--- Single Model Evaluation Finished ---")
    print(f"Total items evaluated: {total_evaluated}")
    print(f"Correct answers: {correct_final}")
    print(f"Accuracy: {Colors.GREEN}{final_accuracy:.2f}%{Colors.ENDC}")
    total_run_time = time.time() - start_run_time
    print(f"Total evaluation time: {total_run_time:.2f} seconds")

    output_dir_path = os.path.dirname(output_results_file)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
    try:
        with open(output_results_file, 'w', encoding='utf-8') as f:
            for res_item in all_run_results:
                f.write(json.dumps(res_item) + '\n')
        print(f"Evaluation results saved to: {output_results_file}")
    except IOError as e:
        print(f"{Colors.RED}Error saving results to {output_results_file}: {e}{Colors.ENDC}")

if __name__ == "__main__":
    if API_PROVIDER.lower() == "vllm" and vlm_client_vllm is None:
         print(f"{Colors.RED}CRITICAL: Local vlm_client_vllm was not initialized successfully in vllm_clients.py. Exiting.{Colors.ENDC}")
    elif API_PROVIDER.lower() == "openai" and vlm_client_vllm is None:
         print(f"{Colors.RED}CRITICAL: OpenAI VLM client (vlm_client_vllm) was not initialized successfully in vllm_clients.py. Exiting.{Colors.ENDC}")
    elif vlm_client_vllm is None:
         print(f"{Colors.RED}CRITICAL: vlm_client_vllm is None for an unknown API_PROVIDER ('{API_PROVIDER}'). Exiting.{Colors.ENDC}")
    else:
        asyncio.run(main_evaluation_logic())
    
    print("\nSingle model evaluation script finished.")