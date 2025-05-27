import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import logging
import sys
import datetime

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    RED = FAIL
    GREEN = OKGREEN
    YELLOW = WARNING
    CYAN = '\033[96m'

class RemoveColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape_pattern.sub('', message)

APP_LOGGER_NAME = "VQA_AutoGen_App"

def setup_logging(cli_args, current_config_dict, outputs_base_dir_config_key="output_base_dir"):
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(logging.DEBUG) 
    if app_logger.hasHandlers():
        app_logger.handlers.clear()
    app_logger.propagate = False

    outputs_base_dir = "outputs" 
    if current_config_dict and "inference_settings" in current_config_dict and \
        current_config_dict["inference_settings"].get(outputs_base_dir_config_key):
        outputs_base_dir = current_config_dict["inference_settings"][outputs_base_dir_config_key]
    
    try:
        os.makedirs(outputs_base_dir, exist_ok=True)
    except OSError as e:
        print(f"{Colors.WARNING}Warning: Could not create/access outputs_base_dir '{outputs_base_dir}': {e}. Falling back to './outputs'.{Colors.ENDC}")
        outputs_base_dir = "outputs"
        os.makedirs(outputs_base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset_name_for_folder = re.sub(r'[^a-zA-Z0-9_.-]', '_', cli_args.dataset_name) if cli_args.dataset_name else "unknown_dataset"
    split_name_for_folder = re.sub(r'[^a-zA-Z0-9_.-]', '_', cli_args.dataset_split) if cli_args.dataset_split else "unknown_split"
    
    vqa_flow_type_from_cli = cli_args.vqa_flow_type

    if cli_args.use_num_test_data:
        num_data_str = f"num{cli_args.num_test_data}"
    else:
        datasets_config = current_config_dict.get("datasets", {})
        percent_test_val = datasets_config.get("percent_test", 1.0)
        num_data_str = f"pct{int(percent_test_val*100)}"

    run_folder_name = f"run_{timestamp}_{dataset_name_for_folder}_{split_name_for_folder}_{vqa_flow_type_from_cli}_{num_data_str}_seed{cli_args.random_seed}"
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

    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    file_formatter = RemoveColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(file_formatter)
    app_logger.addHandler(fh)

    noisy_libraries = ["httpx", "openai", "aiohttp", "asyncio", "urllib3", "httpcore", "uvicorn", "PIL", "huggingface_hub"]
    for lib_name in noisy_libraries:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)

    app_logger.info(f"Application logging initialized. Console Level: {'DEBUG' if cli_args.verbose else 'INFO'}. File Level: DEBUG.")
    app_logger.info(f"Log file: {log_file_path}")
    app_logger.info(f"All outputs for this run will be stored in: {current_run_output_dir}")
    return app_logger, current_run_output_dir

yes_no_starters_global = [
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "am ",
    "can ", "could ", "will ", "would ", "should ", 
    "has ", "have ", "had ", "may ", "might ", "must ",
    "is there ", "are there ", "was there ", "were there ",
    "can there ", "will there ", "is it ", "are they "
]
number_starters_global = [
    "how many", "what is the number of", "count the number of", "what number"
]

def get_question_type(question_text: str) -> str:
    if not isinstance(question_text, str):
        return "other" 
    q_lower = question_text.lower().strip()
    
    if any(q_lower.startswith(s) for s in yes_no_starters_global):
        return "yes/no"
    if any(q_lower.startswith(s) for s in number_starters_global):
        return "number"
    return "other"

def create_error_result_dict(qid, img_path, q_text, target_ans, error_message_str, error_type="UnknownError", flow_type="unknown"):
    q_type = get_question_type(q_text)
    normalized_target = str(target_ans) if not isinstance(target_ans, (str, int, float)) else target_ans
    
    return {
        "question_id": qid, "image_path": img_path, "question": q_text, "question_type": q_type,
        "target_answer": target_ans, 
        "initial_answer": "", "final_answer": "[Process Failed]",
        "flow_type_used": flow_type,
        "match_baseline_failed": False, "is_numeric_reattempt": False, "analysis_output": "",
        "object_attributes_queried": "", "reattempt_answer": "", "grades": [], "majority_vote": "Error",
        "processing_time_seconds": 0.0, "error": f"{error_type}: {error_message_str}",
        "direct_accuracy_check": {
            "is_correct": False, 
            "notes": "Error in processing", 
            "cleaned_model_answer": "[Process Failed]", 
            "normalized_target_answer": str(normalized_target) 
        }
    }

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Grader:
    def __init__(self):
        self.count_correct = 0
        self.count_incorrect = 0
        self.count_correct_baseline = 0
        self.count_incorrect_baseline = 0
        self.count_total = 0

    def average_score(self):
        if self.count_total == 0:
            return 0, 0, None 

        accuracy_baseline = self.count_correct_baseline / self.count_total
        accuracy = self.count_correct / self.count_total

        stat = {
            'count_correct': self.count_correct,
            'count_incorrect': self.count_incorrect,
            'count_correct_baseline': self.count_correct_baseline,
            'count_incorrect_baseline': self.count_incorrect_baseline,
            'count_total': self.count_total
        }
        return accuracy_baseline, accuracy, stat

    def average_score_simple(self):
        if self.count_total == 0:
            return 0, None 

        accuracy = self.count_correct / self.count_total

        stat = {
            'count_correct': self.count_correct,
            'count_incorrect': self.count_incorrect,
            'count_total': self.count_total
        }
        return accuracy, stat

    def accumulate_grades(self, config, grades, match_baseline_failed):
        count_match_correct = 0
        for grade in grades:
            grade_lower = grade.lower() 
            if re.search(r'\[correct\]', grade_lower) or \
               (re.search("correct", grade_lower) and not re.search("incorrect", grade_lower)):
                count_match_correct += 1
        
        num_graders = len(grades) if grades else 0
        majority_threshold = np.ceil(num_graders / 2.0) if num_graders > 0 else 1
        
        match_correct = True if count_match_correct >= majority_threshold else False

        verbose_logging = config.get("inference_settings", {}).get("verbose", False)

        if match_correct:
            majority_vote = f'Majority vote is [Correct] with a score of {count_match_correct}/{num_graders}'
            if verbose_logging:
                print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
        else:
            majority_vote = f'Majority vote is [Incorrect] with a score of {count_match_correct}/{num_graders}'
            if verbose_logging:
                print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')

        self.count_total += 1
        if not match_baseline_failed: 
            if match_correct:
                self.count_correct_baseline += 1
                self.count_correct += 1 
            else:
                self.count_incorrect_baseline += 1
                self.count_incorrect += 1 
        else: 
            self.count_incorrect_baseline += 1
            if match_correct:
                self.count_correct += 1
            else:
                self.count_incorrect += 1

        return majority_vote

    def accumulate_grades_simple(self, config, grades):
        count_match_correct = 0
        for grade in grades:
            grade_lower = grade.lower()
            if re.search(r'\[correct\]', grade_lower) or \
               (re.search("correct", grade_lower) and not re.search("incorrect", grade_lower)):
                count_match_correct += 1
        
        num_graders = len(grades) if grades else 0
        majority_threshold = np.ceil(num_graders / 2.0) if num_graders > 0 else 1
        match_correct = True if count_match_correct >= majority_threshold else False

        verbose_logging = config.get("inference_settings", {}).get("verbose", False)

        if match_correct:
            majority_vote = f'Majority vote is [Correct] with a score of {count_match_correct}/{num_graders}'
            if verbose_logging:
                print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
        else:
            majority_vote = f'Majority vote is [Incorrect] with a score of {count_match_correct}/{num_graders}'
            if verbose_logging:
                print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')

        self.count_total += 1
        if match_correct:
            self.count_correct += 1 
        else:
            self.count_incorrect += 1 

        return majority_vote

def calculate_iou_batch(a, b):
    a = a.unsqueeze(1) 
    b = b.unsqueeze(0) 

    max_xy = torch.min(a[..., 2:], b[..., 2:])
    min_xy = torch.max(a[..., :2], b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[..., 0] * inter[..., 1]

    a_area = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    b_area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = a_area + b_area - intersection
    iou = intersection / union
    return iou

def filter_boxes_pytorch(a, b, iou_threshold=0.5):
    iou = calculate_iou_batch(a, b) 
    max_iou, _ = torch.max(iou, dim=0)
    keep = max_iou > iou_threshold
    return b[keep]

def show_anns(anns):
    if len(anns) == 0:
        return
    valid_anns = [ann for ann in anns if 'area' in ann and 'segmentation' in ann and isinstance(ann['segmentation'], np.ndarray)]
    if not valid_anns:
        print(f"{Colors.WARNING}Warning (show_anns): No valid annotations with 'area' and 'segmentation' found to sort/display.{Colors.ENDC}")
        return
        
    sorted_anns = sorted(valid_anns, key=(lambda x: x['area']), reverse=True)
    if not sorted_anns[0]['segmentation'].shape[0] > 0 or not sorted_anns[0]['segmentation'].shape[1] > 0:
        print(f"{Colors.WARNING}Warning (show_anns): Segmentation shape is invalid.{Colors.ENDC}")
        return

    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4)
    img = np.ones(img_shape, dtype=np.float32) 
    img[:,:,3] = 0 
    for ann in sorted_anns:
        m = ann['segmentation']
        if m.shape != img_shape[:2]: 
            print(f"{Colors.WARNING}Warning (show_anns): Mismatched mask shape {m.shape} for annotation, skipping.{Colors.ENDC}")
            continue
        color_mask = np.concatenate([np.random.random(3), [0.35]]) 
        img[m] = color_mask
    
    output_dir = 'test_images' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.figure(figsize=(img_shape[1]/100 if img_shape[1]>0 else 1, img_shape[0]/100 if img_shape[0]>0 else 1)) 
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'masks.jpg'))
        plt.close() 
    except Exception as e:
        print(f"{Colors.FAIL}Error in show_anns saving image: {e}{Colors.ENDC}")

def plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename):
    try:
        from groundingdino.util.inference import annotate 
    except ImportError:
        print(f"{Colors.WARNING}Warning: groundingdino library not found. Cannot plot GroundingDINO bboxes.{Colors.ENDC}")
        return

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[:, :, ::-1] 
    
    output_dir = 'test_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        if annotated_frame.dtype == np.float32 or annotated_frame.dtype == np.float64:
            if np.max(annotated_frame) <= 1.0 and np.min(annotated_frame) >=0.0 :
                 annotated_frame = (annotated_frame * 255).astype(np.uint8)
            else:
                 annotated_frame = annotated_frame.astype(np.uint8)


        plt.figure(figsize=(annotated_frame.shape[1]/100 if annotated_frame.shape[1]>0 else 1, annotated_frame.shape[0]/100 if annotated_frame.shape[0]>0 else 1))
        plt.imshow(annotated_frame)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'bboxes_' + filename + '.jpg')) 
        plt.close()
    except Exception as e:
        print(f"{Colors.FAIL}Error in plot_grounding_dino_bboxes saving image: {e}{Colors.ENDC}")

def load_answer_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file: 
            answer_list = json.load(file)
        return answer_list
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: Answer list file not found at {file_path}{Colors.ENDC}")
        return None
    except json.JSONDecodeError:
        print(f"{Colors.FAIL}Error: Could not decode JSON from answer list file {file_path}{Colors.ENDC}")
        return None

def save_output_predictions_vqav2(question_id, model_answer, answer_list, split='test', verbose=False, output_dir_base='outputs'):
    def filter_response(response, valid_answers): 
        if not isinstance(response, str) or not valid_answers: 
            return response if isinstance(response, str) else ""

        response_words = response.lower().split() 
        normalized_valid_answers = set(str(ans).lower() for ans in valid_answers if isinstance(ans, str))

        filtered_words = [word for word in response_words if word in normalized_valid_answers]
        if filtered_words:
            return ' '.join(filtered_words)
        return response 

    extracted_answer_text = ""
    if isinstance(model_answer, str):
        extracted_answer_text = model_answer 
    
    filtered_response = extracted_answer_text 
    if answer_list: 
        try:
            filtered_response = filter_response(extracted_answer_text, answer_list)
        except Exception as e: 
            if verbose: print(f"{Colors.WARNING}Could not filter response for QID {question_id}: {e}. Using extracted answer.{Colors.ENDC}")
            pass 

    qid_value = question_id.item() if hasattr(question_id, 'item') and callable(question_id.item) else question_id
    try:
        qid_to_save = int(qid_value)
    except ValueError:
        qid_to_save = str(qid_value)
        
    result = {
        "question_id": qid_to_save, 
        "answer": filtered_response if filtered_response else extracted_answer_text
    }
    if verbose:
        print(f"Saving result for QID {result['question_id']}: {result['answer']}")
    
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base, exist_ok=True)
    
    saved_file_name = os.path.join(output_dir_base, f'submit_vqav2_{split}_results.json') 

    data_to_save = []
    if os.path.exists(saved_file_name) and os.path.getsize(saved_file_name) > 0:
        try:
            with open(saved_file_name, 'r', encoding='utf-8') as f: 
                data_to_save = json.load(f)
                if not isinstance(data_to_save, list):
                    data_to_save = [] 
        except json.JSONDecodeError:
            if verbose: print(f"{Colors.WARNING}Warning: {saved_file_name} contains invalid JSON. Starting with a new list.{Colors.ENDC}")
            data_to_save = [] 
    
    data_to_save.append(result)

    try:
        with open(saved_file_name, 'w', encoding='utf-8') as f: 
            json.dump(data_to_save, f, indent=2, ensure_ascii=False) 
    except Exception as e:
        print(f"{Colors.FAIL}Error saving predictions to {saved_file_name}: {e}{Colors.ENDC}")

def write_response_to_json(question_id, response_dict, output_response_filename):
    data = {}
    if os.path.exists(output_response_filename) and os.path.getsize(output_response_filename) > 0: 
        try:
            with open(output_response_filename, 'r', encoding='utf-8') as file: 
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"{Colors.WARNING}Warning: {output_response_filename} contains invalid JSON. Starting with a new dict.{Colors.ENDC}")
            data = {} 
    
    qid_value = question_id.item() if hasattr(question_id, 'item') and callable(question_id.item) else question_id
    data[str(qid_value)] = response_dict

    try:
        with open(output_response_filename, 'w', encoding='utf-8') as file: 
            json.dump(data, file, indent=4, ensure_ascii=False) 
    except Exception as e:
        print(f"{Colors.FAIL}Error writing response to {output_response_filename}: {e}{Colors.ENDC}")

def record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename):
    data = {}
    if os.path.exists(output_response_filename) and os.path.getsize(output_response_filename) > 0: 
        try:
            with open(output_response_filename, 'r', encoding='utf-8') as file: 
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"{Colors.WARNING}Warning: {output_response_filename} for accuracy recording contains invalid JSON. Starting new.{Colors.ENDC}")
            data = {} 

    data['baseline_accuracy'] = str(baseline_accuracy)
    data['final_accuracy'] = str(final_accuracy)
    data['stats'] = stats 

    try:
        with open(output_response_filename, 'w', encoding='utf-8') as file: 
            json.dump(data, file, indent=4, ensure_ascii=False) 
    except Exception as e:
        print(f"{Colors.FAIL}Error recording final accuracy to {output_response_filename}: {e}{Colors.ENDC}")

def extract_error_analysis_data(result_item: dict, flow_type: str) -> dict:
    """Extract detailed error analysis data from a result item."""
    error_analysis = {
        "question_id": result_item.get("question_id"),
        "question": result_item.get("question"),
        "question_type": result_item.get("question_type"),
        "target_answer": result_item.get("target_answers"),
        "predicted_answer": result_item.get("final_answer"),
        "flow_type": flow_type,
        "error_type": "incorrect_answer",
        "confidence_score": result_item.get("confidence_score", 0.0),
        "processing_time": result_item.get("processing_time_seconds", 0.0),
        "accuracy_details": result_item.get("direct_accuracy_check", {}),
        "model_interactions": {},
        "reasoning_analysis": {},
        "failure_points": []
    }
    
    # Extract flow-specific interaction data
    if flow_type == "specialized":
        # Extract specialized flow interactions
        error_analysis["model_interactions"] = {
            "initial_answer": result_item.get("initial_answer", ""),
            "analysis_output": result_item.get("analysis_output", ""),
            "object_attributes_queried": result_item.get("object_attributes_queried", ""),
            "reattempt_answer": result_item.get("reattempt_answer", ""),
            "grades": result_item.get("grades", []),
            "majority_vote": result_item.get("majority_vote", ""),
            "match_baseline_failed": result_item.get("match_baseline_failed", False),
            "is_numeric_reattempt": result_item.get("is_numeric_reattempt", False)
        }
        
        # Analyze reasoning for specialized flow
        error_analysis["reasoning_analysis"] = {
            "initial_reasoning_quality": "good" if result_item.get("initial_answer") and not any(marker in result_item.get("initial_answer", "").lower() for marker in ["failed", "error", "empty"]) else "poor",
            "analysis_triggered": bool(result_item.get("analysis_output")),
            "reattempt_triggered": bool(result_item.get("reattempt_answer")),
            "grading_consensus": result_item.get("majority_vote", "").lower() if result_item.get("majority_vote") else "no_consensus"
        }
        
        # Identify failure points
        if result_item.get("match_baseline_failed"):
            error_analysis["failure_points"].append("baseline_matching_failed")
        if not result_item.get("initial_answer") or "failed" in result_item.get("initial_answer", "").lower():
            error_analysis["failure_points"].append("initial_answer_generation_failed")
        if result_item.get("analysis_output") and "error" in result_item.get("analysis_output", "").lower():
            error_analysis["failure_points"].append("failure_analysis_inconclusive")
        if result_item.get("grades") and len(result_item.get("grades", [])) == 0:
            error_analysis["failure_points"].append("no_grading_performed")
            
    elif flow_type == "reflection":
        # Extract reflection flow interactions
        error_analysis["model_interactions"] = {
            "final_answer": result_item.get("final_answer", ""),
            "agent_response": "Available" if result_item.get("final_answer") and not result_item.get("final_answer").startswith("[") else "Failed"
        }
        
        error_analysis["reasoning_analysis"] = {
            "response_quality": "good" if result_item.get("final_answer") and not result_item.get("final_answer").startswith("[") else "poor",
            "agent_failure": result_item.get("final_answer", "").startswith("[") if result_item.get("final_answer") else True
        }
        
        # Identify failure points
        if result_item.get("final_answer", "").startswith("["):
            error_analysis["failure_points"].append("agent_response_failed")
        if result_item.get("error"):
            error_analysis["failure_points"].append(f"pipeline_error: {result_item.get('error')}")
            
    elif flow_type == "debate":
        # Extract debate flow interactions
        additional_metadata = result_item.get("additional_metadata", {})
        all_rounds = additional_metadata.get("all_rounds_outputs", [])
        
        error_analysis["model_interactions"] = {
            "num_debate_rounds": additional_metadata.get("debate_rounds", 0),
            "num_solvers": additional_metadata.get("num_solvers", 0),
            "majority_vote_count": additional_metadata.get("majority_vote_count", 0),
            "final_answer": result_item.get("final_answer", ""),
            "solver_responses_by_round": []
        }
        
        # Extract detailed solver interactions
        for round_idx, round_outputs in enumerate(all_rounds):
            round_data = {
                "round_number": round_idx + 1,
                "solvers": []
            }
            for solver_output in round_outputs:
                solver_data = {
                    "solver_name": solver_output.get("solver_name", ""),
                    "answer": solver_output.get("answer", ""),
                    "reasoning": solver_output.get("reasoning", ""),
                    "confidence": solver_output.get("confidence", 0.0),
                    "raw_output": solver_output.get("raw_output", "")[:500] + "..." if len(solver_output.get("raw_output", "")) > 500 else solver_output.get("raw_output", "")  # Truncate for readability
                }
                round_data["solvers"].append(solver_data)
            error_analysis["model_interactions"]["solver_responses_by_round"].append(round_data)
        
        # Analyze debate reasoning
        error_analysis["reasoning_analysis"] = {
            "consensus_reached": additional_metadata.get("majority_vote_count", 0) > 1,
            "solver_agreement_level": additional_metadata.get("majority_vote_count", 0) / max(additional_metadata.get("num_solvers", 1), 1),
            "debate_effectiveness": "high" if additional_metadata.get("majority_vote_count", 0) >= 2 else "low",
            "final_confidence": result_item.get("confidence_score", 0.0)
        }
        
        # Identify failure points
        if additional_metadata.get("majority_vote_count", 0) <= 1:
            error_analysis["failure_points"].append("no_consensus_reached")
        if result_item.get("confidence_score", 0.0) < 0.5:
            error_analysis["failure_points"].append("low_confidence_answer")
        if any("Error" in solver.get("answer", "") for round_outputs in all_rounds for solver in round_outputs):
            error_analysis["failure_points"].append("solver_errors_detected")
    
    return error_analysis

def save_error_analysis(all_results_data: dict, output_dir: str, logger) -> None:
    """Save detailed error analysis for wrong answers to a separate file."""
    import time  # Import here to avoid circular imports
    
    try:
        wrong_answers = []
        total_processed = 0
        
        for qid, result_item in all_results_data.get("results_by_question_id", {}).items():
            if not isinstance(result_item, dict):
                continue
                
            total_processed += 1
            flow_type = result_item.get("flow_type_used", "unknown")
            q_type = result_item.get("question_type", "other")
            
            # Skip if there's a pipeline error
            if result_item.get("error"):
                continue
            
            # Determine if answer is correct
            is_correct = False
            if flow_type == "specialized":
                majority_vote_value = result_item.get("majority_vote", "")
                if majority_vote_value and isinstance(majority_vote_value, str):
                    is_correct = "[correct]" in majority_vote_value.lower()
            elif flow_type in ["reflection", "debate"]:
                accuracy_check = result_item.get("direct_accuracy_check", {})
                if q_type == "yes/no":
                    is_correct = accuracy_check.get("is_correct_strict", accuracy_check.get("is_correct", False))
                elif q_type == "number":
                    is_correct = accuracy_check.get("is_correct_loose_numeric", accuracy_check.get("is_correct", False))
                elif q_type == "other":
                    is_correct = accuracy_check.get("is_correct_f1", accuracy_check.get("is_correct", False))
                else:
                    is_correct = accuracy_check.get("is_correct", False)
            
            # If answer is wrong, extract detailed error analysis
            if not is_correct:
                error_analysis = extract_error_analysis_data(result_item, flow_type)
                wrong_answers.append(error_analysis)
        
        # Create error analysis summary
        error_summary = {
            "analysis_metadata": {
                "total_processed_questions": total_processed,
                "total_wrong_answers": len(wrong_answers),
                "error_rate": (len(wrong_answers) / total_processed * 100) if total_processed > 0 else 0,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "flows_analyzed": list(set(item.get("flow_type", "unknown") for item in wrong_answers))
            },
            "error_patterns": {
                "by_question_type": {},
                "by_flow_type": {},
                "common_failure_points": {},
                "confidence_distribution": {"low": 0, "medium": 0, "high": 0}
            },
            "detailed_wrong_answers": wrong_answers
        }
        
        # Analyze error patterns
        for error_item in wrong_answers:
            # By question type
            q_type = error_item.get("question_type", "other")
            if q_type not in error_summary["error_patterns"]["by_question_type"]:
                error_summary["error_patterns"]["by_question_type"][q_type] = 0
            error_summary["error_patterns"]["by_question_type"][q_type] += 1
            
            # By flow type
            flow_type = error_item.get("flow_type", "unknown")
            if flow_type not in error_summary["error_patterns"]["by_flow_type"]:
                error_summary["error_patterns"]["by_flow_type"][flow_type] = 0
            error_summary["error_patterns"]["by_flow_type"][flow_type] += 1
            
            # Common failure points
            for failure_point in error_item.get("failure_points", []):
                if failure_point not in error_summary["error_patterns"]["common_failure_points"]:
                    error_summary["error_patterns"]["common_failure_points"][failure_point] = 0
                error_summary["error_patterns"]["common_failure_points"][failure_point] += 1
            
            # Confidence distribution
            confidence = error_item.get("confidence_score", 0.0)
            if confidence < 0.3:
                error_summary["error_patterns"]["confidence_distribution"]["low"] += 1
            elif confidence < 0.7:
                error_summary["error_patterns"]["confidence_distribution"]["medium"] += 1
            else:
                error_summary["error_patterns"]["confidence_distribution"]["high"] += 1
        
        # Save error analysis file
        error_analysis_filename = os.path.join(output_dir, "error_analysis_detailed.json")
        with open(error_analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(error_summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"{Colors.CYAN}Error Analysis Report:{Colors.ENDC}")
        logger.info(f"  Total wrong answers: {Colors.RED}{len(wrong_answers)}{Colors.ENDC} out of {total_processed}")
        logger.info(f"  Error rate: {Colors.RED}{error_summary['analysis_metadata']['error_rate']:.2f}%{Colors.ENDC}")
        logger.info(f"  Detailed error analysis saved to: {error_analysis_filename}")
        
        # Log top failure patterns
        if error_summary["error_patterns"]["common_failure_points"]:
            logger.info(f"  Top failure points:")
            sorted_failures = sorted(error_summary["error_patterns"]["common_failure_points"].items(), 
                                   key=lambda x: x[1], reverse=True)
            for failure_point, count in sorted_failures[:3]:
                logger.info(f"    - {failure_point}: {count} occurrences")
        
    except Exception as e:
        logger.error(f"{Colors.RED}ERROR saving error analysis: {e}{Colors.ENDC}", exc_info=True)