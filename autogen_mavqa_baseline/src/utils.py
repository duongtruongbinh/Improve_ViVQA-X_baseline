import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

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
            return 0, 0, None 

        accuracy = self.count_correct / self.count_total

        stat = {
            'count_correct': self.count_correct,
            'count_incorrect': self.count_incorrect,
            'count_total': self.count_total
        }
        return accuracy, stat

    def accumulate_grades(self, config, grades, match_baseline_failed):
        # The 'config' parameter was 'args' in the original, changed for clarity
        # assuming it's the main application configuration object.
        count_match_correct = 0
        for grade in grades:
            grade_lower = grade.lower() # Process a lowercase version
            if re.search(r'\[correct\]', grade_lower) or \
               (re.search("correct", grade_lower) and not re.search("incorrect", grade_lower)):
                count_match_correct += 1
        
        # Majority vote: if at least 2 out of 3 graders agree (assuming 3 graders)
        # This threshold might need to be dynamic based on actual number of graders
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
        # The 'config' parameter was 'args' in the original
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
            self.count_correct_baseline += 1 
            self.count_correct += 1 
        else:
            self.count_incorrect_baseline += 1
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
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    output_dir = 'test_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, 'masks.jpg'), img)


def plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename):
    from groundingdino.util.inference import annotate 

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[:, :, [2, 1, 0]] 
    
    output_dir = 'test_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, 'bboxes' + filename + '.jpg'), annotated_frame)


def load_answer_list(file_path):
    with open(file_path, 'r') as file:
        answer_list = json.load(file)
    return answer_list


def save_output_predictions_vqav2(question_id, model_answer, answer_list, split='test', verbose=False):
    def filter_response(response, answer_list):
        response_words = response.split()
        filtered_words = [word for word in response_words if word in answer_list]
        filtered_response = ' '.join(filtered_words)
        return filtered_response

    extracted_answer_match = re.search(r"\s*\[Answer\](.*)|\s*\[Reattempted Answer\](.*)", model_answer, re.DOTALL)
    extracted_answer_text = ""

    if extracted_answer_match:
        # Get the content from the first non-None group
        ans_group1 = extracted_answer_match.group(1)
        ans_group2 = extracted_answer_match.group(2)
        if ans_group1:
            extracted_answer_text = ans_group1.strip()
        elif ans_group2:
            extracted_answer_text = ans_group2.strip()
    
    filtered_response = ""
    try:
        filtered_response = filter_response(extracted_answer_text, answer_list)
    except Exception: 
        if verbose: print("Could not filter response, using extracted answer or empty.")
        pass


    qid_value = question_id.item() if hasattr(question_id, 'item') else question_id
    result = {
        "question_id": qid_value,
        "answer": filtered_response if filtered_response else extracted_answer_text
    }
    if verbose:
        print(result)
    
    # Ensure outputs directory exists
    output_main_dir = 'outputs'
    if not os.path.exists(output_main_dir):
        os.makedirs(output_main_dir, exist_ok=True)
    saved_file_name = os.path.join(output_main_dir, f'submit_vqav2_{split}_4.json')


    data_to_save = []
    if os.path.exists(saved_file_name) and os.path.getsize(saved_file_name) > 0:
        try:
            with open(saved_file_name, 'r') as f:
                data_to_save = json.load(f)
                if not isinstance(data_to_save, list): # Ensure it's a list
                    data_to_save = [data_to_save] 
        except json.JSONDecodeError:
             if verbose: print(f"Warning: {saved_file_name} contains invalid JSON. Starting with a new list.")
             data_to_save = [] # Reset if file is corrupted
    
    data_to_save.append(result)

    with open(saved_file_name, 'w') as f:
        json.dump(data_to_save, f, indent=2)


def write_response_to_json(question_id, response_dict, output_response_filename):
    data = {}
    if os.path.exists(output_response_filename):
        try:
            with open(output_response_filename, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
             print(f"Warning: {output_response_filename} contains invalid JSON. Starting with a new dict.")
             data = {} # Reset if file is corrupted
    
    qid_value = question_id.item() if hasattr(question_id, 'item') else question_id
    data[str(qid_value)] = response_dict

    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=4)


def record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename):
    data = {}
    if os.path.exists(output_response_filename):
        try:
            with open(output_response_filename, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
             print(f"Warning: {output_response_filename} for accuracy recording contains invalid JSON. Starting new.")
             data = {} # Reset if file is corrupted


    data['baseline_accuracy'] = str(baseline_accuracy)
    data['final_accuracy'] = str(final_accuracy)
    data['stats'] = stats

    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=4)