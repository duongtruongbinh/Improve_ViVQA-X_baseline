import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import re
import os
from collections import defaultdict


class Grader:
    def __init__(self):
        self.scores = []
        self.baseline_scores = []
        self.by_type = defaultdict(list)
        self.baseline_by_type = defaultdict(list)
        self.count_total = 0
        
        # VQA evaluation constants
        self.CONTRACTIONS = {
            'aint': "ain't",
            'arent': "aren't",
            'cant': "can't",
            'couldve': "could've",
            'couldnt': "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            'didnt': "didn't",
            'doesnt': "doesn't",
            'dont': "don't",
            'hadnt': "hadn't",
            "hadnt've": "hadn't've",
            "haven'tve": "hadn't've",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hed': "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            'hes': "he's",
            'howd': "how'd",
            'howll': "how'll",
            'hows': "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            'Im': "I'm",
            'Ive': "I've",
            'isnt': "isn't",
            'itd': "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            'itll': "it'll",
            "let's": "let's",
            'maam': "ma'am",
            'mightnt': "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            'mightve': "might've",
            'mustnt': "mustn't",
            'mustve': "must've",
            'neednt': "needn't",
            'notve': "not've",
            'oclock': "o'clock",
            'oughtnt': "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            'shant': "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            'shouldve': "should've",
            'shouldnt': "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": 'somebodyd',
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            'somebodyll': "somebody'll",
            'somebodys': "somebody's",
            'someoned': "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            'someonell': "someone'll",
            'someones': "someone's",
            'somethingd': "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            'somethingll': "something'll",
            'thats': "that's",
            'thered': "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            'therere': "there're",
            'theres': "there's",
            'theyd': "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            'theyll': "they'll",
            'theyre': "they're",
            'theyve': "they've",
            'twas': "'twas",
            'wasnt': "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            'weve': "we've",
            'werent': "weren't",
            'whatll': "what'll",
            'whatre': "what're",
            'whats': "what's",
            'whatve': "what've",
            'whens': "when's",
            'whered': "where'd",
            'wheres': "where's",
            'whereve': "where've",
            'whod': "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            'wholl': "who'll",
            'whos': "who's",
            'whove': "who've",
            'whyll': "why'll",
            'whyre': "why're",
            'whys': "why's",
            'wont': "won't",
            'wouldve': "would've",
            'wouldnt': "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            'yall': "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            'youd': "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've",
        }
        
        self.MANUAL_MAP = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10',
        }
        
        self.ARTICLES = {'a', 'an', 'the'}
        self.PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')',
                     '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
        self.COMMA_RE = re.compile(r'(\d)(,)(\d)')
        self.PERIOD_RE = re.compile(r'(?!<=\d)(\.)(?!\d)')

    def normalize(self, text: str) -> str:
        """Lower, strip punctuation/digits/articles, expand contractions."""
        text = text.strip().lower()
        # strip bad commas
        text = self.COMMA_RE.sub(r'\1\3', text)
        # remove punctuation
        for p in self.PUNCT:
            text = text.replace(p, ' ')
        # remove stray periods
        text = self.PERIOD_RE.sub('', text)
        # tokenize and map
        words = []
        for w in text.split():
            w = self.MANUAL_MAP.get(w, w)
            if w in self.ARTICLES:
                continue
            words.append(self.CONTRACTIONS.get(w, w))
        return ' '.join(words)

    def vqa_score(self, pred: str, gt_answers: list) -> float:
        """Compute min(#matches/3, 1) with normalized strings."""
        pred_n = self.normalize(pred)
        # Handle both old format (list of strings) and new format (list of dicts)
        if isinstance(gt_answers, list) and len(gt_answers) > 0:
            if isinstance(gt_answers[0], dict):
                gt_n = [self.normalize(d['answer']) for d in gt_answers]
            else:
                gt_n = [self.normalize(str(answer)) for answer in gt_answers]
        else:
            gt_n = [self.normalize(str(gt_answers))]
        
        matches = sum(1 for a in gt_n if a == pred_n)
        return min(matches / 3.0, 1.0)

    def accumulate_grades(self, args, grades, match_baseline_failed, target_answer=None, model_answer=None, answer_type='unknown', initial_answer=None):
        """Accumulate grades using VQA evaluation scoring"""
        self.count_total += 1
        
        # Use VQA scoring if we have target_answer and model_answer
        if target_answer is not None and model_answer is not None:
            # For baseline (initial answer)
            if not match_baseline_failed:
                baseline_score = self.vqa_score(model_answer, target_answer)
                final_score = baseline_score  # Same as baseline if not reattempted
            else:
                # This is a reattempted answer, calculate baseline score from initial_answer
                if initial_answer is not None:
                    baseline_score = self.vqa_score(initial_answer, target_answer)
                else:
                    baseline_score = 0.0  # Assume baseline failed if no initial_answer provided
                final_score = self.vqa_score(model_answer, target_answer)
            
            self.baseline_scores.append(baseline_score)
            self.scores.append(final_score)
            self.baseline_by_type[answer_type].append(baseline_score)
            self.by_type[answer_type].append(final_score)
            
            # Create majority vote message based on VQA score
            if final_score > 0.5:
                majority_vote = f'VQA Score: {final_score:.3f} [Correct]'
                if args['inference']['verbose']:
                    print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
            else:
                majority_vote = f'VQA Score: {final_score:.3f} [Incorrect]'
                if args['inference']['verbose']:
                    print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')
                    
            return majority_vote
            
        # Fall back to original LLM grading logic if VQA inputs not available
        elif len(grades) > 0:
            count_match_correct = 0
            for grade in grades:
                grade = grade.lower()
                if re.search(r'\[correct]', grade) or (re.search("correct", grade) and not re.search("incorrect", grade)):
                    count_match_correct += 1
            
            match_correct = True if count_match_correct >= 2 else False
            score = 1.0 if match_correct else 0.0
            
            if not match_baseline_failed:
                self.baseline_scores.append(score)
                self.scores.append(score)
            else:
                self.baseline_scores.append(0.0)  # Baseline failed
                self.scores.append(score)
            
            self.baseline_by_type[answer_type].append(self.baseline_scores[-1])
            self.by_type[answer_type].append(score)
            
            if match_correct:
                majority_vote = 'Majority vote is [Correct] with a score of ' + str(count_match_correct)
                if args['inference']['verbose']:
                    print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
            else:
                majority_vote = 'Majority vote is [Incorrect] with a score of ' + str(count_match_correct)
                if args['inference']['verbose']:
                    print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')
                    
            return majority_vote
        else:
            # No grading information available - mark as incorrect
            self.baseline_scores.append(0.0)
            self.scores.append(0.0)
            self.baseline_by_type[answer_type].append(0.0)
            self.by_type[answer_type].append(0.0)
            
            majority_vote = 'No grading information available [Incorrect]'
            if args['inference']['verbose']:
                print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')
                
            return majority_vote

    def average_score(self):
        """Calculate and return the average score of the grades."""
        if self.count_total == 0:
            return 0, 0, None

        baseline_accuracy = sum(self.baseline_scores) / len(self.baseline_scores) if self.baseline_scores else 0
        final_accuracy = sum(self.scores) / len(self.scores) if self.scores else 0

        # Count correct answers for compatibility
        count_correct_baseline = sum(1 for score in self.baseline_scores if score > 0.5)
        count_correct = sum(1 for score in self.scores if score > 0.5)

        stats = {
            'count_correct': count_correct,
            'count_incorrect': self.count_total - count_correct,
            'count_correct_baseline': count_correct_baseline,
            'count_incorrect_baseline': self.count_total - count_correct_baseline,
            'count_total': self.count_total,
            'by_type_accuracy': {t: sum(scores)/len(scores) if scores else 0 
                               for t, scores in self.by_type.items()},
            'baseline_by_type_accuracy': {t: sum(scores)/len(scores) if scores else 0 
                                        for t, scores in self.baseline_by_type.items()}
        }
        
        return baseline_accuracy, final_accuracy, stats

    def accumulate_grades_simple(self, args, grades):
        """Simple grade accumulation for backward compatibility"""
        count_match_correct = 0
        for grade in grades:
            if re.search(r'\[Correct\]', grade):
                count_match_correct += 1
        
        match_correct = True if count_match_correct >= 2 else False
        score = 1.0 if match_correct else 0.0
        
        self.count_total += 1
        self.baseline_scores.append(score)
        self.scores.append(score)

        if match_correct:
            majority_vote = 'Majority vote is [Correct] with a score of ' + str(count_match_correct)
            if args['inference']['verbose']:
                print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
        else:
            majority_vote = 'Majority vote is [Incorrect] with a score of ' + str(count_match_correct)
            if args['inference']['verbose']:
                print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')

        return majority_vote

    def average_score_simple(self):
        """Calculate and return the average score of the grades."""
        if self.count_total == 0:
            return 0, None

        accuracy = sum(self.scores) / len(self.scores) if self.scores else 0
        count_correct = sum(1 for score in self.scores if score > 0.5)

        stats = {
            'count_correct': count_correct,
            'count_incorrect': self.count_total - count_correct,
            'count_total': self.count_total
        }
        return accuracy, stats


def calculate_iou_batch(a, b):
    """
    Vectorized calculation of IoU for pairs of bounding boxes in a and b.
    Parameters:
    - a: PyTorch tensor of shape (N, 4), representing bounding boxes.
    - b: PyTorch tensor of shape (M, 4), representing bounding boxes.
    Returns:
    - iou: PyTorch tensor of shape (M, N), IoU values.
    """
    # Expand dimensions to support broadcasting: (N, 1, 4) with (1, M, 4)
    a = a.unsqueeze(1)  # Shape: (N, 1, 4)
    b = b.unsqueeze(0)  # Shape: (1, M, 4)
    print('a', a.shape, a, 'b', b.shape, b)

    # Calculate intersection coordinates
    max_xy = torch.min(a[..., 2:], b[..., 2:])
    min_xy = torch.max(a[..., :2], b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[..., 0] * inter[..., 1]

    # Calculate areas
    a_area = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    b_area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    # Calculate union
    union = a_area + b_area - intersection

    # Compute IoU
    iou = intersection / union
    print('iou', iou.shape)

    return iou


def filter_boxes_pytorch(a, b, iou_threshold=0.5):
    """
    Filters boxes in b based on IoU threshold with boxes in a using PyTorch.
    Parameters:
    - a, b: PyTorch tensors of shapes (N, 4) and (M, 4) respectively.
    - iou_threshold: float, threshold for filtering.
    Returns:
    - filtered_b: PyTorch tensor of filtered bounding boxes from b.
    """
    iou = calculate_iou_batch(a, b)  # Shape: (M, N)
    # Check if any IoU value exceeds the threshold for each box in b
    max_iou, _ = torch.max(iou, dim=0)
    keep = max_iou > iou_threshold
    print('b', b.shape, 'keep', keep.shape, keep, 'iou', max_iou)
    return b[keep]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # ax.imshow(img)
    plt.imsave('test_images/masks.jpg', img)


def plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename):
    from groundingdino.util.inference import annotate

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[:, :, [2, 1, 0]]  # BGR2RGB
    plt.imsave('test_images/bboxes' + filename + '.jpg', annotated_frame)


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def load_answer_list(file_path):
    """
    This function contains all the possible words in ground-truth answers in the VQA v2 dataset.
    """
    # Load the answer list from the JSON file
    with open(file_path, 'r') as file:
        answer_list = json.load(file)
    return answer_list


def save_output_predictions_vqav2(question_id, model_answer, answer_list, split='test', verbose=False):
    """
    This function formats the model answers to the VQA-v2 required format
    for close-sourced evaluation on test and test-dev datasets
    """

    def filter_response(response, answer_list):
        """
        Filters a response from an LLM to only include words that are in the provided answer list.

        Parameters:
        - response (str): The text response from the LLM.
        - answer_list (list): A list of strings containing acceptable answers.

        Returns:
        - str: A filtered response containing only words from the answer_list.
        """
        # Tokenize the response into words
        response_words = response.split()

        # Filter words based on the answer list
        filtered_words = [word for word in response_words if word in answer_list]

        # Join the filtered words back into a string
        filtered_response = ' '.join(filtered_words)

        return filtered_response

    # Regular expression to find sentences after '[Answer]' or '[Reattempted Answer]'
    extracted_answer = re.search(r"\s*\[Answer\](.*)|\s*\[Reattempted Answer\](.*)", model_answer, re.DOTALL)

    if extracted_answer:
        extracted_answer = extracted_answer.group()
        # Handling both '[Answer]' and '[Reattempted Answer]'
        if "[Answer]" in extracted_answer:
            extracted_answer = extracted_answer.replace("[Answer]", "").strip()
        elif "[Reattempted Answer]" in extracted_answer:
            extracted_answer = extracted_answer.replace("[Reattempted Answer]", "").strip()

    # Filter the extracted response using the answer list
    try:
        filtered_response = filter_response(extracted_answer, answer_list)
    except:
        filtered_response = ""

    result = {
        "question_id": question_id.item(),
        "answer": filtered_response if filtered_response else extracted_answer
    }
    if verbose:
        print(result)

    saved_file_name = 'outputs/submit_vqav2_' + split + '_4.json'

    # Check if the file exists and is not empty
    if os.path.exists(saved_file_name) and os.path.getsize(saved_file_name) > 0:
        # Read the existing data
        with open(saved_file_name, 'r') as f:
            data = json.load(f)
            data.append(result)  # Append the new result
    else:
        data = [result]  # Start a new list if the file doesn't exist or is empty

    # Write back the updated data list
    with open(saved_file_name, 'w') as f:
        json.dump(data, f, indent=2)


def write_response_to_json(question_id, response_dict, output_response_filename):
    # Check if the JSON file already exists
    if os.path.exists(output_response_filename):
        # Read the existing content
        with open(output_response_filename, 'r') as file:
            data = json.load(file)
    else:
        # Initialize an empty list if the file doesn't exist
        data = {}

    # Append the new response
    data[str(question_id.item())] = response_dict

    # Write the updated data back to the file
    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=2)


def record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename):
    # Assuming the JSON file exists at this point
    with open(output_response_filename, 'r') as file:
        data = json.load(file)

    # Add the accuracy to the JSON data
    data['baseline_accuracy'] = str(baseline_accuracy)
    data['final_accuracy'] = str(final_accuracy)
    data['stats'] = stats

    # Write the updated data back to the file
    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=2)
