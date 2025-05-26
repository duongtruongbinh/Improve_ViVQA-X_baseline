import json
import os
import openai # For VLLM API calls
from tqdm import tqdm
import re
import sys
from collections import defaultdict # For per-type accuracy tracking

# Try to import image processing utility
try:
    from .image_utils import process_image_for_vlm_agent
    print("Found and imported 'process_image_for_vlm_agent' from 'image_utils.py'.")
    IMAGE_UTILS_AVAILABLE = True
except ImportError:
    print("ERROR: 'image_utils.py' not found. 'process_image_for_vlm_agent' will not be available.")
    print("Ensure 'image_utils.py' is in the same directory or PYTHONPATH.")
    IMAGE_UTILS_AVAILABLE = False
    def process_image_for_vlm_agent(image_path_or_cv2_image, app_config_ref, bbox=None):
        raise ImportError("Actual process_image_for_vlm_agent not loaded because image_utils.py is missing.")

# --- VLLM Configuration ---
VLLM_API_URL = "http://localhost:8005/v1"
VLLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VLLM_API_KEY = "EMPTY"
VLLM_TEMPERATURE = 0.0 # Prefer consistent results for VQA

# --- General Configuration ---
MAX_QUESTIONS_TO_PROCESS = 10000 # Set to an int (e.g., 10) for quick testing, None for all
VAL_IMAGE_DIR = "/mnt/VLAI_data/COCO_Images/val2014"
VAL_JSON_PATH = "/mnt/VLAI_data/VQAv2/minival.json"
OUTPUT_DIR = "../../results_vllm_base64_final"
PROCESSING_CHUNK_SIZE = 5

client = openai.OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_URL,
)

# --- Utility and Evaluation Functions (integrated from check.py logic) ---
w2n = None
try:
    from word2number import w2n
except ImportError:
    print("Warning (check.py integration): 'word2number' library not found.")

nltk_module = None
lemmatizer = None
word_tokenize_func = None
all_nltk_resources_available = False
try:
    import nltk
    nltk_module = nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    word_tokenize_func = word_tokenize
    nltk_resources_to_check = {
        'wordnet': 'corpora/wordnet.zip',
        'omw-1.4': 'corpora/omw-1.4.zip',
        'punkt': 'tokenizers/punkt.zip'
    }
    resources_actually_available_count = 0
    for resource_name, resource_path_fragment in nltk_resources_to_check.items():
        try:
            nltk_module.data.find(resource_path_fragment.replace('.zip', ''))
            resources_actually_available_count += 1
        except LookupError:
            print(f"Warning: NLTK resource '{resource_name}' not found. Attempting download...")
            try:
                nltk_module.download(resource_name)
                nltk_module.data.find(resource_path_fragment.replace('.zip', ''))
                print(f"Successfully downloaded '{resource_name}'.")
                resources_actually_available_count += 1
            except Exception as e_download:
                print(f"ERROR: Could not download or verify NLTK resource '{resource_name}': {e_download}")
    
    if resources_actually_available_count == len(nltk_resources_to_check):
        all_nltk_resources_available = True
        lemmatizer = WordNetLemmatizer()
        print("All necessary NLTK resources are ready.")
    else:
        print("Warning: Some NLTK resources are not available or could not be downloaded. Lemmatization may be affected.")
except ImportError:
    print("Warning: 'nltk' library not found. Lemmatization disabled.")

def clean_answer_for_comparison(answer_text: str) -> str:
    if not isinstance(answer_text, str): return ""
    text_to_clean = answer_text.strip()
    match = re.match(r"^(?:\[Answer\]|\[Reattempted Answer\])\s*(.*)", text_to_clean, re.IGNORECASE)
    if match: text_to_clean = match.group(1)
    return text_to_clean.strip().lower()

def parse_number(text: str):
    if text is None: return None
    try: return float(text)
    except ValueError:
        if w2n:
            try: return float(w2n.word_to_num(text))
            except ValueError: pass
        numbers_found = re.findall(r'-?\d+\.?\d*|-?\.\d+', text)
        if numbers_found:
            try: return float(numbers_found[0])
            except ValueError: return None
        return None

def get_lemmatized_tokens(text: str) -> list:
    if not all_nltk_resources_available or not lemmatizer or not word_tokenize_func:
        return [t for t in re.findall(r'\b\w+\b', text.lower()) if t] if text else []
    if not isinstance(text, str) or not text.strip(): return []
    try:
        tokens = word_tokenize_func(text.lower())
        processed_tokens = []
        for token in tokens:
            lemma = None
            if token == 'kiting': lemma = 'kite'
            elif token.endswith('ing'): lemma = lemmatizer.lemmatize(token, pos='v')
            elif token == 'wooden': lemma = 'wood'
            elif token == 'metallic': lemma = 'metal'
            else: lemma = lemmatizer.lemmatize(token, pos='n')
            if lemma: processed_tokens.append(lemma)
        return [t for t in processed_tokens if t]
    except Exception:
        return [t for t in re.findall(r'\b\w+\b', text.lower()) if t] if text else []

def calculate_token_f1(model_ans_str: str, target_ans_str: str, use_lemmatization: bool = True) -> float:
    if use_lemmatization and all_nltk_resources_available:
        processed_model_tokens = get_lemmatized_tokens(model_ans_str if model_ans_str else "")
        processed_target_tokens = get_lemmatized_tokens(target_ans_str if target_ans_str else "")
    else:
        processed_model_tokens = [t for t in (model_ans_str.lower().split() if model_ans_str else []) if t]
        processed_target_tokens = [t for t in (target_ans_str.lower().split() if target_ans_str else []) if t]
    if not processed_model_tokens and not processed_target_tokens: return 1.0
    if not processed_model_tokens or not processed_target_tokens: return 0.0
    common_tokens = set(processed_model_tokens) & set(processed_target_tokens)
    if not common_tokens: return 0.0
    precision = len(common_tokens) / len(processed_model_tokens)
    recall = len(common_tokens) / len(processed_target_tokens)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def perform_direct_accuracy_check(
    final_answer_text: str, target_answer: str, question_type: str,
    f1_threshold_other: float = 0.5 # Retained as it's part of the function signature provided
) -> tuple:
    direct_accuracy_results = {"is_correct": None, "cleaned_model_answer": "", "normalized_target_answer": "", "notes": ""}
    processed_target_answer = str(target_answer).strip() if target_answer is not None else ""
    if not processed_target_answer:
        direct_accuracy_results["notes"] = "Skipped due to missing target_answer."
        return direct_accuracy_results, [], "Direct comparison: Skipped - No Target Answer"
    
    cleaned_model_ans = clean_answer_for_comparison(final_answer_text if final_answer_text else "")
    normalized_target_ans = processed_target_answer.lower()
    is_correct = False
    match_method_note = "No check performed"
    direct_accuracy_results["cleaned_model_answer"] = cleaned_model_ans
    direct_accuracy_results["normalized_target_answer"] = normalized_target_ans

    if not cleaned_model_ans and cleaned_model_ans != "0": # "0" can be a valid answer
        is_correct = (normalized_target_ans == "" or normalized_target_ans == "none")
        match_method_note = "Model answer empty after cleaning."
    elif question_type == "yes/no":
        positive_indicators = {"yes", "yeah", "yep", "correct", "true", "affirmative", "y", "positive"}
        negative_indicators = {"no", "nope", "incorrect", "false", "negative", "n", "not"}
        model_ans_is_positive = cleaned_model_ans in positive_indicators
        model_ans_is_negative = cleaned_model_ans in negative_indicators
        target_is_positive = normalized_target_ans in positive_indicators
        target_is_negative = normalized_target_ans in negative_indicators
        if (model_ans_is_positive and target_is_positive) or \
           (model_ans_is_negative and target_is_negative):
            is_correct = True; match_method_note = "Yes/No variants (standard)."
        elif not (target_is_positive or target_is_negative) and not (model_ans_is_positive or model_ans_is_negative):
            if cleaned_model_ans == normalized_target_ans: is_correct = True; match_method_note = "Yes/No variants (exact on non-standard)."
            else:
                f1 = calculate_token_f1(cleaned_model_ans, normalized_target_ans, use_lemmatization=all_nltk_resources_available)
                if f1 >= 0.5: is_correct = True; match_method_note = f"Yes/No variants (F1 on non-standard: {f1:.2f})."
                else: match_method_note = f"Yes/No variants (non-standard no match, F1={f1:.2f})."
        else: match_method_note = "Yes/No variants (mismatch)."
    elif question_type == "number":
        model_num = parse_number(cleaned_model_ans); target_num = parse_number(normalized_target_ans)
        if model_num is not None and target_num is not None:
            # Tolerance for float comparison
            is_correct = abs(model_num - target_num) < 1e-5 if isinstance(model_num, float) or isinstance(target_num, float) else int(model_num) == int(target_num)
            match_method_note = f"Number parsed (model={model_num}, target={target_num})."
        else: 
            is_correct = (cleaned_model_ans == normalized_target_ans) # Fallback
            match_method_note = f"Number string match (parsed: model={model_num}, target={target_num})."
    elif question_type == "other":
        if cleaned_model_ans == normalized_target_ans: is_correct = True; match_method_note = "Other: Exact match."
        else:
            f1 = calculate_token_f1(cleaned_model_ans, normalized_target_ans, use_lemmatization=all_nltk_resources_available)
            if f1 >= f1_threshold_other: 
                is_correct = True; match_method_note = f"Other: Token F1 ({'lemma' if all_nltk_resources_available else 'no lemma'}) ({f1:.2f} >= {f1_threshold_other})."
            else: 
                match_method_note = f"Other: No match (Exact & F1 ({'lemma' if all_nltk_resources_available else 'no lemma'}) < {f1_threshold_other} failed, F1={f1:.2f})."
    else: 
        is_correct = (cleaned_model_ans == normalized_target_ans)
        match_method_note = f"Unknown QType ({question_type}): Exact match."
    
    direct_accuracy_results["is_correct"] = is_correct
    direct_accuracy_results["notes"] = match_method_note
    final_note_summary = match_method_note.split('-')[0].strip() if '-' in match_method_note else match_method_note
    grades_output = [f"[Direct ({question_type}): {'Correct' if is_correct else 'Incorrect'} - {final_note_summary}]"]
    majority_vote_output = f"Direct ({question_type}): {'Correct' if is_correct else 'Incorrect'}"
    return direct_accuracy_results, grades_output, majority_vote_output

def batch_inference_vllm_with_base64(questions, image_paths):
    system_instruction = (
        "You are a Visual Question Answering (VQA) system. "
        "Use only the information visible in the image. "
        "Answer each question with a single word or short phrase whenever possible. "
        "Always use exactly this output format, with no extra text:\n\n"
        "Answer: <your concise answer>\n\n"
        "Examples:\n"
        "Question: What is the man doing?\nAnswer: skiing\n\n"
        "Question: What material is the table made of?\nAnswer: wood\n\n"
        "Question: Which animal is shown in the picture?\nAnswer: giraffe\n\n"
        "Question: What color is the car?\nAnswer: red\n\n"
        "Question: How many people are there?\nAnswer: two"
    )
    predicted_answers = []

    if not IMAGE_UTILS_AVAILABLE:
        print("CRITICAL ERROR: image_utils.py not loaded. Cannot process images for VLLM.")
        return [""] * len(questions)

    for q_text, img_path in zip(questions, image_paths):
        if not os.path.exists(img_path):
            print(f"Error: Image file {img_path} not found for question '{q_text}'. Skipping.")
            predicted_answers.append("")
            continue
        try:
            image_data_url = process_image_for_vlm_agent(image_path_or_cv2_image=img_path, app_config_ref=None)
        except Exception as e:
            print(f"Error processing image {img_path} with image_utils: {e}")
            predicted_answers.append("")
            continue

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": q_text},
                ],
            },
            {"role": "assistant", "content": "Answer:"} # Guide model to start with "Answer:"
        ]
        try:
            response = client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=messages,
                max_tokens=30,
                temperature=VLLM_TEMPERATURE,
            )
            raw_answer = response.choices[0].message.content.strip()
            answer = raw_answer.split("\n")[0].strip()
            if answer.lower().startswith("answer:"):
                answer = answer.split(":", 1)[-1].strip()
            predicted_answers.append(answer)
        except Exception as e:
            print(f"Error calling VLLM API for question '{q_text}': {e}")
            predicted_answers.append("")
    return predicted_answers

def main():
    print(f"Starting VQA processing...")
    print(f"Using VLLM model: {VLLM_MODEL_NAME} at {VLLM_API_URL} (with base64 images)")

    if not IMAGE_UTILS_AVAILABLE:
        print("Cannot continue because image_utils.py is not available for image processing.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_file_name = f"VQAv2_{VLLM_MODEL_NAME.split('/')[-1]}_vllm_base64_eval.json"
    result_file_path = os.path.join(OUTPUT_DIR, result_file_name)

    try:
        with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
            val_data_full = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Validation JSON file not found at: {VAL_JSON_PATH}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not read validation JSON file: {VAL_JSON_PATH}")
        return

    if MAX_QUESTIONS_TO_PROCESS is not None and MAX_QUESTIONS_TO_PROCESS > 0 and MAX_QUESTIONS_TO_PROCESS < len(val_data_full):
        val_data = val_data_full[:MAX_QUESTIONS_TO_PROCESS]
        print(f"Processing the first {MAX_QUESTIONS_TO_PROCESS} questions.")
    else:
        val_data = val_data_full
        print(f"Processing all {len(val_data)} questions from the validation file.")
        if MAX_QUESTIONS_TO_PROCESS is not None and MAX_QUESTIONS_TO_PROCESS <= 0:
            print("MAX_QUESTIONS_TO_PROCESS is set to 0 or negative, no questions will be processed.")
            return

    final_results_with_eval = []
    correct_predictions_count = 0
    total_evaluated_questions = 0
    
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)

    for i in tqdm(range(0, len(val_data), PROCESSING_CHUNK_SIZE), desc="Processing question batches"):
        chunk = val_data[i:i + PROCESSING_CHUNK_SIZE]
        questions = [item['question'] for item in chunk]
        image_paths = [os.path.join(VAL_IMAGE_DIR, f"COCO_val2014_{item['image_id']:012d}.jpg") for item in chunk]
        
        predicted_batch_answers = batch_inference_vllm_with_base64(questions, image_paths)

        for item_data, model_predicted_answer in zip(chunk, predicted_batch_answers):
            if not item_data.get('question_id'): continue

            total_evaluated_questions += 1
            gt_answers_list = item_data['answers']
            question_type_from_data = item_data['answer_type'].lower()
            current_question_id = item_data['question_id']

            is_item_correct = False
            best_match_notes = "No match with any ground truth answers."

            for gt_entry in gt_answers_list:
                gt_text = gt_entry['answer']
                accuracy_results, _, _ = perform_direct_accuracy_check(
                    final_answer_text=model_predicted_answer,
                    target_answer=gt_text,
                    question_type=question_type_from_data
                )
                if accuracy_results.get("is_correct"):
                    is_item_correct = True
                    best_match_notes = accuracy_results.get("notes", "Exact match.")
                    break 
            
            if is_item_correct:
                correct_predictions_count += 1
            
            total_by_type[question_type_from_data] += 1
            if is_item_correct:
                correct_by_type[question_type_from_data] += 1
            
            print(f"QID {current_question_id}: Pred='{model_predicted_answer}', Correct={is_item_correct}, Note='{best_match_notes}'")

            final_results_with_eval.append({
                'question_id': current_question_id,
                'question': item_data['question'],
                'predict_ans': model_predicted_answer,
                'gt_ans': gt_answers_list,
                'answer_type': question_type_from_data,
                'is_correct_by_check': is_item_correct,
                'eval_notes': best_match_notes
            })

    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_results_with_eval, f, indent=2, ensure_ascii=False)
    print(f"\nResults (including evaluation) saved to: {result_file_path}")

    if total_evaluated_questions > 0:
        overall_accuracy = (correct_predictions_count / total_evaluated_questions) * 100
        print(f"\n--- Overall Evaluation Statistics ---")
        print(f"Total questions evaluated: {total_evaluated_questions}")
        print(f"Correct predictions: {correct_predictions_count}")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

        print(f"\n--- Accuracy Statistics By Question Type ---")
        sorted_q_types = sorted(total_by_type.keys()) # Sort for consistent output
        for q_type in sorted_q_types:
            if total_by_type[q_type] > 0:
                acc_type = (correct_by_type[q_type] / total_by_type[q_type]) * 100
                print(f"Type '{q_type}': {acc_type:.2f}% (Correct: {correct_by_type[q_type]}, Total: {total_by_type[q_type]})")
            else:
                # Should be rare if q_type comes from item_data and total_evaluated_questions > 0
                print(f"Type '{q_type}': 0.00% (Correct: 0, Total: 0)")
    else:
        print("\nNo questions were evaluated.")

if __name__ == "__main__":
    main()