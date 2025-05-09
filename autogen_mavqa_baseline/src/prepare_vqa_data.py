import json
import os
import glob
import random
from collections import defaultdict, Counter
import re
from tqdm import tqdm
import sys

try:
    from transformers import XLMRobertaTokenizer
except ImportError:
    print("Error: Hugging Face transformers library not found.")
    print("Please install it using: pip install transformers")
    sys.exit(1)


# --- Absolute Paths ---
# Paths configured based on your existing directory structure and Multi-Agent VQA config.
IMAGE_ROOT = '/mnt/VLAI_data/COCO_Images'
RAW_JSON_PATH = '/mnt/VLAI_data/VQAv2'
OUTPUT_PATH = '/mnt/VLAI_data/VQAv2'
TOKENIZER_PATH = '/mnt/VLAI_data/models/beit3/beit3.spm'


# --- Helper functions ---
def _write_data_into_jsonl(items, jsonl_file):
    """Writes a list of dictionary items into a JSONL file."""
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print(f"Successfully wrote {len(items)} items to {jsonl_file}")

def normalize_word(word):
    """Normalizes a word for VQA answer processing."""
    word = word.lower()
    word = word.replace(',', '').replace(';', '').replace('.', '').replace('?', '-').replace('(', '').replace(')', '')
    word = word.replace('"', '').replace("'", '').replace('-', ' ').replace('/', ' ')
    word = word.split(' ')
    word = [w.strip() for w in word]
    word = [w for w in word if len(w) > 0]
    word = ' '.join(word)
    return word

def get_score(occurences):
    """Calculates VQA accuracy score based on answer occurrences."""
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0

# --- Logic to create VQA-v2 JSONL index files ---
def create_vqa_jsonl_indices(image_root_path, raw_json_path, output_path, tokenizer):
    """
    Processes raw VQA-v2 JSON data and creates .jsonl index files,
    including the vqa.rest_val.jsonl and answer2label.txt.
    """
    print(f"--- Creating VQA-v2 JSONL indices ---")

    try:
        print("Loading raw question files...")
        with open(os.path.join(raw_json_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(raw_json_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(raw_json_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(raw_json_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]
        print("Raw question files loaded.")
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Raw question file not found: {e}")
        print(f"Please ensure the raw VQA-v2 question JSONs are in {raw_json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Error decoding JSON in a raw question file: {e}")
        sys.exit(1)

    try:
        print("Loading raw annotation files (train/val)...")
        with open(os.path.join(raw_json_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(raw_json_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]
        print("Raw annotation files loaded.")
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Raw annotation file not found: {e}")
        print(f"Please ensure the raw VQA-v2 annotation JSONs (train/val) are in {raw_json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Error decoding JSON in a raw annotation file: {e}")
        sys.exit(1)

    annotations_dict = dict()

    for split, questions in zip(
        ["train", "val", "test", "test-dev"],
        [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions, desc=f"Tokenizing {split} questions"):
            question_text = q["question"]
            tokens = tokenizer.tokenize(question_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            _annot[q["image_id"]][q["question_id"]] = {
                "question": question_text,
                "token_ids": token_ids,
            }
        annotations_dict[split] = _annot

    all_major_answers = []
    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
         for a in tqdm(annots, desc=f"Collecting answers for vocab ({split})"):
             all_major_answers.append(a["multiple_choice_answer"])

    all_major_answers = [normalize_word(word) for word in all_major_answers]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations_dict[split]
        for a in tqdm(annots, desc=f"Adding labels/scores for {split} annotations"):
            answers = a["answers"]
            answer_count = {}
            for answer_item in answers:
                 answer_ = normalize_word(answer_item["answer"])
                 answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            for answer_text, count in answer_count.items():
                if answer_text in ans2label:
                    labels.append(ans2label[answer_text])
                    score = get_score(count)
                    scores.append(score)

            image_id = a["image_id"]
            question_id = a["question_id"]

            if image_id in _annot and question_id in _annot[image_id]:
                 _annot[image_id][question_id]["labels"] = labels
                 _annot[image_id][question_id]["scores"] = scores

    for split in ["train", "val", "test", "test-dev"]:
        filtered_annot = dict()
        for img_id, q_dict in annotations_dict[split].items():
             new_q_dict = dict()
             for qid, q_info in q_dict.items():
                 if split in ["train", "val"]:
                     if "labels" in q_info and len(q_info["labels"]) > 0:
                         new_q_dict[qid] = q_info
                 else:
                     new_q_dict[qid] = q_info

             if len(new_q_dict) > 0:
                 filtered_annot[img_id] = new_q_dict

        annotations_dict[split] = filtered_annot

    split2items = {}
    for split in ["train", "val", "test", "test-dev"]:
        annot = annotations_dict[split]
        split_name_map = {
            "train": "train2014",
            "val": "val2014",
            "test": "test2015",
            "test-dev": "test2015",
        }
        split_folder_name = split_name_map[split]

        image_files_pattern = os.path.join(image_root_path, split_folder_name, "*.jpg")
        all_image_paths_in_folder = glob.glob(image_files_pattern)

        items = []
        print(f"Creating items for {split} split...")
        for img_path in tqdm(all_image_paths_in_folder):
             image_id = int(os.path.basename(img_path).split("_")[-1].split(".")[0])

             if image_id in annot:
                 _annot = annot[image_id]
                 for qid, q_info in _annot.items():
                     relative_image_path = os.path.join(split_folder_name, os.path.basename(img_path))

                     item = {
                         "image_path": relative_image_path,
                         "text_segment": q_info["token_ids"],
                         "qid": qid,
                     }

                     if "labels" in q_info:
                          item["labels"] = q_info["labels"]
                          item["scores"] = q_info["scores"]

                     items.append(item)
        split2items[split] = items
        _write_data_into_jsonl(items=items, jsonl_file=os.path.join(output_path, f"vqa.{split}.jsonl"))

    val_items = split2items["val"]
    val_image2items = defaultdict(list)
    for item in val_items:
        val_image2items[item["image_path"]].append(item)

    print(f"\nSplitting original val set:")
    print(f"  Original val set has {len(val_image2items)} images and {len(val_items)} pairs.")

    val_images = list(val_image2items.keys())
    random.seed(0)
    random.shuffle(val_images)

    rest_val_images = val_images[:1000]
    trainable_val_images = val_images[1000:]

    rest_val_items = []
    trainable_val_items = []
    val1000_items = []

    for img_path in tqdm(rest_val_images, desc="Collecting rest_val items"):
        rest_val_items.extend(val_image2items[img_path])
        if val_image2items[img_path] and len(val1000_items) < 1000:
             val1000_items.append(val_image2items[img_path][0])

    for img_path in tqdm(trainable_val_images, desc="Collecting trainable_val items"):
        trainable_val_items.extend(val_image2items[img_path])

    _write_data_into_jsonl(items=trainable_val_items, jsonl_file=os.path.join(output_path, "vqa.trainable_val.jsonl"))
    _write_data_into_jsonl(items=rest_val_items, jsonl_file=os.path.join(output_path, "vqa.rest_val.jsonl"))
    _write_data_into_jsonl(items=val1000_items, jsonl_file=os.path.join(output_path, "vqa.val1000.jsonl"))


    print(f"\nGenerated VQA-v2 splits:")
    print(f"  vqa.trainable_val.jsonl ({len(trainable_val_items)} items)")
    print(f"  vqa.rest_val.jsonl ({len(rest_val_items)} items)")
    print(f"  vqa.val1000.jsonl ({len(val1000_items)} items)")

    print(f"\n--- Creating answer2label.txt ---")
    with open(os.path.join(output_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
        for ans, label in ans2label.items():
            to_json = {
                "answer": ans,
                "label": label
            }
            writer.write("%s\n" % json.dumps(to_json))
    print(f"Created answer2label.txt with {len(ans2label)} entries at {os.path.join(output_path, 'answer2label.txt')}")

    answer_list_path = os.path.join(output_path, "answer_list.json")
    print(f"\n--- Creating answer_list.json (based on answer2label.txt vocabulary) ---")
    try:
        with open(answer_list_path, 'w', encoding='utf-8') as f:
             json.dump(label2ans, f, indent=4)
        print(f"Created answer_list.json with {len(label2ans)} entries at {answer_list_path}")
    except Exception as e:
        print(f"Warning: Could not create answer_list.json: {e}")


# --- Logic to convert rest_val jsonl to VQA JSON format ---
def create_rest_val_json_from_jsonl(input_jsonl_path, raw_val_questions_json_path, raw_val_annotations_json_path, output_questions_json_path, output_annotations_json_path):
    """
    Converts the vqa.rest_val.jsonl into VQA JSON format for questions and annotations.
    """
    print(f"\n--- Converting {os.path.basename(input_jsonl_path)} to VQA JSON format ---")

    try:
        vqa_rest_val_items = []
        with open(input_jsonl_path, 'r') as file:
            for line in tqdm(file, desc=f"Loading {os.path.basename(input_jsonl_path)}"):
                vqa_rest_val_items.append(json.loads(line))
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input JSONL file not found at {input_jsonl_path}.")
        print(f"Please run the index creation step (create_vqa_jsonl_indices) first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Could not decode JSON from {input_jsonl_path}. File might be corrupted.")
        print(e)
        sys.exit(1)

    try:
        with open(raw_val_questions_json_path, 'r') as file:
            coco_val_questions = json.load(file)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Original val questions JSON not found at {raw_val_questions_json_path}.")
        print(f"Please ensure the raw VQA-v2 data is in place.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Could not decode JSON from {raw_val_questions_json_path}. Please check file integrity.")
        print(e)
        sys.exit(1)

    try:
        with open(raw_val_annotations_json_path, 'r') as file:
            coco_val_annotations = json.load(file)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Original val annotations JSON not found at {raw_val_annotations_json_path}.")
        print(f"Please ensure the raw VQA-v2 data is in place.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Could not decode JSON from {raw_val_annotations_json_path}. Please check file integrity.")
        print(e)
        sys.exit(1)

    rest_val_qids = {item['qid'] for item in vqa_rest_val_items}

    matched_questions = []
    print("Matching questions...")
    original_val_questions_by_qid = {q['question_id']: q for q in coco_val_questions['questions']}

    for qid in tqdm(rest_val_qids, desc="Matching questions"):
         if qid in original_val_questions_by_qid:
             matched_questions.append(original_val_questions_by_qid[qid])

    print(f"Matched {len(matched_questions)} questions.")

    os.makedirs(os.path.dirname(output_questions_json_path), exist_ok=True)
    with open(output_questions_json_path, 'w') as outfile:
        json.dump({"questions": matched_questions}, outfile)
    print(f"Stored {len(matched_questions)} matched questions in {os.path.basename(output_questions_json_path)}")

    matched_annotations_list = []
    print("Matching annotations...")
    original_val_annotations_by_qid = {ann['question_id']: ann for ann in coco_val_annotations['annotations']}

    for qid in tqdm(rest_val_qids, desc="Matching annotations"):
         if qid in original_val_annotations_by_qid:
             matched_annotations_list.append(original_val_annotations_by_qid[qid])

    print(f"Matched {len(matched_annotations_list)} annotations.")

    matched_annotations_dict = {str(ann['question_id']): ann for ann in matched_annotations_list}

    os.makedirs(os.path.dirname(output_annotations_json_path), exist_ok=True)
    with open(output_annotations_json_path, 'w') as outfile:
        json.dump(matched_annotations_dict, outfile, indent=4)
    print(f"Stored {len(matched_annotations_dict)} matched annotations in {os.path.basename(output_annotations_json_path)}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting VQA-v2 Data Preparation Script ---")

    if not os.path.exists(TOKENIZER_PATH):
         print(f"\nCRITICAL ERROR: Tokenizer file not found at the specified path: {TOKENIZER_PATH}")
         print("Please ensure the TOKENIZER_PATH variable is correct and the file exists.")
         sys.exit(1)

    if not os.path.exists(IMAGE_ROOT) or not os.path.isdir(IMAGE_ROOT):
        print(f"\nCRITICAL ERROR: Image root directory not found: {IMAGE_ROOT}")
        print("Please ensure your COCO images are placed correctly.")
        sys.exit(1)

    if not os.path.exists(RAW_JSON_PATH) or not os.path.isdir(RAW_JSON_PATH):
        print(f"\nCRITICAL ERROR: Raw JSON path not found: {RAW_JSON_PATH}")
        print("Please ensure your raw VQA JSON files are placed correctly.")
        sys.exit(1)

    try:
        tokenizer = XLMRobertaTokenizer(TOKENIZER_PATH)
        print(f"\nSuccessfully loaded tokenizer from {TOKENIZER_PATH}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not load tokenizer from {TOKENIZER_PATH}")
        print("Please check the TOKENIZER_PATH and ensure the file is valid.")
        print(e)
        sys.exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"\nEnsured output directory exists: {OUTPUT_PATH}")

    create_vqa_jsonl_indices(IMAGE_ROOT, RAW_JSON_PATH, OUTPUT_PATH, tokenizer)

    input_rest_val_jsonl = os.path.join(OUTPUT_PATH, "vqa.rest_val.jsonl")
    raw_val_questions = os.path.join(RAW_JSON_PATH, "v2_OpenEnded_mscoco_val2014_questions.json")
    raw_val_annotations = os.path.join(RAW_JSON_PATH, "v2_mscoco_val2014_annotations.json")
    output_rest_val_questions = os.path.join(OUTPUT_PATH, "v2_OpenEnded_mscoco_rest_val2014_questions.json")
    output_rest_val_annotations = os.path.join(OUTPUT_PATH, "v2_mscoco_rest_val2014_annotations.json")

    create_rest_val_json_from_jsonl(
        input_rest_val_jsonl,
        raw_val_questions,
        raw_val_annotations,
        output_rest_val_questions,
        output_rest_val_annotations
    )

    print("\n--- VQA-v2 Data Preparation Script Finished ---")
    print(f"Check the directory '{OUTPUT_PATH}' for the generated files.")
    print("These files should now match the paths specified in your Multi-Agent VQA config.yaml")