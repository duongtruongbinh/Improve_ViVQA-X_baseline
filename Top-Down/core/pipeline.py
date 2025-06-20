# Top-Down/core/pipeline.py
import os
import json
import yaml
import logging
from openai import OpenAI
from tqdm import tqdm

from .agents import ResponderAgent, SeekerAgent, IntegratorAgent

# --- Helper Functions ---

def load_config(config_path):
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_api_client(config):
    """Sets up and validates the OpenAI client from a key file specified in the config."""
    key_file_path = config['model'].get('api_key_file')
    if not key_file_path:
        raise ValueError("Config error: 'api_key_file' not specified in the model configuration.")
    
    try:
        with open(key_file_path, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError(f"API key file '{key_file_path}' is empty.")
        return OpenAI(api_key=api_key)
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at path: {key_file_path}")

def load_dataset(config):
    """Loads the questions and annotations based on the configured split."""
    # Check if this is ViVQA-X format
    if config.get('vivqax', {}).get('format') == 'vivqax':
        return load_vivqax_dataset(config)
    
    # Original VQA-v2 format
    split = config['inference']['dataset_split']
    paths = config['dataset_paths']
    
    key_prefix = split.replace('-', '_')
    q_file = paths[f'{key_prefix}_questions_file']
    a_file = paths.get(f'{key_prefix}_annotations_file') # Annotations might not exist for test splits

    logging.info(f"Loading questions for split '{split}' from {q_file}...")
    with open(q_file, 'r') as f:
        questions_data = json.load(f)['questions']
    
    annotations = {}
    if a_file and os.path.exists(a_file):
        logging.info(f"Loading annotations for split '{split}' from {a_file}...")
        with open(a_file, 'r') as f:
            data = json.load(f)
            # Handle cases where the JSON root is the list itself, or it's wrapped in a dict
            annotations_data = data.get('annotations')
            if annotations_data is None and isinstance(data, list):
                annotations_data = data
            elif annotations_data is None:
                annotations_data = [] # Failed to find annotations
                logging.warning(f"Could not find 'annotations' key in {a_file}. Proceeding without annotations.")

        # Create a lookup map for question_id -> most common answer
        for ann in annotations_data:
            # Simple VQA eval: consider the most frequent answer as ground truth
            answers = [ans['answer'] for ans in ann['answers']]
            if answers:
                annotations[ann['question_id']] = max(set(answers), key=answers.count)

    return questions_data, annotations

def load_vivqax_dataset(config):
    """Loads ViVQA-X dataset with Vietnamese questions and answers."""
    split = config['inference']['dataset_split']
    paths = config['dataset_paths']
    
    # Get the appropriate file for the split
    file_key = f'{split}_file'
    data_file = paths[file_key]
    
    logging.info(f"Loading ViVQA-X data for split '{split}' from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert ViVQA-X format to pipeline format
    questions_data = []
    annotations = {}
    
    for item in data:
        # Convert to pipeline format
        question_entry = {
            'question_id': item['question_id'],
            'question': item['question'],
            'image_id': int(item['image_id'])
        }
        questions_data.append(question_entry)
        
        # Store ground truth answer
        annotations[item['question_id']] = item['answer']
    
    logging.info(f"Loaded {len(questions_data)} ViVQA-X questions with answers")
    return questions_data, annotations

# --- Main SIRI Pipeline ---

def run_siri_pipeline(config_path: str, use_vllm: bool = True):
    """
    Orchestrates the full SIRI (Seeker, Integrator, Responder) pipeline.
    Now supports both OpenAI and local vLLM backends.
    """
    config = load_config(config_path)
    
    # 1. Initialize Agents with simple backend selection
    if use_vllm:
        logging.info("🚀 Initializing SIRI agents with local vLLM backend...")
    else:
        logging.info("🌐 Initializing SIRI agents with OpenAI backend...")
    
    # Use the simplified backend manager
    responder = ResponderAgent(use_vllm=use_vllm)
    seeker = SeekerAgent(responder=responder, use_vllm=use_vllm)
    integrator = IntegratorAgent(responder)

    # 2. Load Data
    questions, annotations = load_dataset(config)
    num_questions = config['inference']['num_questions']
    if num_questions > 0:
        questions = questions[:num_questions]
    
    split = config['inference']['dataset_split']
    
    # Handle ViVQA-X format (uses single images_dir)
    if config.get('vivqax', {}).get('format') == 'vivqax':
        image_dir = config['dataset_paths']['images_dir']
        image_prefix = 'val2014'  # ViVQA-X uses COCO val2014 images
    else:
        # Original VQA-v2 format
        if 'val' in split:
            image_dir = config['dataset_paths']['val_images_dir']
            image_prefix = 'val2014'
        elif 'test' in split:
            image_dir = config['dataset_paths']['test_images_dir']
            image_prefix = 'test2015'
        else:
            raise ValueError(f"Could not determine image directory for split: {split}")

    # 3. Run Pipeline
    all_results = []
    correct_count = 0
    evaluated_count = 0
    
    for item in tqdm(questions, desc="SIRI Pipeline Processing"):
        question_id = item['question_id']
        question_text = item['question']
        
        # Construct image path based on format
        if config.get('vivqax', {}).get('format') == 'vivqax':
            # ViVQA-X uses direct image names like COCO_val2014_000000393271.jpg
            image_path = os.path.join(image_dir, f"COCO_{image_prefix}_{str(item['image_id']).zfill(12)}.jpg")
        else:
            # Original VQA-v2 format
            image_path = os.path.join(image_dir, f"COCO_{image_prefix}_{str(item['image_id']).zfill(12)}.jpg")

        logging.info(f"\n--- Processing Question ID: {question_id} ---")
        
        # Step 1 (Responder): Get initial candidates and caption
        initial_response = responder.generate_initial_response(question_text, image_path)
        answer_candidates = initial_response.get('answer_candidates', [])
        caption = initial_response.get('caption', '')
        
        # Step 2 (Seeker): Build the Multi-View Knowledge Base
        mvkv = seeker.build_mvkv(question_text, image_path, answer_candidates, caption)

        # Step 3 (Integrator): Conduct weighted voting to get the final answer
        final_answer = integrator.conduct_weighted_voting(question_text, image_path, answer_candidates, mvkv)
        
        # 4. Evaluate and Store Results
        ground_truth = annotations.get(question_id)
        is_correct = None # Use None when not applicable
        if ground_truth is not None:
            evaluated_count += 1
            is_correct = final_answer.lower().strip() == ground_truth.lower().strip()
            if is_correct:
                correct_count += 1
        
        result_entry = {
            "question_id": question_id,
            "question": question_text,
            "image_path": image_path,
            "ground_truth_answer": ground_truth,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "explainability_trace": {
                "initial_caption": caption,
                "initial_answer_candidates": answer_candidates,
                "multi_view_knowledge_base": mvkv
            }
        }
        all_results.append(result_entry)

    # 5. Save Outputs
    output_dir = os.path.dirname(config['inference']['output_file'])
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config['inference']['output_file'], 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"Pipeline complete. Full results saved to {config['inference']['output_file']}")

    # Write summary
    accuracy = (correct_count / evaluated_count) * 100 if evaluated_count > 0 else 0
    with open(config['inference']['summary_file'], 'w') as f:
        f.write("SIRI Pipeline Final Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Processed {len(questions)} questions.\n")
        f.write(f"Evaluated {evaluated_count} questions with annotations.\n")
        if evaluated_count > 0:
            f.write(f"Final Accuracy: {accuracy:.2f}% ({correct_count}/{evaluated_count})\n")
        else:
            f.write("Accuracy not calculated (no annotations found for the processed questions).\n")
    
    logging.info(f"Summary saved to {config['inference']['summary_file']}")