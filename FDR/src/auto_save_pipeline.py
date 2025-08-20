
# FDR/src/auto_save_pipeline.py - Auto Save Pipeline with Correct/Wrong Classification
import os
import json
import yaml
import logging
import shutil
from collections import Counter
from tqdm import tqdm
import sys
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path to allow for utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agents import VerifierAgent, StrategistAgent, SynthesizerAgent
from src.eval import EvalModule, GEvaluator

# Import BackendManager with error handling
try:
    from utils.backend_manager import BackendManager
except ImportError as e:
    logging.error(f"Failed to import BackendManager: {e}")
    BackendManager = None

# --- Helper Functions ---

def _normalize_answer(answer: str) -> str:
    """Normalize answer format for better evaluation accuracy."""
    if not answer:
        return answer

    # Remove trailing punctuation and extra whitespace
    normalized = answer.strip().rstrip('.!?').strip()

    # Handle common format variations
    if normalized.lower() in ['yes', 'no']:
        return normalized.lower()

    # For other answers, keep original case but remove trailing punctuation
    return normalized

def load_config(config_path=None):
    """Loads the YAML configuration file from unified config.yaml or specified path."""
    if config_path is None:
        # Use unified config.yaml by default
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, 'config.yaml')
    
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_directories(base_dir="auto_output"):
    """Create correct and wrong directories for saving results"""
    correct_dir = os.path.join(base_dir, "correct")
    wrong_dir = os.path.join(base_dir, "wrong")
    
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)
    
    return correct_dir, wrong_dir

def save_groundingdino_image(result, sample_dir):
    """Save GroundingDINO annotated image if available"""
    # First try to get the path from result
    groundingdino_path = result.get('groundingdino_annotated_path')

    # If not in result, try to find it based on input image path
    if not groundingdino_path:
        input_image_path = result.get('image_path', '')
        if input_image_path:
            input_dir = os.path.dirname(input_image_path)
            input_filename = os.path.basename(input_image_path)
            groundingdino_filename = f"groundingdino_{input_filename}"
            groundingdino_path = os.path.join(input_dir, groundingdino_filename)

    if groundingdino_path and os.path.exists(groundingdino_path):
        try:
            # Copy GroundingDINO annotated image to sample directory
            groundingdino_img = Image.open(groundingdino_path)
            groundingdino_img.save(os.path.join(sample_dir, "groundingdino_annotated.jpg"))
            logging.info(f"✅ Saved GroundingDINO annotated image: {groundingdino_path}")
        except Exception as e:
            logging.warning(f"Could not copy GroundingDINO image: {e}")
    else:
        logging.debug(f"GroundingDINO annotated image not found: {groundingdino_path}")

def save_sample_result(result, output_dir, sample_type, sample_id):
    """Save sample result with text and images"""
    # Create sample directory
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save text information
    text_info = {
        "id": result.get('question_id', f"sample_{sample_id}"),
        "question": result.get('question', ''),
        "answer": result.get('final_answer', ''),
        "explanation": result.get('explanation', ''),
        "evidence_set": result.get('evidence_set', []),
        "hypothesis_set": result.get('hypothesis_set', []),
        "evidence_count": result.get('evidence_count', 0),
        "hypothesis_count": result.get('hypothesis_count', 0),
        "causal_trace": result.get('causal_trace', []),
        "ground_truth": result.get('ground_truth', ''),
        "synthesis_status": result.get('synthesis_status', ''),
        "type": sample_type
    }

    with open(os.path.join(sample_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(text_info, f, indent=2, ensure_ascii=False)

    # Save input image
    input_image_path = result.get('image_path', '')
    if input_image_path and os.path.exists(input_image_path):
        try:
            input_img = Image.open(input_image_path)
            input_img.save(os.path.join(sample_dir, "input_image.jpg"))
        except Exception as e:
            logging.warning(f"Could not save input image: {e}")

    # Save GroundingDINO annotated image if available
    try:
        save_groundingdino_image(result, sample_dir)
    except Exception as e:
        logging.warning(f"Could not save GroundingDINO image: {e}")

    # Create and save output visualization
    try:
        create_output_visualization(result, sample_dir)
    except Exception as e:
        logging.warning(f"Could not create output visualization: {e}")

def create_output_visualization(result, sample_dir):
    """Create a visualization of the output with question, answer, explanation, evidence and hypothesis"""
    # Check if GroundingDINO image exists
    groundingdino_path = os.path.join(sample_dir, "groundingdino_annotated.jpg")
    has_groundingdino = os.path.exists(groundingdino_path)

    if has_groundingdino:
        # 3-panel layout: Input, GroundingDINO, Text
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # Larger size for full text
    else:
        # 2-panel layout: Input, Text
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 8))  # Larger size for full text
        ax2 = None

    # Left panel: Input image
    input_image_path = result.get('image_path', '')
    if input_image_path and os.path.exists(input_image_path):
        try:
            img = Image.open(input_image_path)
            ax1.imshow(img)
            ax1.set_title("Input Image", fontsize=14, fontweight='bold')
            ax1.axis('off')
        except Exception as e:
            ax1.text(0.5, 0.5, f"Image not available\n{str(e)}",
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Input Image (Error)", fontsize=14, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "Image not found",
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Input Image (Not Found)", fontsize=14, fontweight='bold')

    # Middle panel: GroundingDINO annotated image (if available)
    if has_groundingdino and ax2 is not None:
        try:
            groundingdino_img = Image.open(groundingdino_path)
            ax2.imshow(groundingdino_img)
            ax2.set_title("GroundingDINO Detection", fontsize=14, fontweight='bold')
            ax2.axis('off')
        except Exception as e:
            ax2.text(0.5, 0.5, f"GroundingDINO image error\n{str(e)}",
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("GroundingDINO (Error)", fontsize=14, fontweight='bold')

    # Right panel: Text information with evidence and hypothesis details
    ax3.axis('off')

    # Question
    question = result.get('question', '')
    ax3.text(0.05, 0.98, f"Question: {question}",
            fontsize=10, fontweight='bold', transform=ax3.transAxes,
            verticalalignment='top', wrap=True)

    # Answer
    answer = result.get('final_answer', '')
    ax3.text(0.05, 0.93, f"Answer: {answer}",
            fontsize=10, fontweight='bold', color='green', transform=ax3.transAxes,
            verticalalignment='top', wrap=True)

    # Ground truth
    ground_truth = result.get('ground_truth', '')
    ax3.text(0.05, 0.88, f"Ground Truth: {ground_truth}",
            fontsize=10, fontweight='bold', color='blue', transform=ax3.transAxes,
            verticalalignment='top', wrap=True)

    # Status
    status = result.get('synthesis_status', '')
    ax3.text(0.05, 0.83, f"Status: {status}",
            fontsize=10, color='orange', transform=ax3.transAxes,
            verticalalignment='top')

    # Explanation (full text)
    explanation = result.get('explanation', '')
    ax3.text(0.05, 0.75, f"Explanation: {explanation}",
            fontsize=9, transform=ax3.transAxes,
            verticalalignment='top', wrap=True)

    # Add Evidence details
    evidence_set = result.get('evidence_set', [])
    if evidence_set:
        evidence_text = "Evidence Details:\n"
        for i, evidence in enumerate(evidence_set):  # Show ALL evidence
            evidence_id = evidence.get('evidence_id', f'E{i+1}')
            answer = evidence.get('answer', 'N/A')  # Full answer, no truncation
            question = evidence.get('question', '')
            if question:
                evidence_text += f"  • {evidence_id}: Q: {question}\n"  # Full question
                evidence_text += f"    A: {answer}\n"  # Full answer
            else:
                evidence_text += f"  • {evidence_id}: {answer}\n"  # Full answer

        ax3.text(0.05, 0.60, evidence_text,
                fontsize=9, color='purple', transform=ax3.transAxes,
                verticalalignment='top')

    # Add Hypothesis details
    hypothesis_set = result.get('hypothesis_set', [])
    if hypothesis_set:
        hypothesis_text = "Hypothesis Details:\n"
        for i, hypothesis in enumerate(hypothesis_set):  # Show ALL hypothesis
            hyp_id = hypothesis.get('hypothesis_id', f'H{i+1}')
            final_answer = hypothesis.get('THEN', {}).get('final_answer', 'N/A')
            confidence = hypothesis.get('confidence_source', 0.0)
            reasoning = hypothesis.get('reasoning_description', '')

            hypothesis_text += f"  • {hyp_id}: → {final_answer} (conf: {confidence:.2f})\n"
            if reasoning:
                hypothesis_text += f"    Reasoning: {reasoning}\n"  # Full reasoning

        # Position hypothesis below evidence (dynamic positioning)
        y_pos = 0.20 if evidence_set else 0.50
        ax3.text(0.05, y_pos, hypothesis_text,
                fontsize=9, color='brown', transform=ax3.transAxes,
                verticalalignment='top')

    # Add summary counts at bottom
    evidence_count = result.get('evidence_count', 0)
    hypothesis_count = result.get('hypothesis_count', 0)

    # Dynamic positioning for summary based on content
    summary_y_pos = 0.05
    if hypothesis_set and evidence_set:
        summary_y_pos = 0.02
    elif evidence_set or hypothesis_set:
        summary_y_pos = 0.05

    ax3.text(0.05, summary_y_pos, f"Total: {evidence_count} Evidence, {hypothesis_count} Hypothesis",
            fontsize=9, color='gray', fontweight='bold', transform=ax3.transAxes,
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "output_visualization.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

def is_answer_correct(result):
    """Check if the answer is correct by comparing with ground truth"""
    final_answer = result.get('final_answer', '')
    ground_truth = result.get('ground_truth', '')
    
    if not final_answer or not ground_truth:
        return False
    
    # Normalize answers for comparison
    final_answer_norm = _normalize_answer(final_answer).lower()
    ground_truth_norm = _normalize_answer(ground_truth).lower()
    
    # Simple string matching
    return (final_answer_norm == ground_truth_norm or 
            ground_truth_norm in final_answer_norm or 
            final_answer_norm in ground_truth_norm)

# --- Main Auto Save Pipeline ---

def run_auto_save_pipeline(use_vllm: bool = True,
                          config_path: str = None,
                          active_dataset_override: str = None,
                          max_correct: int = 3,
                          max_wrong: int = 3,
                          output_base_dir: str = "auto_output"):
    """
    Run the FDR pipeline with automatic saving to correct/wrong directories.
    Stops when both correct and wrong samples reach their limits.
    
    Args:
        use_vllm: Whether to use vLLM backend (True) or OpenAI (False)
        config_path: Optional path to custom configuration file
        active_dataset_override: Override the active dataset from config
        max_correct: Maximum number of correct samples to collect
        max_wrong: Maximum number of wrong samples to collect
        output_base_dir: Base directory for output
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Initialize logging
        logging_config = config.get('logging_config', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        if logging_config.get('reduce_http_logs', True):
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
        
        logging.info("🚀 Starting Auto Save FDR Pipeline")
        logging.info(f"📁 Target: {max_correct} correct, {max_wrong} wrong samples")
        
        # Create output directories
        correct_dir, wrong_dir = create_output_directories(output_base_dir)
        logging.info(f"📂 Output directories created: {correct_dir}, {wrong_dir}")
        
        # Initialize agents
        agents_config = config.get('agents_config', {})
        
        # Verifier Agent (VLM + GroundingDINO + DAM)
        verifier_config = agents_config.get('verifier', {})
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=use_vllm,
            enable_dam=verifier_config.get('enable_dam', False),
            enable_groundingdino=verifier_config.get('enable_groundingdino', True),
            groundingdino_docker=verifier_config.get('groundingdino_docker', False),
            model_preference="vlm"
        )

        # Strategist Agent (LLM for MVKB construction + explanation generation)
        strategist_config = agents_config.get('strategist', {})
        strategist = StrategistAgent(
            model_name=config.get('backend_config', {}).get('model_name'),
            verifier=verifier,
            use_vllm=use_vllm,
            model_preference="llm"
        )
        
        # Synthesizer Logic Engine
        synthesizer = SynthesizerAgent(verifier=verifier)
        
        # Load dataset
        active_dataset = active_dataset_override or config.get('active_dataset', 'vqax')
        datasets_config = config.get('datasets', {})
        
        if active_dataset not in datasets_config:
            raise ValueError(f"Active dataset '{active_dataset}' not found in datasets config. Available: {list(datasets_config.keys())}")
        
        dataset_config = datasets_config[active_dataset]
        input_file = dataset_config.get('data_path')
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Dataset file not found: {input_file}")
        
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle different dataset formats
        dataset_format = dataset_config.get('format', 'standard')
        if dataset_format == 'vqax':
            # VQA-X format: {question_id: {question, answers, image_name, explanation}}
            dataset = []
            for question_id, item in raw_data.items():
                # Get most common answer from multiple annotators
                answers = [ans['answer'] for ans in item['answers']]
                most_common_answer = Counter(answers).most_common(1)[0][0]
                
                sample = {
                    'question_id': question_id,
                    'question': item['question'],
                    'image_name': item['image_name'],
                    'image_id': item.get('image_id', question_id.split('00')[0] if '00' in question_id else question_id),
                    'answer': most_common_answer,
                    'all_answers': answers,
                    'explanation': item.get('explanation', [])
                }
                dataset.append(sample)
            
            logging.info(f"Loaded VQA-X format: {len(dataset)} samples")
        else:
            # Standard format: [{question, image_path, answer}]
            if isinstance(raw_data, dict) and 'data' in raw_data:
                dataset = raw_data['data']
            else:
                dataset = raw_data
            
            logging.info(f"Loaded standard format: {len(dataset)} samples")
        
        # Shuffle dataset for random sampling
        processing_config = config.get('processing_config', {})
        num_samples = processing_config.get('num_samples', len(dataset))
        if num_samples > 0 and num_samples < len(dataset):
            dataset = dataset[:num_samples]  # Limit to first 100
            # Comment out: random.shuffle(dataset)
        
        # Initialize counters
        correct_count = 0
        wrong_count = 0
        total_processed = 0
        sample_details = []  # Store detailed information for summary
        
        # Process samples until we have enough of each type
        logging.info(f"Processing samples until we have {max_correct} correct and {max_wrong} wrong...")
        
        for i, sample in enumerate(tqdm(dataset, desc="Auto Save Processing")):
            # Check if we have enough samples
            if correct_count >= max_correct and wrong_count >= max_wrong:
                logging.info(f"✅ Target reached: {correct_count} correct, {wrong_count} wrong")
                break
            
            try:
                # Extract sample data based on format
                if dataset_format == 'vqax' or dataset_format == 'vivqax':
                    # VQA-X or ViVQA-X format
                    image_name = sample['image_name']
                    image_dir = dataset_config.get('image_dir', '/mnt/VLAI_data/COCO_Images/val2014')
                    image_path = os.path.join(image_dir, image_name)
                    question = sample['question']
                    ground_truth = sample['answer']
                    question_id = sample['question_id']
                else:
                    # Standard format
                    if 'image_path' in sample:
                        image_path = sample['image_path']
                    elif 'image' in sample:
                        image_path = sample['image']
                    else:
                        logging.error(f"Sample {i}: No image path found")
                        continue
                    
                    question = sample.get('question', sample.get('question_text', ''))
                    ground_truth = sample.get('answer', sample.get('ground_truth', None))
                    question_id = sample.get('question_id', f"sample_{i}")
                
                if not question:
                    logging.error(f"Sample {i}: No question found")
                    continue
                
                # Ensure absolute image path
                if not os.path.isabs(image_path):
                    dataset_dir = os.path.dirname(input_file)
                    image_path = os.path.join(dataset_dir, image_path)
                
                if not os.path.exists(image_path):
                    logging.warning(f"Sample {i}: Image not found: {image_path}")
                    continue
                
                total_processed += 1
                logging.info(f"Processing sample {total_processed}: {question[:50]}...")
                
                # Step 1: Verifier - Generate initial context (caption)
                initial_response = verifier.generate_initial_response(question, image_path)
                answer_candidates = initial_response['answer_candidates']
                caption = initial_response['caption']

                # Check if GroundingDINO annotated image was created
                groundingdino_annotated_path = None
                input_dir = os.path.dirname(image_path)
                input_filename = os.path.basename(image_path)
                potential_groundingdino_path = os.path.join(input_dir, f"groundingdino_{input_filename}")
                if os.path.exists(potential_groundingdino_path):
                    groundingdino_annotated_path = potential_groundingdino_path
                
                # Step 2: Strategist - Decompose question and create reasoning plan
                mvkb_payload = strategist.build_mvkb(question, image_path, answer_candidates, caption)
                
                if not mvkb_payload:
                    logging.error(f"Sample {i}: Strategist failed to build MVKB. Skipping.")
                    continue

                evidence_set = mvkb_payload.get("evidence_set", [])
                hypothesis_set = mvkb_payload.get("hypothesis_set", [])
                
                # Step 3: Synthesizer - Execute the reasoning plan
                synthesis_result = synthesizer.synthesize(
                    evidence_set=evidence_set, 
                    hypothesis_set=hypothesis_set,
                    answer_candidates=answer_candidates
                )
                
                final_answer = synthesis_result.get('answer')
                synthesis_status = synthesis_result.get('status', 'UNKNOWN')
                causal_trace = synthesis_result.get('causal_trace', [])

                # Normalize answer format for better evaluation accuracy
                if final_answer:
                    final_answer = _normalize_answer(final_answer)
                
                # Step 4: Enhanced Explanation with Causal Trace
                explanation_text = strategist.generate_explanation(
                    question=question,
                    synthesis_result=synthesis_result,
                    caption=caption,
                    evidence_set=evidence_set
                )
                
                # Store result
                result = {
                    'question_id': question_id,
                    'sample_id': total_processed,
                    'question': question,
                    'image_path': image_path,
                    'final_answer': final_answer,
                    'explanation': explanation_text,
                    'synthesis_status': synthesis_status,
                    'causal_trace': causal_trace,
                    'evidence_count': len(evidence_set),
                    'hypothesis_count': len(hypothesis_set),
                    'evidence_set': evidence_set,  # Add detailed evidence
                    'hypothesis_set': hypothesis_set,  # Add detailed hypothesis
                    'ground_truth': ground_truth,
                    'initial_candidates': answer_candidates,
                    'caption': caption,
                    'mvkb_entries_count': len(mvkb_payload),
                    'groundingdino_annotated_path': groundingdino_annotated_path
                }
                
                # Add VQA-X specific fields if available
                if dataset_format == 'vqax':
                    result.update({
                        'ground_truth_explanations': sample.get('explanation', [])
                    })
                
                # Check if answer is correct
                is_correct = is_answer_correct(result)

                # Store sample details for summary
                sample_detail = {
                    'question_id': question_id,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'final_answer': final_answer,
                    'is_correct': is_correct,
                    'evidence_count': len(evidence_set),
                    'hypothesis_count': len(hypothesis_set),
                    'evidence_set': evidence_set,
                    'hypothesis_set': hypothesis_set,
                    'synthesis_status': synthesis_status
                }
                sample_details.append(sample_detail)

                # Save based on correctness
                if is_correct and correct_count < max_correct:
                    save_sample_result(result, correct_dir, "correct", correct_count + 1)
                    correct_count += 1
                    logging.info(f"✅ Correct sample {correct_count}/{max_correct} saved")
                elif not is_correct and wrong_count < max_wrong:
                    save_sample_result(result, wrong_dir, "wrong", wrong_count + 1)
                    wrong_count += 1
                    logging.info(f"❌ Wrong sample {wrong_count}/{max_wrong} saved")

                logging.info(f"📊 Progress: {correct_count}/{max_correct} correct, {wrong_count}/{max_wrong} wrong")
                
            except Exception as e:
                logging.error(f"❌ Sample {i} failed: {e}")
                continue
        
        # Create summary file
        summary = {
            "total_processed": total_processed,
            "correct_samples": correct_count,
            "wrong_samples": wrong_count,
            "correct_samples_saved": list(range(1, correct_count + 1)),
            "wrong_samples_saved": list(range(1, wrong_count + 1)),
            "target_reached": (correct_count >= max_correct and wrong_count >= max_wrong),
            "sample_details": sample_details  # Include detailed information
        }
        
        summary_file = os.path.join(output_base_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📋 Summary saved to: {summary_file}")
        logging.info(f"🎉 Auto Save Pipeline completed!")
        logging.info(f"📊 Final results: {correct_count} correct, {wrong_count} wrong samples")
        logging.info(f"📁 Output directory: {output_base_dir}")
        
        return summary
        
    except Exception as e:
        logging.error(f"❌ Auto Save Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    run_auto_save_pipeline(
        use_vllm=True,
        max_correct=3,
        max_wrong=3,
        output_base_dir="auto_output"
    ) 
