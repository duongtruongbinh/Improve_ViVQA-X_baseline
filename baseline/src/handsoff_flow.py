from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_core.models import ChatCompletionClient
from agents.EncyclopedicAgent import singlehop_encyclopedic, twohop_encyclopedic
from agents.TwoImageVQAAgent import twoimage_vqa
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken, Image
from pathlib import Path
import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
import logging
import os
import json
import sys
import traceback
from typing import Dict, Any, Optional
from collections import defaultdict
import time

# Import local modules
try:
    from .config_loader import app_config
    from .vllm_clients import vlm_client_vllm
    from .image_utils import process_image_for_vlm_agent
    from .dataloader import VQAv2Dataset
    from .utils import setup_logging, get_question_type
    from .evaluation import perform_direct_accuracy_check
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# VLLM Configuration
VLLM_API_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VLLM_API_KEY = "EMPTY"
VLLM_TEMPERATURE = 0.0

# Dataset Configuration
DATASET_SPLIT_NAME = "rest-val"
VAL_IMAGE_DIR = "/mnt/VLAI_data/COCO_Images/val2014"
REST_VAL_QUESTIONS_PATH = "/mnt/VLAI_data/VQAv2/v2_OpenEnded_mscoco_rest_val2014_questions.json"
REST_VAL_ANNOTATIONS_PATH = "/mnt/VLAI_data/VQAv2/v2_mscoco_rest_val2014_annotations.json"

# Get configuration from app_config
VQA_CONFIG = app_config.get("datasets", {})
INFERENCE_CONFIG = app_config.get("inference_settings", {})

# VQA Dataset Configuration
APP_CONFIG_DATASET_NAME = VQA_CONFIG.get("dataset_name", "vqa-v2")
APP_CONFIG_DATASET_SPLIT = VQA_CONFIG.get("vqa_v2_dataset_split", DATASET_SPLIT_NAME)

# Inference Settings
APP_CONFIG_MAX_QUESTIONS_TO_LOAD = VQA_CONFIG.get("num_test_data") if VQA_CONFIG.get("use_num_test_data", False) else None
APP_CONFIG_EVAL_F1_THRESHOLD = INFERENCE_CONFIG.get("f1_threshold_for_other_type", 0.5)

# Use VLLM client from vllm_clients.py
client = ChatCompletionClient(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_URL,
    model=VLLM_MODEL_NAME,
    temperature=VLLM_TEMPERATURE
)

dispatcher_system_message = """
    You are a multimodal Dispatcher Agent. Your job is to analyze the user's visual question and route it to the correct specialist agent.

    Currently available:
    - SingleHopEncyclopedicAgent → for factual questions that require one step of reasoning.
    - TwoHopEncyclopedicAgent → for complex questions that require identifying something in the image and then retrieving related knowledge.
    - TwoImageVQAAgent → for questions that involve reasoning about two images.
    Task workflow:

    PHASE 1: PLANNING
    - Briefly describe your reasoning about the question (e.g. "This is a factual question about a church in the image").
    - Then, hand off the task to ONE appropriate agent.

    PHASE 2: AFTER AGENT RESPONDS
    - Extract the final answer and explanation from the agent's message.
    - Output a JSON in this exact format: "answer": "[Direct answer to the question]"

    - Immediately after the JSON, on a new line, output: TERMINATE

    Constraints:
    - Do NOT skip or forget the word TERMINATE.
    - Route only to one agent per task.
    - Keep your reasoning short and relevant.
"""


dispatcher = AssistantAgent(
    name="Dispatcher",
    system_message=dispatcher_system_message,
    model_client=client,
    handoffs=["SingleHopEncyclopedicAgent", "TwoHopEncyclopedicAgent","TwoImageVQAAgent"],
)

text_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_termination | max_messages_termination
vqa_team = Swarm(
    participants=[dispatcher,
                  singlehop_encyclopedic, twohop_encyclopedic, twoimage_vqa], termination_condition=termination
)

# Đọc ảnh từ file
image_path = Path("cat.jpg")  # Đổi đường dẫn tới ảnh phù hợp
image = Image.from_file(image_path)

# Tạo message đầu vào dạng multimodal
# message = MultiModalMessage(
#     content=[
#         "Question: What breed is this cat?, Image_url: https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
#     ],
#     source="user"
# )

message = "Question: What is the Köppen climate classification for the city where this mosque is located?, Image_url: https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/%D9%85%D8%B3%D8%AC%D8%AF_%D8%A7%D9%84%D9%85%D8%B1%D8%B3%D9%8A_%D8%A7%D8%A8%D9%88_%D8%A7%D9%84%D8%B9%D8%A8%D8%A7%D8%B3.jpg/1200px-%D9%85%D8%B3%D8%AC%D8%AF_%D8%A7%D9%84%D9%85%D8%B1%D8%B3%D9%8A_%D8%A7%D8%A8%D9%88_%D8%A7%D9%84%D8%B9%D8%A8%D8%A7%D8%B3.jpg?20160929000632"

async def main():
  await Console(vqa_team.run_stream(task=message))
  await client.close()

async def run_handsoff_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str,
    target_answer: Any,
    question_type: str,
    logger_instance: logging.Logger
) -> Dict[str, Any]:
    logger = logger_instance
    logger.info(f"Starting Handsoff VQA Pipeline for QID: {question_id}")
    
    if not os.path.exists(image_path):
        error_msg = f"Image not found at {image_path}"
        logger.error(error_msg)
        return {
            "question_id": question_id,
            "error": error_msg,
            "flow_type_used": "handsoff"
        }
    
    try:
        # Create message for the VQA team
        message = f"Question: {question}, Image_url: {image_path}"
        
        # Run the VQA team
        result = await Console(vqa_team.run_stream(task=message))
        
        # Extract answer from result
        if isinstance(result, str):
            # Try to parse JSON if present
            try:
                import json
                result_dict = json.loads(result)
                final_answer = result_dict.get("answer", result)
            except:
                final_answer = result
        else:
            final_answer = str(result)
        
        # Evaluate accuracy
        accuracy_result, grades_output, majority_vote_output = perform_direct_accuracy_check(
            final_answer_text=final_answer,
            target_answers=target_answer,
            question_type=question_type,
            verbose=True
        )
        
        # Log evaluation details
        logger.info(f"Evaluation for QID {question_id}:")
        logger.info(f"Question: {question}")
        logger.info(f"Target Answer: {target_answer}")
        logger.info(f"Predicted Answer: {final_answer}")
        logger.info(f"Accuracy Result: {accuracy_result}")
        logger.info(f"Grades Output: {grades_output}")
        logger.info(f"Majority Vote: {majority_vote_output}")
        
        return {
            "question_id": question_id,
            "question": question,
            "image_path": image_path,
            "target_answer": target_answer,
            "question_type": question_type,
            "predicted_answer": final_answer,
            "direct_accuracy_check": accuracy_result,
            "grades_output": grades_output,
            "majority_vote": majority_vote_output,
            "flow_type_used": "handsoff"
        }
        
    except Exception as e:
        logger.error(f"Error in handsoff VQA pipeline for QID {question_id}: {e}", exc_info=True)
        return {
            "question_id": question_id,
            "error": str(e),
            "flow_type_used": "handsoff"
        }

async def main_handsoff_vqa(num_items: Optional[int] = 10, save_interval: int = 1000):
    """
    Run the handsoff VQA pipeline on the dataset.
    
    Args:
        num_items (Optional[int]): Number of items to process. If None, process entire dataset.
        save_interval (int): Interval for saving intermediate results.
    """
    # Create a configuration object with required attributes
    class Config:
        def __init__(self):
            self.dataset_name = "vqa-v2"
            self.dataset_split = DATASET_SPLIT_NAME
            self.vqa_flow_type = "handsoff"
            self.use_num_test_data = num_items is not None
            self.num_test_data = num_items
            self.random_seed = 42
            self.verbose = True

    # Setup logging
    logger, output_dir = setup_logging(Config(), {
        "inference_settings": {"verbose": True},
        "datasets": {
            "dataset_name": "vqa-v2",
            "vqa_v2_dataset_split": DATASET_SPLIT_NAME,
            "vqa_v2_val_images_dir": VAL_IMAGE_DIR,
            "vqa_v2_rest_val_questions_file": REST_VAL_QUESTIONS_PATH,
            "vqa_v2_rest_val_annotations_file": REST_VAL_ANNOTATIONS_PATH,
            "use_num_test_data": num_items is not None,
            "num_test_data": num_items,
            "random_seed": 42
        }
    })
    
    # Load dataset
    try:
        dataset = VQAv2Dataset({
            "datasets": {
                "dataset_name": "vqa-v2",
                "vqa_v2_dataset_split": DATASET_SPLIT_NAME,
                "vqa_v2_val_images_dir": VAL_IMAGE_DIR,
                "vqa_v2_rest_val_questions_file": REST_VAL_QUESTIONS_PATH,
                "vqa_v2_rest_val_annotations_file": REST_VAL_ANNOTATIONS_PATH,
                "use_num_test_data": num_items is not None,
                "num_test_data": num_items,
                "random_seed": 0
            }
        })
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Initialize statistics tracking
    stats = {
        "correct_by_type": defaultdict(int),
        "total_by_type": defaultdict(int),
        "total_correct": 0,
        "total_questions": 0,
        "errors": [],
        "timeouts": 0,
        "processing_times": []
    }
    
    # Process the dataset
    results = []
    start_time = time.time()
    total_items = len(dataset)
    items_to_process = min(num_items, total_items) if num_items is not None else total_items
    
    logger.info(f"Starting evaluation with {items_to_process} items")
    
    for i, item in enumerate(dataset):
        if num_items is not None and i >= num_items:
            break
            
        item_start_time = time.time()
        result = await run_handsoff_vqa_pipeline(
            image_path=item["image_path"],
            question=item["question"],
            question_id=str(item["question_id"]),
            target_answer=item["answer"],
            question_type=get_question_type(item["question"]),
            logger_instance=logger
        )
        
        # Update statistics
        question_type = get_question_type(item["question"])
        stats["total_by_type"][question_type] += 1
        
        # Track processing time
        item_processing_time = time.time() - item_start_time
        stats["processing_times"].append(item_processing_time)
        
        # Check result status
        if result.get("status") == "error":
            stats["errors"].append({
                "question_id": item["question_id"],
                "error": result.get("error", "Unknown error")
            })
            if "timeout" in str(result.get("error", "")).lower():
                stats["timeouts"] += 1
            continue
            
        # Check if result is correct based on direct_accuracy_check
        accuracy_result = result.get("direct_accuracy_check", {})
        is_correct = accuracy_result.get("is_correct", False) if isinstance(accuracy_result, dict) else False
        
        if is_correct:
            stats["correct_by_type"][question_type] += 1
            stats["total_correct"] += 1
        stats["total_questions"] += 1
        
        results.append(result)
        
        # Save intermediate results at specified interval or at the end
        if (i + 1) % save_interval == 0 or i == items_to_process - 1:
            output_file = os.path.join(output_dir, "handsoff_vqa_results.json")
            with open(output_file, "w") as f:
                json.dump({
                    "results": results,
                    "statistics": {
                        "correct_by_type": dict(stats["correct_by_type"]),
                        "total_by_type": dict(stats["total_by_type"]),
                        "total_correct": stats["total_correct"],
                        "total_questions": stats["total_questions"],
                        "error_count": len(stats["errors"]),
                        "timeout_count": stats["timeouts"],
                        "average_processing_time": sum(stats["processing_times"]) / len(stats["processing_times"]) if stats["processing_times"] else 0
                    }
                }, f, indent=2)
            
            logger.info(f"Processed {i+1}/{items_to_process} items. Results saved to {output_file}")
            
            # Print progress statistics
            current_accuracy = (stats['total_correct'] / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
            logger.info(f"Current accuracy: {current_accuracy:.2f}%")
            logger.info(f"Average processing time: {sum(stats['processing_times'])/len(stats['processing_times']):.2f} seconds per question")
    
    # Print final statistics
    total_time = time.time() - start_time
    logger.info("\n--- Overall Evaluation Statistics (rest-val) ---")
    logger.info(f"Total questions evaluated: {stats['total_questions']}")
    logger.info(f"Correct predictions: {stats['total_correct']}")
    overall_accuracy = (stats['total_correct'] / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
    logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average processing time per question: {total_time/stats['total_questions']:.2f} seconds")
    logger.info(f"Error count: {len(stats['errors'])}")
    logger.info(f"Timeout count: {stats['timeouts']}")
    
    logger.info("\n--- Accuracy Statistics By Question Type (rest-val) ---")
    sorted_q_types = sorted(stats["total_by_type"].keys())
    for q_type in sorted_q_types:
        if stats["total_by_type"][q_type] > 0:
            acc_type = (stats["correct_by_type"][q_type] / stats["total_by_type"][q_type]) * 100
            logger.info(f"Type '{q_type}': {acc_type:.2f}% (Correct: {stats['correct_by_type'][q_type]}, Total: {stats['total_by_type'][q_type]})")
        else:
            logger.info(f"Type '{q_type}': 0.00% (Correct: 0, Total: 0)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run VQA evaluation with specified number of items')
    parser.add_argument('--num-items', type=int, default=10,
                      help='Number of dataset items to process (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1000,
                      help='Interval for saving intermediate results (default: 1000)')
    args = parser.parse_args()
    
    try:
        asyncio.run(main_handsoff_vqa(args.num_items, args.save_interval))
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
    finally:
        print("\nHandsoff VQA script execution finished.")