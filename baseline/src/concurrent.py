import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, AsyncGenerator, Optional, Tuple
import os
import json
import sys
import traceback
import logging
import argparse
from collections import defaultdict
import time
from datetime import datetime
import re

from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage, CreateResult, RequestUsage
from .vllm_clients import vlm_client_vllm
from .image_utils import process_image_for_vlm_agent
from .config_loader import app_config
from .dataloader import VQAv2Dataset
from .utils import setup_logging, get_question_type
from .evaluation import perform_direct_accuracy_check

# Topic types for concurrent processing
URGENT_VQA_TOPIC_TYPE = "urgent_vqa_tasks"
NORMAL_VQA_TOPIC_TYPE = "normal_vqa_tasks"
VQA_RESULTS_TOPIC_TYPE = "vqa_task_results"

# Topic sources
URGENT_VQA_SOURCE = "urgent_processor"
NORMAL_VQA_SOURCE = "normal_processor"
VQA_RESULTS_SOURCE = "results_collector"

vqa_results_topic_id = TopicId(type=VQA_RESULTS_TOPIC_TYPE, source=VQA_RESULTS_SOURCE)

# Enhanced System Prompt
SYSTEM_PROMPT = """You are a Visual Question Answering (VQA) expert. Your task is to analyze images and answer questions about them accurately and precisely.

Guidelines for different question types:

1. Number Questions:
   - Count carefully and precisely
   - For objects that are partially visible or unclear, provide a range (e.g., "3-4")
   - For large groups, use relative terms (e.g., "about 20", "more than 15")
   - If uncertain, use words like "approximately" or "around"
   - For very small numbers (1-5), be extra precise
   - Consider perspective and occlusion when counting

2. Color Questions:
   - Be precise with color descriptions
   - Use primary colors when possible
   - Consider lighting conditions
   - For complex colors, use compound terms (e.g., "light blue", "dark red")
   - Note any patterns or gradients
   - Consider color context (e.g., "sky blue" vs "navy blue")

3. Yes/No Questions:
   - Answer directly with "yes" or "no"
   - If uncertain, explain your reasoning
   - Consider all visual evidence
   - Be explicit about assumptions
   - Note any conditions that affect the answer

4. Other Questions:
   - Be specific and detailed
   - Consider spatial relationships
   - Note any relevant context
   - Explain your reasoning
   - Use precise language

General Guidelines:
- Always consider the full context of the image
- Be explicit about any uncertainties
- Use precise and clear language
- Consider multiple perspectives
- Note any relevant details that support your answer
- If the answer is unclear, explain your reasoning
- For ambiguous cases, provide the most likely answer with confidence level

Remember: Accuracy and precision are crucial. If you're uncertain, explain your reasoning rather than making assumptions."""

class ColorAnswerValidator:
    def __init__(self):
        self.color_mapping = {
            'red': ['red', 'crimson', 'scarlet', 'maroon', 'burgundy', 'ruby'],
            'blue': ['blue', 'navy', 'azure', 'cobalt', 'sapphire', 'teal', 'turquoise'],
            'green': ['green', 'emerald', 'olive', 'lime', 'mint', 'forest'],
            'yellow': ['yellow', 'gold', 'amber', 'mustard', 'lemon'],
            'purple': ['purple', 'violet', 'lavender', 'lilac', 'plum', 'magenta'],
            'orange': ['orange', 'peach', 'coral', 'salmon'],
            'pink': ['pink', 'rose', 'fuchsia', 'hot pink'],
            'brown': ['brown', 'tan', 'beige', 'khaki', 'chocolate'],
            'black': ['black', 'ebony', 'onyx'],
            'white': ['white', 'ivory', 'cream', 'pearl'],
            'gray': ['gray', 'grey', 'silver', 'charcoal']
        }
        self.reverse_mapping = {}
        for main_color, variants in self.color_mapping.items():
            for variant in variants:
                self.reverse_mapping[variant] = main_color

    def normalize_color(self, color: str) -> str:
        color = color.lower().strip()
        # Check if it's a compound color
        if ' ' in color:
            parts = color.split()
            # Handle cases like "light blue", "dark red"
            if parts[0] in ['light', 'dark', 'bright', 'deep', 'pale']:
                base_color = ' '.join(parts[1:])
                if base_color in self.reverse_mapping:
                    return self.reverse_mapping[base_color]
        # Direct mapping
        if color in self.reverse_mapping:
            return self.reverse_mapping[color]
        return color

    def validate(self, answer: str, target: str) -> bool:
        normalized_answer = self.normalize_color(answer)
        normalized_target = self.normalize_color(target)
        return normalized_answer == normalized_target

class NumberAnswerValidator:
    def __init__(self):
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
            'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000
        }

    def extract_number(self, text: str) -> Optional[float]:
        text = text.lower().strip()
        
        # Handle decimal numbers
        if '.' in text:
            try:
                return float(text)
            except ValueError:
                pass

        # Handle ranges (e.g., "3-4", "between 3 and 4")
        if '-' in text or ' to ' in text or ' between ' in text:
            parts = re.split(r'[- to between and]', text)
            numbers = []
            for part in parts:
                num = self.extract_number(part.strip())
                if num is not None:
                    numbers.append(num)
            if len(numbers) == 2:
                return sum(numbers) / 2

        # Handle number words
        words = text.split()
        if len(words) == 1 and words[0] in self.number_words:
            return float(self.number_words[words[0]])

        # Handle compound number words (e.g., "twenty one")
        if len(words) == 2 and words[0] in self.number_words and words[1] in self.number_words:
            return float(self.number_words[words[0]] + self.number_words[words[1]])

        # Try direct number conversion
        try:
            return float(text)
        except ValueError:
            pass

        return None

    def validate(self, answer: str, target: str) -> bool:
        answer_num = self.extract_number(answer)
        target_num = self.extract_number(target)
        
        if answer_num is None or target_num is None:
            return False
            
        # Allow for small differences in decimal numbers
        if isinstance(answer_num, float) and isinstance(target_num, float):
            return abs(answer_num - target_num) < 0.1
            
        return answer_num == target_num

class YesNoAnswerValidator:
    def __init__(self):
        self.positive_words = {'yes', 'yeah', 'yep', 'sure', 'correct', 'right', 'true', 'indeed', 'absolutely', 'definitely', 'certainly', 'of course', 'affirmative', 'positive'}
        self.negative_words = {'no', 'nope', 'nah', 'never', 'incorrect', 'wrong', 'false', 'negative', 'not', 'none', 'neither', 'nor', 'negative'}

    def normalize_answer(self, answer: str) -> Optional[bool]:
        answer = answer.lower().strip()
        
        # Check for explicit yes/no
        if answer in self.positive_words:
            return True
        if answer in self.negative_words:
            return False
            
        # Check for "yes" or "no" in the answer
        words = answer.split()
        for word in words:
            if word in self.positive_words:
                return True
            if word in self.negative_words:
                return False
                
        return None

    def validate(self, answer: str, target: str) -> bool:
        answer_bool = self.normalize_answer(answer)
        target_bool = self.normalize_answer(target)
        
        if answer_bool is None or target_bool is None:
            return False
            
        return answer_bool == target_bool

@dataclass
class VQAUserTask:
    task: str
    image_path: str
    question_id: Any = None
    ground_truth_answers_struct: List[Dict[str, str]] = field(default_factory=list)
    answer_type: str = "other"
    confidence_threshold: float = 0.7
    required_consensus: int = 2
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VQAWorkerTask:
    task: str
    image_path: str
    previous_results: List[str] = field(default_factory=list)

@dataclass
class WorkerTaskResult:
    result: str
    worker_key: str

@dataclass
class VQAFinalResult:
    result: str
    question_id: Any = None

@type_subscription(topic_type=URGENT_VQA_TOPIC_TYPE)
class UrgentVQAProcessorAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)
        self.model_client = vlm_client_vllm

    @message_handler
    async def on_urgent_vqa_task(self, message: VQAUserTask, ctx: MessageContext) -> None:
        try:
            worker_task = VQAWorkerTask(
                task=message.task,
                image_path=message.image_path,
                previous_results=[]
            )
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=worker_task.task)
            ]
            
            result = await self.model_client.create(
                messages=messages,
                image_path_for_create=worker_task.image_path
            )
            
            await ctx.publish(
                WorkerTaskResult(
                    result=result.content,
                    worker_key="urgent"
                ),
                vqa_results_topic_id
            )
        except Exception as e:
            print(f"Error in urgent processor: {str(e)}")
            traceback.print_exc()

@type_subscription(topic_type=NORMAL_VQA_TOPIC_TYPE)
class NormalVQAProcessorAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)
        self.model_client = vlm_client_vllm

    @message_handler
    async def on_normal_vqa_task(self, message: VQAUserTask, ctx: MessageContext) -> None:
        try:
            worker_task = VQAWorkerTask(
                task=message.task,
                image_path=message.image_path,
                previous_results=[]
            )
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=worker_task.task)
            ]
            
            result = await self.model_client.create(
                messages=messages,
                image_path_for_create=worker_task.image_path
            )
            
            await ctx.publish(
                WorkerTaskResult(
                    result=result.content,
                    worker_key="normal"
                ),
                vqa_results_topic_id
            )
        except Exception as e:
            print(f"Error in normal processor: {str(e)}")
            traceback.print_exc()

class VQADelegatorAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)
        self.model_client = vlm_client_vllm
        self.logger = logging.getLogger(self.__class__.__name__)

    @message_handler
    async def on_user_vqa_task(self, message: VQAUserTask, ctx: MessageContext) -> VQAFinalResult:
        self.logger.info(f"Received task for QID '{message.question_id}': '{message.task}'")
        # Create specialized subtasks based on question type
        object_task = VQAUserTask(
            task=message.task,
            image_path=message.image_path,
            question_id=f"{message.question_id}_obj",
            answer_type=message.answer_type
        )
        context_task = VQAUserTask(
            task=message.task,
            image_path=message.image_path,
            question_id=f"{message.question_id}_ctx",
            answer_type=message.answer_type
        )
        # Publish tasks to different topics in parallel
        await self.publish_message(object_task, topic_id=TopicId(type=URGENT_VQA_TOPIC_TYPE, source=URGENT_VQA_SOURCE))
        await self.publish_message(context_task, topic_id=TopicId(type=NORMAL_VQA_TOPIC_TYPE, source=NORMAL_VQA_SOURCE))
        # Wait for results
        results_queue = asyncio.Queue()
        async def collect_result(_agent: ClosureContext, message: WorkerTaskResult, ctx: MessageContext) -> None:
            await results_queue.put(message)
        await ClosureAgent.register_closure(
            self.runtime,
            "result_collector",
            collect_result,
            subscriptions=lambda: [TypeSubscription(topic_type=VQA_RESULTS_TOPIC_TYPE, agent_type="result_collector")]
        )
        # Wait for both results
        results = []
        for _ in range(2):
            result = await results_queue.get()
            results.append(result)
        # Normalize and deduplicate answers
        answers = [r.result.strip().lower() for r in results if r.result and not r.result.lower().startswith("error")] 
        self.logger.info(f"Collected answers: {answers}")
        
        # Enhanced answer synthesis based on question type
        if len(answers) == 2 and answers[0] == answers[1]:
            final_answer = answers[0]
        elif len(answers) == 1:
            final_answer = answers[0]
        elif len(answers) == 2:
            # Enhanced synthesis prompt with better guidance for different question types
            synthesis_prompt = (
                "You are a VQA answer combiner. Given two answers to the same question, pick the best or combine them into a short, direct answer.\n"
                "For yes/no questions: Output only 'yes' or 'no'.\n"
                "For number questions: Output a single number, no text. If answers differ, choose the more precise one.\n"
                "For other questions: Combine key details from both answers into a concise response.\n"
                f"Question: {message.task}\nAnswer 1: {answers[0]}\nAnswer 2: {answers[1]}\nFormat: Answer: <your answer>"
            )
            response = await self.model_client.create(
                messages=[UserMessage(content=synthesis_prompt, source="vqa_delegator")],
                image_path_for_create=message.image_path
            )
            final_answer = response.content if isinstance(response.content, str) else str(response.content)
            if "Answer:" in final_answer:
                final_answer = final_answer.split("Answer:")[-1].strip()
        else:
            final_answer = ""
        self.logger.info(f"Final synthesized answer for QID '{message.question_id}': '{final_answer}'")
        return VQAFinalResult(result=final_answer, question_id=message.question_id)

async def run_concurrent_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str,
    target_answer: Any,
    question_type: str,
    logger_instance: logging.Logger
) -> Dict[str, Any]:
    logger = logger_instance
    logger.info(f"Starting Concurrent VQA Pipeline for QID: {question_id}")
    
    if not os.path.exists(image_path):
        error_msg = f"Image not found at {image_path}"
        logger.error(error_msg)
        return {
            "question_id": question_id,
            "error": error_msg,
            "flow_type_used": "concurrent"
        }
    
    try:
        runtime = SingleThreadedAgentRuntime()
        
        # Register agents
        await UrgentVQAProcessorAgent.register(
            runtime,
            "urgent_processor",
            lambda: UrgentVQAProcessorAgent("Urgent VQA Processor")
        )
        
        await NormalVQAProcessorAgent.register(
            runtime,
            "normal_processor",
            lambda: NormalVQAProcessorAgent("Normal VQA Processor")
        )
        
        await VQADelegatorAgent.register(
            runtime,
            "vqa_delegator",
            lambda: VQADelegatorAgent("VQA Delegator")
        )
        
        runtime.start()
        
        # Create and send task
        user_task = VQAUserTask(
            task=question,
            image_path=image_path,
            question_id=question_id,
            ground_truth_answers_struct=[{"answer": target_answer}] if target_answer else [],
            answer_type=question_type
        )
        
        delegator_id = AgentId("vqa_delegator", "main_delegator")
        final_result = await runtime.send_message(user_task, delegator_id)
        
        await runtime.stop_when_idle()
        
        if not isinstance(final_result, VQAFinalResult):
            raise ValueError(f"Unexpected result type: {type(final_result)}")
        
        # Evaluate accuracy with detailed logging
        accuracy_result, grades_output, majority_vote_output = perform_direct_accuracy_check(
            final_answer_text=final_result.result,
            target_answers=target_answer,
            question_type=question_type,
            verbose=True  # Enable verbose output for detailed evaluation
        )
        
        # Log evaluation details
        logger.info(f"Evaluation for QID {question_id}:")
        logger.info(f"Question: {question}")
        logger.info(f"Target Answer: {target_answer}")
        logger.info(f"Predicted Answer: {final_result.result}")
        logger.info(f"Accuracy Result: {accuracy_result}")
        logger.info(f"Grades Output: {grades_output}")
        logger.info(f"Majority Vote: {majority_vote_output}")
        
        return {
            "question_id": question_id,
            "question": question,
            "image_path": image_path,
            "target_answer": target_answer,
            "question_type": question_type,
            "predicted_answer": final_result.result,
            "direct_accuracy_check": accuracy_result,
            "grades_output": grades_output,
            "majority_vote": majority_vote_output,
            "flow_type_used": "concurrent"
        }
        
    except Exception as e:
        logger.error(f"Error in concurrent VQA pipeline for QID {question_id}: {e}", exc_info=True)
        return {
            "question_id": question_id,
            "error": str(e),
            "flow_type_used": "concurrent"
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Run VQA evaluation with specified number of items')
    parser.add_argument('--num-items', type=int, default=10,
                      help='Number of dataset items to process (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1000,
                      help='Interval for saving intermediate results (default: 1000)')
    return parser.parse_args()

async def main_concurrent_vqa(num_items: Optional[int] = 10, save_interval: int = 1000):
    """
    Run the concurrent VQA pipeline on the dataset.
    
    Args:
        num_items (Optional[int]): Number of items to process. If None, process entire dataset.
        save_interval (int): Interval for saving intermediate results.
    """
    # Create a configuration object with required attributes
    class Config:
        def __init__(self):
            self.dataset_name = "vqa-v2"
            self.dataset_split = "rest-val"
            self.vqa_flow_type = "concurrent"
            self.use_num_test_data = num_items is not None
            self.num_test_data = num_items
            self.random_seed = 42
            self.verbose = True

    # Setup logging
    logger, output_dir = setup_logging(Config(), {
        "inference_settings": {"verbose": True},
        "datasets": {
            "dataset_name": "vqa-v2",
            "vqa_v2_dataset_split": "rest-val",
            "vqa_v2_val_images_dir": app_config.get("datasets", {}).get("vqa_v2_val_images_dir"),
            "vqa_v2_rest_val_questions_file": app_config.get("datasets", {}).get("vqa_v2_rest_val_questions_file"),
            "vqa_v2_rest_val_annotations_file": app_config.get("datasets", {}).get("vqa_v2_rest_val_annotations_file"),
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
                "vqa_v2_dataset_split": "rest-val",
                "vqa_v2_val_images_dir": app_config.get("datasets", {}).get("vqa_v2_val_images_dir"),
                "vqa_v2_rest_val_questions_file": app_config.get("datasets", {}).get("vqa_v2_rest_val_questions_file"),
                "vqa_v2_rest_val_annotations_file": app_config.get("datasets", {}).get("vqa_v2_rest_val_annotations_file"),
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
        result = await run_concurrent_vqa_pipeline(
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
            output_file = os.path.join(output_dir, "concurrent_vqa_results.json")
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
    args = parse_args()
    try:
        asyncio.run(main_concurrent_vqa(args.num_items, args.save_interval))
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
    finally:
        print("\nConcurrent VQA script execution finished.") 