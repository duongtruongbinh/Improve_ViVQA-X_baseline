#!/usr/bin/env python3
"""
Simple Multi-Agent ViVQA-X evaluation script
This script simulates multi-agent approach using iterative VLM + LLM refinement
"""

import os
import sys
import json
import yaml
import argparse
import base64
import requests
import io
import re
from datetime import datetime
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
module_x_dir = os.path.dirname(current_dir)
main_project_dir = os.path.dirname(module_x_dir)

sys.path.append(module_x_dir)
sys.path.append(main_project_dir)

# Import module_x components directly
sys.path.append(os.path.join(module_x_dir, 'dataloader'))
sys.path.append(os.path.join(module_x_dir, 'evaluation'))

from vivqa_x_dataloader import get_dataloader
from vivqa_x_evaluator import ViVQAXEvaluator

def load_config(config_path: str) -> dict:
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"✓ Successfully loaded config with keys: {list(config.keys()) if config else 'None'}")
            return config
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_output_directory(args: dict) -> str:
    """Setup output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args['vlm']['vllm_8000']['model'].split('/')[-1]
    dataset_split = args['datasets']['vivqa_x_dataset_split']
    
    output_dir = os.path.join(module_x_dir, "outputs", f"vivqa_x_multiagent_{dataset_split}_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def query_vllm_api(messages: list, args: dict, session: requests.Session) -> str:
    """
    Query vLLM server with messages using a persistent session.
    
    Args:
        messages (list): The list of messages for the payload.
        args (dict): The configuration arguments.
        session (requests.Session): The session object for making HTTP requests.
    """
    try:
        # Get vLLM config
        provider = args['vlm'].get('provider', 'vllm_8000')
        vlm_config = args['vlm'][provider]
        base_url = vlm_config['base_url']
        model_name = vlm_config['model']
        max_tokens = vlm_config['max_tokens']
        temperature = vlm_config['temperature']
        
        # vLLM API format
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {vlm_config['api_key']}"
        }
        
        # Send request using the provided session
        response = session.post(base_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0].get('message', {}).get('content', '')
            else:
                return "Error: No valid response from server"
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
            return f"Error: HTTP {response.status_code}"
            
    except Exception as e:
        print(f"Error in vLLM query: {str(e)}")
        return f"Error: {str(e)}"

def vlm_agent_analysis(question: str, image_path: str, args: dict, session: requests.Session) -> dict:
    """
    Agent 1: VLM performs initial visual analysis.
    This function now also returns the base64 encoded image to avoid re-processing.
    """
    try:
        # Load and encode image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # VLM system prompt for initial analysis
        vlm_system_prompt = (
            "Bạn là một chuyên gia phân tích hình ảnh. Nhiệm vụ của bạn là mô tả chi tiết những gì bạn thấy trong hình ảnh.\n"
            "Hãy mô tả:\n"
            "1. Các đối tượng chính trong hình\n"
            "2. Màu sắc, hình dạng, vị trí của chúng\n"
            "3. Bối cảnh và môi trường\n"
            "4. Bất kỳ chi tiết nào có thể liên quan đến câu hỏi\n"
            "Trả lời bằng tiếng Việt một cách chi tiết và chính xác."
        )
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": vlm_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": f"Hãy mô tả chi tiết hình ảnh này. Câu hỏi cần trả lời là: {question}"}
                ]
            }
        ]
        
        visual_analysis = query_vllm_api(messages, args, session)
        
        return {
            "agent": "VLM",
            "task": "visual_analysis",
            "input": question,
            "output": visual_analysis,
            "base64_image": base64_image,  # Return encoded image for reuse
            "success": True
        }
        
    except Exception as e:
        return {
            "agent": "VLM",
            "task": "visual_analysis", 
            "input": question,
            "output": f"Error: {str(e)}",
            "base64_image": None,
            "success": False
        }

def llm_agent_reasoning(question: str, visual_analysis: str, args: dict, session: requests.Session) -> dict:
    """
    Agent 2: LLM performs reasoning based on visual analysis
    """
    try:
        # LLM system prompt for reasoning
        llm_system_prompt = (
            "Bạn là một chuyên gia lý luận và phân tích. Dựa vào mô tả hình ảnh được cung cấp, hãy suy luận để trả lời câu hỏi một cách chính xác.\n"
            "Trả lời theo định dạng:\n"
            "Phân tích: <Quá trình suy luận của bạn>\n"
            "Kết luận: <Câu trả lời cuối cùng>"
        )
        
        user_prompt = (
            f"Mô tả hình ảnh: {visual_analysis}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Dựa vào mô tả trên, hãy phân tích và suy luận để trả lời câu hỏi."
        )
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": llm_system_prompt}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        reasoning_result = query_vllm_api(messages, args, session)
        
        return {
            "agent": "LLM",
            "task": "reasoning",
            "input": {"question": question, "visual_analysis": visual_analysis},
            "output": reasoning_result,
            "success": True
        }
        
    except Exception as e:
        return {
            "agent": "LLM",
            "task": "reasoning",
            "input": {"question": question, "visual_analysis": visual_analysis},
            "output": f"Error: {str(e)}",
            "success": False
        }

def coordinator_agent_synthesis(question: str, base64_image: str, visual_analysis: str, reasoning_result: str, args: dict, session: requests.Session) -> dict:
    """
    Agent 3: Coordinator synthesizes final answer with explanation.
    This function now accepts a base64 string instead of an image path to avoid reloading.
    """
    try:
        # Coordinator system prompt
        coordinator_system_prompt = (
            "Bạn là một điều phối viên chuyên nghiệp. Nhiệm vụ của bạn là tổng hợp thông tin từ "
            "chuyên gia phân tích hình ảnh và chuyên gia lý luận để đưa ra câu trả lời cuối cùng.\n"
            "Bạn phải tuân thủ TUYỆT ĐỐI các quy tắc định dạng đầu ra dưới đây.\n\n"
            "--- QUY TẮC BẮT BUỘC ---\n"
            "1. 'Answer:' Chỉ chứa MỘT TỪ hoặc MỘT CỤM TỪ ngắn gọn duy nhất.\n"
            "2. 'Explain:' Lời giải thích ngắn gọn cho câu trả lời, dựa trên các phân tích đã có và bằng chứng thị giác trong hình."
        )
        
        user_prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Phân tích hình ảnh:\n{visual_analysis}\n\n"
            f"Kết quả lý luận:\n{reasoning_result}\n\n"
            f"Nhiệm vụ: Dựa vào các phân tích trên, đưa ra câu trả lời cuối cùng. Tuân thủ nghiêm ngặt các QUY TẮC BẮT BUỘC về định dạng. "
            f"Phần giải thích phải nêu bật bằng chứng thị giác quan trọng nhất để chứng minh cho câu trả lời, "
            f"tránh lặp lại nguyên văn các phân tích đã có."
        )
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": coordinator_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    # Use the pre-encoded base64 image string
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        final_response = query_vllm_api(messages, args, session)
        
        return {
            "agent": "Coordinator",
            "task": "synthesis",
            "input": {
                "question": question,
                "visual_analysis": visual_analysis,
                "reasoning_result": reasoning_result
            },
            "output": final_response,
            "success": True
        }
        
    except Exception as e:
        return {
            "agent": "Coordinator",
            "task": "synthesis",
            "input": {
                "question": question,
                "visual_analysis": visual_analysis, 
                "reasoning_result": reasoning_result
            },
            "output": f"Error: {str(e)}",
            "success": False
        }

def parse_coordinator_response(response_text: str) -> tuple:
    """
    Parse coordinator response to extract answer and explanation.
    This updated version sanitizes the output to remove punctuation and special characters
    for more robust evaluation.
    """
    try:
        response_text = response_text.strip()
        
        answer = ""
        explanation = ""
        
        # Method 1: Parse line by line (most reliable)
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer = line[len("Answer:"):].strip()
            elif line.lower().startswith("explain:"):
                explanation = line[len("Explain:"):].strip()
        
        # Method 2: Use regex pattern matching if line-by-line fails
        if not answer or not explanation:
            answer_match = re.search(r'Answer:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip().split('\n')[0]
            
            explain_match = re.search(r'Explain:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            if explain_match:
                explanation = explain_match.group(1).strip()
        
        # Fallback: extract from reasoning structure
        if not answer and ("Kết luận:" in response_text or "kết luận:" in response_text):
            if "Kết luận:" in response_text:
                answer = response_text.split("Kết luận:")[1].strip().split('\n')[0]
            elif "kết luận:" in response_text:
                answer = response_text.split("kết luận:")[1].strip().split('\n')[0]
        
        # Final fallback
        if not answer:
            lines = [l.strip() for l in response_text.split('\n') if l.strip()]
            if lines:
                for line in lines:
                    if not any(word in line.lower() for word in ['answer:', 'explain:', 'phân tích:', 'kết luận:']):
                        answer = line
                        break
                if not answer and lines:
                    answer = lines[0]

        if answer and not explanation:
             if len(response_text.split('\n')) > 1:
                  explanation = '\n'.join(response_text.split('\n')[1:]).strip()

        # 1. Chuẩn hóa 'answer': loại bỏ tất cả các dấu câu và ký tự đặc biệt.
        if answer:
            sanitized_answer = re.sub(r'[^\w\s]', ' ', answer)
            answer = ' '.join(sanitized_answer.split()).strip()

        # 2. Chuẩn hóa 'explanation': loại bỏ dấu câu nhưng giữ lại dấu phẩy.
        if explanation:
            sanitized_explanation = re.sub(r'[^\w\s,]', ' ', explanation)
            explanation = ' '.join(sanitized_explanation.split()).strip()
        else:
            explanation = "Không có lời giải thích chi tiết"
        
        return answer, explanation
        
    except Exception as e:
        print(f"Error parsing coordinator response: {str(e)}")
        return response_text.strip(), "Lỗi phân tích cú pháp"

def evaluate_vietnamese_answer(predicted: str, target: str) -> bool:
    """Evaluate Vietnamese answer accuracy"""
    pred_norm = predicted.strip().lower()
    target_norm = target.strip().lower()
    return pred_norm == target_norm

def multi_agent_process_item(data_item: dict, args: dict, session: requests.Session, verbose: bool = False) -> dict:
    """
    Process a single VQA-X item using multi-agent approach.
    Now uses a requests.Session for all API calls and passes encoded image data.
    """
    
    # Extract information from data item
    image_path = data_item['image_path']
    question = data_item['question']
    target_answer = data_item['answer']
    target_explanation = data_item['explanation']
    question_id = data_item['question_id']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-Agent Processing: Question {question_id}")
        print(f"Question: {question}")
        print(f"Target: {target_answer}")
        print(f"{'='*60}")
    
    agent_results = []
    
    try:
        # Step 1: VLM Agent - Visual Analysis
        if verbose:
            print("\n[Agent 1: VLM] Performing visual analysis...")
        
        vlm_result = vlm_agent_analysis(question, image_path, args, session)
        agent_results.append(vlm_result)
        
        if not vlm_result['success'] or not vlm_result['base64_image']:
            raise Exception(f"VLM Agent failed: {vlm_result['output']}")
        
        visual_analysis = vlm_result['output']
        base64_image = vlm_result['base64_image'] # Get the encoded image
        
        if verbose:
            print(f"Visual Analysis: {visual_analysis[:200]}")
        
        # Step 2: LLM Agent - Reasoning
        if verbose:
            print("\n[Agent 2: LLM] Performing reasoning...")
        
        llm_result = llm_agent_reasoning(question, visual_analysis, args, session)
        agent_results.append(llm_result)
        
        if not llm_result['success']:
            raise Exception(f"LLM Agent failed: {llm_result['output']}")
        
        if verbose:
            print(f"Reasoning: {llm_result['output'][:200]}")
        
        # Step 3: Coordinator Agent - Synthesis
        if verbose:
            print("\n[Agent 3: Coordinator] Synthesizing final answer...")
        
        # Pass the pre-encoded base64 image instead of the path
        coordinator_result = coordinator_agent_synthesis(
            question, base64_image, visual_analysis, llm_result['output'], args, session
        )
        agent_results.append(coordinator_result)
        
        if not coordinator_result['success']:
            raise Exception(f"Coordinator Agent failed: {coordinator_result['output']}")
        
        # Parse final response
        answer, explanation = parse_coordinator_response(coordinator_result['output'])
        
        # Evaluate answer accuracy
        answer_correct = evaluate_vietnamese_answer(answer, target_answer)
        
        if verbose:
            print(f"\nFinal Answer: {answer}")
            print(f"Explanation: {explanation}")
            print(f"Correct: {answer_correct}")
        
        result = {
            'question_id': question_id,
            'image_id': data_item.get('image_id', 0),
            'image_path': image_path,
            'question': question,
            'gt_ans': target_answer,
            'gt_explain': target_explanation,
            'pred_ans': answer,
            'pred_explain': explanation,
            'answer_correct': answer_correct,
            'multiagent_triggered': True,
            'agent_results': agent_results,
            'raw_response': coordinator_result['output']
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\nError in multi-agent processing: {str(e)}")
        
        return {
            'question_id': question_id,
            'image_id': data_item.get('image_id', 0),
            'image_path': image_path,
            'question': question,
            'gt_ans': target_answer,
            'gt_explain': target_explanation,
            'pred_ans': "[Error]",
            'pred_explain': "[Error]",
            'answer_correct': False,
            'multiagent_triggered': True,
            'agent_results': agent_results,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Simple Multi-Agent ViVQA-X Evaluation")
    parser.add_argument("--config", default="../configs/vivqa_x_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args_cmd = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args_cmd.config}")
    args = load_config(args_cmd.config)
    if args is None:
        print("Failed to load configuration")
        return
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("Loading ViVQA-X dataset...")
    try:
        dataloader, dataset_size = get_dataloader(args)
        print(f"Dataset loaded: {dataset_size} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize results storage
    results = []
    
    # Create a single session to be reused for all requests
    with requests.Session() as session:
        # Process each item with multi-agent approach
        print("\nStarting Multi-Agent evaluation...")
        print("="*60)
        for i, data_item in enumerate(tqdm(dataloader, desc="Multi-Agent Processing")):
            
            # Process item with multi-agent, passing the session
            result = multi_agent_process_item(data_item, args, session, verbose=args_cmd.verbose)
            results.append(result)
            
            # Print progress every 5 items
            if (i + 1) % 5 == 0:
                correct_so_far = sum(1 for r in results if r.get('answer_correct'))
                accuracy_so_far = correct_so_far / len(results)
                tqdm.write(f"Progress: {i+1}/{dataset_size}, Multi-Agent Accuracy: {accuracy_so_far:.3f}")
    
    # Save results
    results_file = os.path.join(output_dir, "vivqa_x_multiagent_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {results_file}")
    
    # Run evaluation
    print("\nRunning evaluation metrics...")
    try:
        evaluator = ViVQAXEvaluator()
        
        # Run evaluation using the results data directly
        evaluation_results = evaluator.evaluate_results_from_data(results, output_dir)
        
        print(f"\nDetailed results saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # Basic accuracy calculation as fallback
        correct_count = sum(1 for r in results if r.get('answer_correct'))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        print(f"\nBasic Multi-Agent Evaluation Results:")
        print(f"Total samples: {total_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Multi-Agent Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
