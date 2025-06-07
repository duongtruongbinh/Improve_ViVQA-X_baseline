from tqdm import tqdm
import os
import sys
import torch
from PIL import Image
import numpy as np
import re

# Add CLIP_Count to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CLIP_Count'))
from run import Model as CLIP_Count

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from query_vlm import QueryVLM
from query_llm import QueryLLM
from detections import query_sam, query_grounded_sam, query_grounding_dino
from counting import query_clip_count
from utils import *

def inference(device, args, test_loader):
    # Building GroundingDINO, LLM, VLM, and CLIP_Count models as multi-agents
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    LLM, VLM = QueryLLM(args), QueryVLM(args)
    clip_count = CLIP_Count.load_from_checkpoint('CLIP_Count/ckpt/clipcount_pretrained.ckpt', strict=False).to(device)
    clip_count.eval()  # Set the model to evaluation mode

    # Use hardcoded VQA evaluation from utils
    grader = Grader()
    output_response_filename = args['inference']['output_response_filename']

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, question_id, target_answer = data['image_id'], data['image_path'], data['question'], data['question_id'], data['answer']
            assert len(image_path) == 1

            image = np.asarray(Image.open(image_path[0]).convert("RGB"))

            # First try to answer using baseline VLM
            answer = VLM.query_vlm(image, question[0], step='ask_directly', verbose=args['inference']['verbose'])

            # Improved trigger logic
            baseline_answer = answer[0].strip()
            
            # Check for explicit failure signals
            explicit_failure = any(signal in baseline_answer for signal in ['[Answer Failed]', 'sorry', "I don't know", "I cannot", "unclear"])
            
            # Check for numeric answer triggers
            zero_numeric = '[Zero Numeric Answer]' in baseline_answer
            nonzero_numeric = '[Non-zero Numeric Answer]' in baseline_answer
            verify_numeric_answer = nonzero_numeric
            
            # Check if answer is too vague or empty
            vague_answer = len(baseline_answer.strip()) == 0 or baseline_answer.lower().strip() in ['unknown', 'unsure', 'maybe']
            
            # Original author's large number detection for counting questions
            is_counting_question = any(phrase in question[0].lower() for phrase in ['how many', 'what number', 'count'])
            if is_counting_question and not zero_numeric and not nonzero_numeric and not explicit_failure:
                # Check if it's a large number that needs verification
                try:
                    # Try to extract number from answer
                    import re
                    numbers = re.findall(r'\d+', baseline_answer)
                    if numbers:
                        largest_num = max(int(num) for num in numbers)
                        if largest_num > 10:  # Consider large if > 10
                            verify_numeric_answer = True
                            explicit_failure = True
                except:
                    pass
            
            # Decide if multi-agent approach is needed
            match_baseline_failed = explicit_failure or zero_numeric or vague_answer

            # Use multi-agent approach when triggered
            if match_baseline_failed:
                if args['inference']['verbose']:
                    if verify_numeric_answer:
                        msg = "The baseline model needs assistance with counting/numeric answer. Triggering multi-agent approach."
                    else:
                        msg = f"The baseline model failed (answer: '{baseline_answer}'). Triggering multi-agent approach."
                    print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                # Extract needed objects for the task
                needed_objects = LLM.query_llm(question, previous_response=baseline_answer, llm_model=args['llm']['llm_model'], 
                                               step='needed_objects', verify_numeric_answer=verify_numeric_answer, 
                                               verbose=args['inference']['verbose'])

                if verify_numeric_answer:
                    # Use CLIP-Count for precise counting
                    reattempt_answer = query_clip_count(device, image, clip_count, prompts=needed_objects, verbose=args['inference']['verbose'])
                else:
                    # Use Grounding DINO + VLM for detailed analysis
                    image_annotated, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)

                    if len(boxes) > 0:
                        # Analyze object attributes with VLM
                        object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=args['inference']['verbose'])

                        # Generate final answer with object context
                        reattempt_answer = VLM.query_vlm(image, question[0], step='reattempt', obj_descriptions=object_attributes[0], 
                                                         prev_answer=baseline_answer, needed_objects=needed_objects, verbose=args['inference']['verbose'])[0]
                    else:
                        # No objects detected, try direct reattempt with question
                        if args['inference']['verbose']:
                            print(f'{Colors.WARNING}No objects detected, attempting direct reattempt{Colors.ENDC}')
                        reattempt_answer = VLM.query_vlm(image, question[0], step='reattempt', obj_descriptions=[f"No specific objects detected for: {needed_objects}"], 
                                                         prev_answer=baseline_answer, verbose=args['inference']['verbose'])[0]
                
                # Post-process reattempt answer
                reattempt_answer = reattempt_answer.strip()
                if '[Answer Failed]' in reattempt_answer or len(reattempt_answer) == 0:
                    # Reattempt also failed, fall back to baseline
                    final_answer = baseline_answer
                    if args['inference']['verbose']:
                        print(f'{Colors.FAIL}Reattempt also failed, using baseline answer{Colors.ENDC}')
                else:
                    final_answer = reattempt_answer
            else:
                final_answer = baseline_answer

            # Use VQA evaluation from utils
            gt_answers = [{'answer': target_answer[0]}] * 3  # VQA format with 3 annotators
            vqa_score = grader.vqa_score(final_answer, gt_answers)
            baseline_score = grader.vqa_score(baseline_answer, gt_answers)
            
            if args['inference']['verbose']:
                print(f"VQA Score: {vqa_score:.3f} [{'Correct' if vqa_score > 0 else 'Incorrect'}]")

            # Use grader's accumulate_grades for proper tracking
            majority_vote = grader.accumulate_grades(args, [], match_baseline_failed, 
                                                   target_answer=gt_answers, 
                                                   model_answer=final_answer, 
                                                   answer_type='unknown',
                                                   initial_answer=baseline_answer)

            # Record response
            response_dict = {
                'image_id': str(image_id[0].item()), 
                'image_path': image_path[0], 
                'question_id': str(question_id[0].item()), 
                'question': question[0], 
                'target_answer': target_answer[0],
                'initial_answer': baseline_answer,
                'final_answer': final_answer,
                'match_baseline_failed': match_baseline_failed, 
                'verify_numeric_answer': verify_numeric_answer,
                'vqa_score': vqa_score,
                'baseline_score': baseline_score,
                'majority_vote': majority_vote
            }

            if match_baseline_failed:
                response_dict.update({
                    'needed_objects': needed_objects,
                    'object_attributes': object_attributes[0] if 'object_attributes' in locals() else '',
                    'boxes': str(boxes) if 'boxes' in locals() else '',
                    'logits': str(logits) if 'logits' in locals() else '',
                    'phrases': phrases if 'phrases' in locals() else []
                })

            # Progress reporting
            if (batch_count + 1) % args['inference']['print_every'] == 0:
                baseline_accuracy, final_accuracy, _ = grader.average_score()
                print(f'\n📊 Progress - Batch {batch_count + 1}:')
                print(f'   Baseline: {baseline_accuracy:.1%} | Multi-Agent: {final_accuracy:.1%}')

            if args['inference']['save_output_response']:
                write_response_to_json(question_id, response_dict, output_response_filename)

        # Final results
        baseline_accuracy, final_accuracy, stats = grader.average_score()
        if args['inference']['save_output_response']:
            record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename)
        
        print(f'\n{"="*50}')
        print(f'🎯 FINAL RESULTS')
        print(f'{"="*50}')
        print(f'Baseline Accuracy:    {baseline_accuracy:.1%}')
        print(f'Multi-Agent Accuracy: {final_accuracy:.1%}')
        improvement = (final_accuracy - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
        print(f'Improvement:          {improvement:+.1f}%')
        print(f'Total Questions:      {stats["count_total"]}')
        print(f'Correct (Final):      {stats["count_correct"]}/{stats["count_total"]}')
        print(f'{"="*50}')
        
        return baseline_accuracy, final_accuracy, stats
