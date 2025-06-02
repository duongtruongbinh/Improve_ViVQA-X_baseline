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

            # Original author's trigger logic
            match_baseline_failed = re.search(r'\[Answer Failed\]', answer[0]) is not None or re.search(r'sorry', answer[0].lower()) is not None or len(answer[0]) == 0
            verify_numeric_answer = re.search(r'\[Non-zero Numeric Answer\]', answer[0]) is not None

            # Original author's large number detection
            is_numeric_answer = re.search(r'\[Numeric Answer\](.*)', answer[0])
            if is_numeric_answer is not None:
                numeric_answer = is_numeric_answer.group(1)
                number_is_large = LLM.query_llm([numeric_answer], llm_model=args['llm']['llm_model'], 
                                              step='check_numeric_answer', verbose=args['inference']['verbose'])
                if re.search(r'Yes|yes', number_is_large) is not None:
                    match_baseline_failed, verify_numeric_answer = True, True

            # Use multi-agent approach when triggered
            if match_baseline_failed:
                if args['inference']['verbose']:
                    if verify_numeric_answer:
                        msg = "The baseline model needs further assistance to predict a numeric answer. Reattempting with multi-agents."
                    else:
                        msg = "The baseline model failed to answer the question initially with missing objects. Reattempting with multi-agents."
                    print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                # Extract needed objects for the task
                needed_objects = LLM.query_llm(question, previous_response=answer[0], llm_model=args['llm']['llm_model'], 
                                               step='needed_objects', verify_numeric_answer=verify_numeric_answer, 
                                               verbose=args['inference']['verbose'])

                if verify_numeric_answer:
                    # Use CLIP-Count for precise counting
                    reattempt_answer = query_clip_count(device, image, clip_count, prompts=needed_objects, verbose=args['inference']['verbose'])
                else:
                    # Use Grounding DINO + VLM for detailed analysis
                    image_annotated, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)

                    # Analyze object attributes with VLM
                    object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=args['inference']['verbose'])

                    # Generate final answer with object context
                    reattempt_answer = VLM.query_vlm(image, question[0], step='reattempt', obj_descriptions=object_attributes[0], 
                                                     prev_answer=answer[0], needed_objects=needed_objects, verbose=args['inference']['verbose'])[0]

                final_answer = reattempt_answer
            else:
                final_answer = answer[0]

            # Use VQA evaluation from utils
            gt_answers = [{'answer': target_answer[0]}] * 3  # VQA format with 3 annotators
            vqa_score = grader.vqa_score(final_answer, gt_answers)
            baseline_score = grader.vqa_score(answer[0], gt_answers)
            
            if args['inference']['verbose']:
                print(f"VQA Score: {vqa_score:.3f} [{'Correct' if vqa_score > 0 else 'Incorrect'}]")

            # Use grader's accumulate_grades for proper tracking
            majority_vote = grader.accumulate_grades(args, [], match_baseline_failed, 
                                                   target_answer=gt_answers, 
                                                   model_answer=final_answer, 
                                                   answer_type='unknown',
                                                   initial_answer=answer[0])

            # Record response
            response_dict = {
                'image_id': str(image_id[0].item()), 
                'image_path': image_path[0], 
                'question_id': str(question_id[0].item()), 
                'question': question[0], 
                'target_answer': target_answer[0],
                'initial_answer': answer[0],
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
