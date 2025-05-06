import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from CLIP_Count.run import Model as CLIP_Count
from groundingdino.util.inference import load_model
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from counting import query_clip_count
from detections import query_grounded_sam, query_grounding_dino, query_sam
from query_llm import QueryLLM
from query_vlm import QueryVLM
from utils import Colors, Grader, record_final_accuracy, write_response_to_json


def inference(device: torch.device, args: Dict, test_loader: torch.utils.data.DataLoader) -> None:
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    llm_model = args['llm']['llm_model']
    verbose = args['inference']['verbose']
    output_response_filename = args['inference']['output_response_filename']
    print_every = args['inference']['print_every']
    save_output_response = args['inference']['save_output_response']

    llm_agent = QueryLLM(args)
    vlm_agent = QueryVLM(args)
    clip_count_model = CLIP_Count.load_from_checkpoint('CLIP_Count/ckpt/clipcount_pretrained.ckpt', strict=False).to(device)
    clip_count_model.eval()

    grader = Grader()

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, question_id, target_answer = data['image_id'], data['image_path'], data['question'], data['question_id'], data['answer']
            assert len(image_path) == 1

            image = np.asarray(Image.open(image_path[0]).convert("RGB"))

            answer = vlm_agent.query_vlm(image, question[0], step='ask_directly', verbose=verbose)

            match_baseline_failed = re.search(r'\[Answer Failed\]', answer[0]) is not None or re.search(r'sorry', answer[0].lower()) is not None or len(answer[0]) == 0
            verify_numeric_answer = re.search(r'\[Non-zero Numeric Answer\]', answer[0]) is not None

            is_numeric_answer = re.search(r'\[Numeric Answer\](.*)', answer[0])
            if is_numeric_answer:
                numeric_answer = is_numeric_answer.group(1)
                number_is_large = llm_agent.query_llm([numeric_answer], llm_model=llm_model, step='check_numeric_answer', verbose=verbose)
                if re.search(r'Yes', number_is_large, re.IGNORECASE):
                    match_baseline_failed, verify_numeric_answer = True, True

            if match_baseline_failed:
                if verbose:
                    msg = "The baseline model needs further assistance to predict a numeric answer. Reattempting with multi-agents." if verify_numeric_answer else "The baseline model failed to answer the question initially with missing objects. Reattempting with multi-agents."
                    print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                needed_objects = llm_agent.query_llm(question, previous_response=answer[0], llm_model=llm_model, step='needed_objects',
                                                    verify_numeric_answer=verify_numeric_answer, verbose=verbose)

                if verify_numeric_answer:
                    reattempt_answer = query_clip_count(device, image, clip_count_model, prompts=needed_objects, verbose=verbose)
                else:
                    image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)
                    object_attributes = vlm_agent.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=verbose)
                    reattempt_answer = vlm_agent.query_vlm(image, question[0], step='reattempt', obj_descriptions=object_attributes[0], prev_answer=answer[0],
                                                            needed_objects=needed_objects, verbose=verbose)[0]

                grades = [llm_agent.query_llm(question, target_answer=target_answer[0], model_answer=reattempt_answer, step='grade_answer', grader_id=i, verbose=verbose) for i in range(3)]

                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer': verify_numeric_answer, 'initial_answer': answer[0], 'reattempt_answer': reattempt_answer,
                                 'needed_objects': needed_objects, 'grades': grades}
                if not verify_numeric_answer:
                    response_dict['object_attributes'] = object_attributes[0]
                    response_dict['boxes'] = str(boxes)
                    response_dict['logits'] = str(logits)
                    response_dict['phrases'] = phrases

            else:
                grades = [llm_agent.query_llm(question, target_answer=target_answer[0], model_answer=answer[0], step='grade_answer', grader_id=i, verbose=verbose) for i in range(3)]
                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer': verify_numeric_answer, 'initial_answer': answer[0], 'grades': grades}

            majority_vote = grader.accumulate_grades(args, grades, match_baseline_failed)
            response_dict['majority_vote'] = majority_vote

            if (batch_count + 1) % print_every == 0:
                baseline_accuracy, final_accuracy, _ = grader.average_score()
                print(f'Accuracy at batch idx {batch_count} (baseline, final) {baseline_accuracy} {final_accuracy}')

            if save_output_response:
                write_response_to_json(question_id, response_dict, output_response_filename)

        baseline_accuracy, final_accuracy, stats = grader.average_score()
        if save_output_response:
            record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename)
        print(f'Accuracy (baseline, final) {baseline_accuracy} {final_accuracy} stats {stats}')