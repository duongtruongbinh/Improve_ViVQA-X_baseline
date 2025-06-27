## for 965, only need to get statement and probabily
import json
from tqdm import tqdm
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
import os
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Global variables for model name and local models
model_name = None
local_llm_model = None
local_llm_tokenizer = None
local_vlm_model = None
local_vlm_processor = None

def load_local_models():
    """Load local Qwen models with specific GPU allocation"""
    global local_llm_model, local_llm_tokenizer, local_vlm_model, local_vlm_processor
    
    # Load LLM model (Qwen2.5-7B-Instruct) on GPU 1 (mapped to cuda:0)
    llm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct"
    print(f"Loading LLM model from: {llm_model_path} on GPU 1 (cuda:0)")
    
    local_llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    local_llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype="auto",
        device_map="cuda:0"  # LLM on GPU 1 (mapped to cuda:0)
    )
    
    # Load VLM model (Qwen2.5-VL-7B-Instruct) on GPU 2 (mapped to cuda:1)
    vlm_model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
    print(f"Loading VLM model from: {vlm_model_path} on GPU 2 (cuda:1)")
    
    local_vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_path,
        torch_dtype="auto",
        device_map="cuda:1"  # VLM on GPU 2 (mapped to cuda:1)
    )
    
    # Default processor for VLM
    local_vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
    
    print("Local models loaded successfully!")

def local_llm_generate(prompt, max_tokens=512, temperature=0.7):
    """Generate text using local LLM model"""
    global local_llm_model, local_llm_tokenizer
    
    # Use only user prompt, no system message to keep consistency with original pipeline
    messages = [{"role": "user", "content": prompt}]
    text = local_llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = local_llm_tokenizer([text], return_tensors="pt").to(local_llm_model.device)
    
    with torch.no_grad():
        generated_ids = local_llm_model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation
            pad_token_id=local_llm_tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = local_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def local_vlm_generate(image_path, prompt, max_tokens=512):
    """Generate text using local VLM model with image input"""
    global local_vlm_model, local_vlm_processor
    
    # Keep the exact prompt structure from the original pipeline
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Preparation for inference
    text = local_vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = local_vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_vlm_model.device)
    
    # Generation of the output
    with torch.no_grad():
        generated_ids = local_vlm_model.generate(**inputs, max_new_tokens=max_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = local_vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()

def local_vlm_generate_with_beam_search(image_path, prompt, max_tokens=10, num_beams=4, num_return_sequences=4):
    """Generate text using local VLM model with beam search and real log-probabilities"""
    global local_vlm_model, local_vlm_processor
    
    # Keep the exact prompt structure from the original pipeline
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Preparation for inference
    text = local_vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = local_vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_vlm_model.device)
    
    # Generation with beam search and scores - mimicking BLIP2's behavior
    with torch.no_grad():
        outputs = local_vlm_model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,           # ✅ Get real scores
            return_dict_in_generate=True, # ✅ Return dict format
            do_sample=False,              # Deterministic beam search
            early_stopping=True
        )
    
    # Extract sequences and scores (mimicking BLIP2's logic)
    sequences_scores = outputs.sequences_scores  # Real log-probabilities
    sequences = outputs.sequences
    
    # Decode all generated sequences
    generated_ids_trimmed = [
        seq[len(inputs.input_ids[0]):] for seq in sequences
    ]
    
    output_texts = local_vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Build candidates dict with real scores (like BLIP2)
    candidates_dict = {}
    for i, (text, score) in enumerate(zip(output_texts, sequences_scores)):
        text = text.strip()
        if text:  # Only add non-empty answers
            candidates_dict[text] = score.cpu().item()  # Real log-probability
    
    # Return primary answer (first in beam search) and all candidates
    primary_answer = output_texts[0].strip() if output_texts else ""
    
    # Ensure we have at least the primary answer
    if not candidates_dict and primary_answer:
        candidates_dict[primary_answer] = 0.0  # Default log-prob
    
    return primary_answer, candidates_dict

def vqa_answer_initial(image_path, question):
    """Use local VLM model for initial VQA answering - returns candidate answers WITHOUT scores (Step 1 Responder)"""
    prompt = f"Question: {question} Short answer:"
    
    try:
        # Generate multiple candidate answers using beam search
        primary_answer, candidates_with_scores = local_vlm_generate_with_beam_search(
            image_path, prompt, max_tokens=10, num_beams=8, num_return_sequences=8
        )
        
        # Extract only the answer texts, ignore scores (according to paper Step 1)
        candidates_list = list(candidates_with_scores.keys())
        
        # Ensure we have at least 2 unique candidates
        unique_candidates = []
        for candidate in candidates_list:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        # If we have less than 2 unique candidates, add a generic alternative
        if len(unique_candidates) < 2:
            if primary_answer.lower() in ['yes', 'y']:
                unique_candidates.append('no')
            elif primary_answer.lower() in ['no', 'n']:
                unique_candidates.append('yes')
            else:
                unique_candidates.append('unknown')
        
        # Return only top 2 candidates as simple list (no scores per paper Step 1)
        return unique_candidates[:2]
        
    except Exception as e:
        print(f"Beam search failed: {e}, falling back to simple generation")
        
        # Fallback to simple generation
        primary_answer = local_vlm_generate(image_path, prompt, max_tokens=10).strip()
        
        # Create basic alternative based on answer type
        if primary_answer.lower() in ['yes', 'y']:
            return ['yes', 'no']
        elif primary_answer.lower() in ['no', 'n']:
            return ['no', 'yes']
        else:
            return [primary_answer, 'unknown']

def vqa_answer(image_path, question):
    """Use local VLM model for VQA answering with confidence scores (for assist queries in Step 3)"""
    prompt = f"Question: {question} Short answer:"
    
    try:
        # For assist queries, we DO need scores for evidence confidence
        primary_answer, candidates_dict = local_vlm_generate_with_beam_search(
            image_path, prompt, max_tokens=10, num_beams=8, num_return_sequences=8
        )
        
        if candidates_dict and primary_answer:
            return primary_answer, candidates_dict
            
    except Exception as e:
        print(f"Beam search failed: {e}, falling back to simple generation")
    
    # Fallback for assist queries
    primary_answer = local_vlm_generate(image_path, prompt, max_tokens=10).strip()
    
    # Create candidates dict with default scores for assist queries
    candidates_dict = {
        primary_answer: -1.0,
        "unknown": -5.0
    }
    
    return primary_answer, candidates_dict

def vlm_assist_query(image_path, question):
    # Use local VLM model for assist query generation
    # For assist query generation, we can use simple generation (not beam search)
    return [local_vlm_generate(image_path, question)]

def generate_image_caption(image_path):
    """Generate image caption using local VLM model"""
    caption_prompt = "Describe this image in detail."
    try:
        caption = local_vlm_generate(image_path, caption_prompt, max_tokens=100)
        return caption.strip()
    except Exception as e:
        print(f"Failed to generate caption: {e}")
        return "An image"  # Fallback caption




def get_assist_query_single(dataset, res, assist_query_method, prompt, model, vis_processors, txt_processors, idx):
    
    top_k = 2 # num of top k candidates
    # model, vis_processors, txt_processors parameters are now unused but kept for compatibility
    gpt_idx_now = 0
    # for res in tqdm(query_reses):
    try:
        if assist_query_method == 'llm':
            # this branch is for use llm to get the assist query
            if dataset == 'scienceqa':
                question_line = 'Question: ' + res['question_text']
            else:
                question_line = 'Question: ' + res['question']

            candidates_dict = res['candidates_dict']
            # Handle both old dict format and new list format for backward compatibility
            if isinstance(candidates_dict, dict):
                candidates_answer_list = list(candidates_dict.keys())
            else:
                candidates_answer_list = candidates_dict  # Already a list
            candidate_answers_line = 'Candidate answers:'
            candidates_answer_list = candidates_answer_list[:2]
            
            # Generate caption if not available or empty
            if 'captions' not in res or not res['captions'] or not res['captions'][0].strip():
                # Get image path to generate caption
                if dataset == 'vqa':
                    image_root_path = '/mnt/VLAI_data/VQAv2/val2014/'
                    image_id = str(res['image_id'])
                    image_id_filled = image_id.zfill(12)  # VQAv2 uses 12 digits
                    image_name = f'COCO_val2014_{image_id_filled}.jpg'
                    image_path = image_root_path + image_name
                    generated_caption = generate_image_caption(image_path)
                    res['captions'] = [generated_caption]
                    print(f"Generated caption: {generated_caption}")
                else:
                    res['captions'] = ["An image"]  # Fallback
            
            caption_line = 'Caption: ' + res['captions'][0]
            for c_canswer in candidates_answer_list:
                if c_canswer == candidates_answer_list[-1]:
                    candidate_answers_line = candidate_answers_line + ' ' + c_canswer
                else:
                    candidate_answers_line = candidate_answers_line + ' ' + c_canswer + ','
            # print('answer line::->', candidate_answers_line)
            # get_assist_query_prompt = prompt + '\n\n'+ question_line + '\n\n' + caption_line + '\n\n' + candidate_answers_line + '\n\n' + 'Relevant issues:'
            get_assist_query_prompt = prompt + '\n\n'+ question_line + '\n\n' + caption_line + '\n\n' + candidate_answers_line + '\n\n' + 'Assistive questions:'
            print('final assist prompt:', get_assist_query_prompt)
            
            # Use local LLM instead of API
            res_assist_query = local_llm_generate(get_assist_query_prompt)
            print('geted_res:', res_assist_query)
            
            # ready to get the re context
            
            # for short version
            
            
            match = re.search(r'<list>(.*?)</list>', res_assist_query)
            if match:
                res_assist_query = match.group(1)
            # print('after re res:', res_assist_query)
            
            
            
            res_assist_query_eval = eval(res_assist_query)

        elif assist_query_method=='vlm':
            prompt = 'Reasoning Question : Is the banana ripe enough to eat?\n'+ \
                     'Perception Question : Is the banana yellow ?\n' + \
                     'Reasoning Question : Is it cold outside ?\n' + \
                     'Perception Question : Are any people wearing jackets ?\n' + \
                     'Reasoning Question : {}\n' + \
                     'Perception Question : '
            prompt = prompt.format(res['question'])
            print('input assist query prompt is:', prompt)
            
            # Get image path based on dataset
            if dataset =='vqa':
                image_root_path = '/mnt/VLAI_data/VQAv2/val2014/'
                image_id = str(res['image_id'])
                image_id_filled = image_id.zfill(12)  # VQAv2 uses 12 digits
                image_name = f'COCO_val2014_{image_id_filled}.jpg'
                image_path = image_root_path + image_name
            else:
                raise ValueError(f'Unknown dataset: {dataset}')
            
            # Use local VLM model
            res_assist_query_eval = vlm_assist_query(image_path, prompt)
            # print(res_assist_query_eval, type(res_assist_query_eval))
            # assert False
        # Get image path based on dataset for final processing
        if dataset =='vqa':
            image_root_path = '/mnt/VLAI_data/VQAv2/val2014/'
            image_id = str(res['image_id'])
            image_id_filled = image_id.zfill(12)  # VQAv2 uses 12 digits
            image_name = f'COCO_val2014_{image_id_filled}.jpg'
            image_path = image_root_path + image_name         
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
            
        # No need for BLIP2 preprocessing anymore, use image path directly
        assist_info = {}
        for assist_query in res_assist_query_eval:
            # Use local VLM for VQA answering instead of BLIP2
            assist_answer, assist_candidates = vqa_answer(image_path, assist_query)
            # print('assist query:', assist_query)
            # print('assist answer:',assist_answer)
            # print('assist candidates:', assist_candidates)
            assist_answers_list = list(assist_candidates.keys())
            
            # Ensure we have at least 2 candidates for downstream processing
            if len(assist_answers_list) < 2:
                # Add a default alternative if beam search didn't produce enough candidates
                # This maintains compatibility with the original pipeline expectations
                if assist_answers_list:
                    # Use the first answer as base for creating a dummy alternative
                    main_answer = assist_answers_list[0]
                    default_alt = "unknown" if "unknown" not in assist_candidates else "other"
                    # Assign a lower log-probability for the dummy candidate
                    min_score = min(assist_candidates.values()) if assist_candidates else -10.0
                    assist_candidates[default_alt] = min_score - 1.0
                else:
                    # Fallback if no candidates at all
                    assist_candidates["unknown"] = -5.0
                    assist_candidates["other"] = -10.0
                assist_answers_list = list(assist_candidates.keys())
            
            # Take top 2 candidates
            assist_answers_list = assist_answers_list[:2]
            
            assist_candidates_top2 = {assist_answers_list[0]:assist_candidates[assist_answers_list[0]],
                                      assist_answers_list[1]:assist_candidates[assist_answers_list[1]]}
            # print(assist_candidates_top2)
            # print(assist_candidates)
            assist_info[assist_query] = {}
            assist_info[assist_query]['assist_answer'] = assist_answer
            
            assist_info[assist_query]['assist_candidates'] = assist_candidates_top2
            
            # trans scores to probability
            # Now using REAL log-probabilities from beam search (like BLIP2)
            assist_candidates_dict = assist_candidates_top2 #res['assist_info'][assist_info_key]['assist_candidates']
            values = list(assist_candidates_dict.values())  # Real log-probabilities
            probabilities = np.exp(values) / np.sum(np.exp(values))  # Proper softmax normalization
            prob_res = {}
            keys = list(assist_candidates_dict.keys())
            for i in range(len(keys)):
                prob_res[keys[i]] = probabilities[i]
            assist_info[assist_query]['assist_candidates'] = prob_res
            
        res['assist_info'] = assist_info
    except Exception as e:
        print('error in get assist query')
        print(e)
        # continue
    return res
    # return query_reses




def get_statements_json_format_4_single(dataset, res, prompt, idx):
    # for res in tqdm(query_reses):
    try:
        assist_info = res['assist_info']
        assist_queries = list(assist_info.keys())
        # Handle both old dict format and new list format for candidates_dict
        if isinstance(res['candidates_dict'], dict):
            query_candidates = list(res['candidates_dict'].keys())
        else:
            query_candidates = res['candidates_dict']  # Already a list
        for assist_query in assist_queries:
            assist_query_info = assist_info[assist_query]
            assist_query_candidates = list(assist_query_info['assist_candidates'].keys())
            assist_info[assist_query]['statements_info'] = {}
            
            
            # from here get the statement info
            if dataset == 'scienceqa':
                question_line = 'Question: ' + res['question_text']
            else:
                question_line = 'Question: ' + res['question']
            answer_line = 'Answers: ' + query_candidates[0] + ', ' + query_candidates[1]
            priori_question_line = 'Priori Question: ' + assist_query
            priori_answer_line = 'Priori Answers: ' +  assist_query_candidates[0] + ',' + assist_query_candidates[1]
            get_statements_prompt = prompt + '\n\n' + question_line + '\n\n' + answer_line + '\n\n' + priori_question_line + '\n\n' + priori_answer_line + '\n\n' +'statements: '
            geted_statements = local_llm_generate(get_statements_prompt)
            # print('before re statements:', geted_statements)
            
            # for short version
            match = re.search(r'<convert>(.*?)</convert>|json\s+(.*?)\s*|```json\s+(.*?)\s+```|<convert>(.*?)', geted_statements, re.DOTALL)
            if match:
                geted_statements = match.group(1) or match.group(2) or match.group(3) or match.group(4)
            # for short version
            
            # print('after re statements:', geted_statements)
            assist_info[assist_query]['statements_info_ori'] = geted_statements
            
            try:
                eval_geted_statements = eval(geted_statements)
                for assist_query_answer in assist_query_candidates:
                    assist_query_key = 'Priori Answer: ' + assist_query_answer
                    statement_info_list = []
                    for query_answer in query_candidates:
                        question_key = 'Question Answer: ' + query_answer
                        # assist_info[assist_query]['statements_info_ori'][assist_query_answer] = geted_statements
                        # print('Question key: ', question_key)
                        # print('Assist key: ', assist_query_key)
                        # print('eval_statement: ', eval_geted_statements)
                        if question_key in eval_geted_statements and assist_query_key in eval_geted_statements[question_key]:
                            statement_info_list.append(eval_geted_statements[question_key][assist_query_key])
                        else:
                            # Fallback statement if parsing fails
                            statement_info_list.append(f"Statement about {query_answer} and {assist_query_answer}")
                        # print(geted_statements)
                    assist_info[assist_query]['statements_info'][assist_query_answer] = str(statement_info_list)
            except Exception as e:
                print(f"Statement parsing error: {e}")
                # Create fallback statements for all combinations
                for assist_query_answer in assist_query_candidates:
                    fallback_statements = []
                    for query_answer in query_candidates:
                        fallback_statements.append(f"Statement about {query_answer} and {assist_query_answer}")
                    assist_info[assist_query]['statements_info'][assist_query_answer] = str(fallback_statements)
    except Exception as e:
        print(e)
        print('error in trans statement!!')
    return res




def get_probability_json_format_single_with_caption(res, prompt, idx, with_caption):
    while True:
        try:
            assist_info = res['assist_info']
            assist_queries = list(assist_info.keys())
            # Handle both old dict format and new list format for candidates_dict
            if isinstance(res['candidates_dict'], dict):
                query_candidates = list(res['candidates_dict'].keys())
            else:
                query_candidates = res['candidates_dict']  # Already a list
            if with_caption:
                caption = res['captions'][0]
            for assist_query in assist_queries:
                assist_answer_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                assist_info[assist_query]['statements_prob'] = {}
                assist_info[assist_query]['statements_prob_norm'] = {}
                for assist_answer in assist_answer_candidates:
                    # print(assist_info[assist_query])
                    geted_statement = assist_info[assist_query]['statements_info'][assist_answer]
                    if geted_statement[0] == '\'':
                        geted_statement = '['+ geted_statement + ']'
                    eval_geted_statement = eval(geted_statement)
                    # format_temp = 'This is a scene {}. In above scene, {}'.format()
                    # re_format_statement = [caption+','+eval_geted_statement[0], caption+','+eval_geted_statement[1]]
                    if with_caption:
                        re_format_statement = ['This is a scene of '+caption+'. In the above scene, '+eval_geted_statement[0], 
                                            'This is a scene of '+caption+'. In the above scene, '+eval_geted_statement[1]]
                        eval_geted_statement = re_format_statement
                    else:
                        re_format_statement = [eval_geted_statement[0], eval_geted_statement[1]]
                        pass
                    res['add_caption_statement'] = re_format_statement
                    statement_line = 'Statements: ' + str(re_format_statement)
                    # print('statement line:---------: ', statement_line)
                    geted_probability_prompt = prompt + '\n\n' + statement_line + '\n\n' + 'Probability:'
                    # print('geted probability prompt', geted_probability_prompt)
                    idx = (idx + 1) % 4
                    geted_probability = local_llm_generate(geted_probability_prompt)
                
                    # print('before re prob:', geted_probability)
                    # for short version
                    match = re.search(r'<dict>(.*?)</dict>|json\s+(.*?)\s+`', geted_probability, re.DOTALL)
                    if match:
                        geted_probability = match.group(1) or match.group(2)
                    # for short version
                    # print('after re prob:', geted_probability)
                
                
                
                    assist_info[assist_query]['statements_prob'][assist_answer] = geted_probability
                    try:
                        eval_geted_probability = eval(geted_probability)
                        sum_probability = eval_geted_probability[eval_geted_statement[0]] + eval_geted_probability[eval_geted_statement[1]]
                        assist_info[assist_query]['statements_prob_norm'][assist_answer] = [eval_geted_probability[eval_geted_statement[0]]/sum_probability, eval_geted_probability[eval_geted_statement[1]]/sum_probability]
                    except Exception as e:
                        continue
            break
        except Exception as e:
            print(e)
            print('error in res', res)
            break
    return res
    



def run_single(dataset, use_equal_answer, with_caption, res, assist_query_method, get_assist_query_prompt, get_statement_prompt, get_probability_prompt,model, vis_processors, txt_processors, idx):
    # cập nhập idx theo chu kỳ 0-3 -> phân chia tải, load balancing khi có nhiều API endpoint.
    idx = (idx + 1) % 4
    # check equal answer
    if use_equal_answer:
        if res.get('equal_answer', False):  # Use .get() to avoid KeyError
            return res
    
    # Generate candidates_dict if not present using initial VQA answering (Step 1 Responder)
    if 'candidates_dict' not in res or len(res['candidates_dict']) < 2:
        # Get image path based on dataset
        if dataset == 'vqa':
            image_root_path = '/mnt/VLAI_data/VQAv2/val2014/'
            image_id = str(res['image_id'])
            image_id_filled = image_id.zfill(12)  # VQAv2 uses 12 digits
            image_name = f'COCO_val2014_{image_id_filled}.jpg'
            image_path = image_root_path + image_name
            
            # Use initial VQA to generate candidate answers WITHOUT scores (per paper Step 1)
            candidates_list = vqa_answer_initial(image_path, res['question'])
            
            # Store as simple list (no scores per paper Step 1 Responder specification)
            res['candidates_dict'] = candidates_list
            print(f"Generated candidates (Step 1 Responder): {res['candidates_dict']}")
        else:
            print(f"Dataset {dataset} not supported for candidate generation")
            return res
    
    if len(res['candidates_dict'])<2:
        return res
    res = get_assist_query_single(
        dataset=dataset,
        res=res, 
        assist_query_method=assist_query_method, 
        prompt=get_assist_query_prompt, 
        model=model, 
        vis_processors=vis_processors, 
        txt_processors=txt_processors, 
        idx=idx
    )
    # res = get_statements_json_format_single(res=res, prompt=get_statement_prompt, idx=idx)
    res = get_statements_json_format_4_single(
        dataset = dataset, 
        res=res, 
        prompt=get_statement_prompt, 
        idx=idx
    )
    # res = get_probability_json_format_single(res=res, prompt=get_probability_prompt, idx=idx)
    res = get_probability_json_format_single_with_caption(with_caption=with_caption, res=res, prompt=get_probability_prompt, idx=idx)
    # run_res.append(res)
    
    # count = count + 1
    return res
    
    
    pass

def run_mutil_process(dataset, use_equal_answer, with_caption, query_reses, max_workers_num, output_path, assist_query_method, get_assist_query_prompt, get_statement_prompt, get_probability_prompt):
    idx = 0
    run_res = []
    geted_res_id = []
    output_freq = 10
    count = 0
    
    # Load local models only - no need for BLIP2 anymore
    print("Loading local Qwen models...")
    load_local_models()
    
    if assist_query_method == 'vlm':
        print("Using local Qwen VLM for assist query generation and VQA answering...")
    else:
        print("Using local Qwen LLM for assist query generation and local Qwen VLM for VQA answering...")
        
    executor = ThreadPoolExecutor(max_workers=max_workers_num)
    futures = [
        executor.submit(run_single,
                        dataset,
                        use_equal_answer, 
                        with_caption,
                        res,
                        assist_query_method,
                        get_assist_query_prompt, 
                        get_statement_prompt, 
                        get_probability_prompt, 
                        None,  # model - not needed anymore
                        None,  # vis_processors - not needed anymore
                        None,  # txt_processors - not needed anymore
                        idx)
        for res in query_reses if res['question_id'] not in geted_res_id
    ]

    # multi process excute
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_reses)):
        idx = (idx + 1) % 4
        idx = 0
        res = future.result()
        if res['question_id'] in geted_res_id:
            continue
        run_res.append(res)
        geted_res_id.append(res['question_id'])
        print('len::::::::::',len(run_res))
        if count % output_freq == 0:
            json.dump(run_res, open(output_path, "w") , indent=4, ensure_ascii=False)
        count = count + 1
    json.dump(run_res, open(output_path, "w"), indent=4, ensure_ascii=False)
    pass


def get_args():
    parser = argparse.ArgumentParser(description='step_eval_mutil_process_api_zoo')
    parser.add_argument('--dataset', type=str, default='winoground')    # gqa scienceqa vqa vqa_rad art_vqa winoground a_okvqa
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepSeek-llm-67b-chat')
    parser.add_argument('--assist_query_method', type=str, default='llm')   # vlm / llm
    parser.add_argument('--assist_query_prompt_path', type=str, default='')
    parser.add_argument('--with_caption', type=bool, default=True) # False True
    parser.add_argument('--use_equal_answer', type=bool, default=False) # False True
    parser.add_argument('--transfor_statement_prompt_path', type=str, default='')
    parser.add_argument('--probability_prompt_path', type=str, default='')
    parser.add_argument('--query_file_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--prompt_version', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    model_name = "local-qwen"  # Set to indicate using local models
    with_caption = args.with_caption
    use_equal_answer = args.use_equal_answer
    assist_query_method = args.assist_query_method
    assist_query_prompt_path = args.assist_query_prompt_path
    transfor_statement_prompt_path = args.transfor_statement_prompt_path
    probability_prompt_path = args.probability_prompt_path
    query_file_path = args.query_file_path
    output_path = args.output_path
    prompt_version = args.prompt_version

    output_file_path = '{}/{}/{}/{}/step_1.json'.format(output_path, "local-qwen", prompt_version, dataset)
    
    output_folder_path = output_file_path.rsplit('/',maxsplit=1)[0]

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    queries = json.load(open(query_file_path, "r"))
    print('query len: ', len(queries))
    
    # Limit to first 50 samples for quick testing
    if len(queries) > 50:
        queries = queries[:50]
        print(f'Limited to first 50 samples for testing. New query len: {len(queries)}')
    
    # assert False
    with open(assist_query_prompt_path) as f:
        assist_query_prompt = f.read().strip()
    with open(transfor_statement_prompt_path) as f:
        statement_prompt = f.read().strip()
    with open(probability_prompt_path) as f:
        probability_prompt = f.read().strip()

    run_mutil_process(dataset=dataset,
                      use_equal_answer = use_equal_answer, 
                      with_caption = with_caption, 
                      query_reses=queries, 
                      max_workers_num=8, 
                      output_path=output_file_path, 
                      assist_query_method = assist_query_method, 
                      get_assist_query_prompt=assist_query_prompt, 
                      get_statement_prompt=statement_prompt, 
                      get_probability_prompt=probability_prompt)
    
    pass