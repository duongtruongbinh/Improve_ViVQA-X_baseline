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

class AnswerNormalizer:
    """Answer normalizer to handle common VQA answer variations"""
    
    def __init__(self):
        self.normalizations = {
            'colors': {
                'dark blue': 'blue', 'light blue': 'blue', 'navy': 'blue',
                'maroon': 'red', 'crimson': 'red', 'dark red': 'red',
                'forest green': 'green', 'lime': 'green', 'dark green': 'green',
                'light green': 'green', 'bright green': 'green',
                'light yellow': 'yellow', 'bright yellow': 'yellow',
                'dark brown': 'brown', 'light brown': 'brown',
                'dark gray': 'gray', 'light gray': 'gray', 'grey': 'gray',
                'dark grey': 'gray', 'light grey': 'gray',
                'off white': 'white', 'cream': 'white',
                'dark purple': 'purple', 'light purple': 'purple',
                'pink': 'red',  # VQA often treats pink as red
                'orange': 'orange'
            },
            'numbers': {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 
                'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                'eight': '8', 'nine': '9', 'ten': '10',
                'eleven': '11', 'twelve': '12', 'thirteen': '13',
                'fourteen': '14', 'fifteen': '15', 'sixteen': '16',
                'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
                'twenty': '20'
            },
            'objects': {
                'automobile': 'car', 'vehicle': 'car', 'truck': 'car',
                'canine': 'dog', 'puppy': 'dog', 'feline': 'cat', 'kitten': 'cat',
                'laptop': 'computer', 'notebook': 'computer', 'pc': 'computer',
                'cellphone': 'phone', 'smartphone': 'phone', 'mobile': 'phone',
                'bicycle': 'bike', 'motorbike': 'motorcycle',
                'airplane': 'plane', 'aircraft': 'plane',
                'person': 'man', 'people': 'man', 'woman': 'man',  # VQA often uses 'man' generically
                'building': 'house', 'structure': 'building'
            },
            'yes_no': {
                'yeah': 'yes', 'yep': 'yes', 'correct': 'yes', 'true': 'yes',
                'nope': 'no', 'false': 'no', 'incorrect': 'no', 'nah': 'no'
            }
        }
    
    def detect_question_type(self, question):
        """Detect the type of question to apply appropriate normalization"""
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ['what color', 'color of', 'what colour']):
            return 'color'
        elif any(phrase in question_lower for phrase in ['how many', 'count', 'number of']):
            return 'counting'
        elif any(phrase in question_lower for phrase in ['is there', 'are there', 'can you see']):
            return 'yes_no'
        else:
            return 'general'
    
    def normalize_answer(self, answer, question=None):
        """Normalize answer to match VQA evaluation standards"""
        if not answer:
            return answer
            
        original_answer = answer
        answer = answer.lower().strip()
        
        # Remove common prefixes/suffixes
        answer = re.sub(r'^(the |a |an )', '', answer)
        answer = re.sub(r'(\.|,|!|\?)$', '', answer)
        
        question_type = self.detect_question_type(question) if question else 'general'
        
        # Number normalization (priority for counting questions)
        if question_type == 'counting' or answer.isdigit() or answer in self.normalizations['numbers']:
            if answer in self.normalizations['numbers']:
                return self.normalizations['numbers'][answer]
            # Extract first number if answer contains multiple
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]
        
        # Yes/No normalization
        if question_type == 'yes_no' or answer in ['yes', 'no'] or answer in self.normalizations['yes_no']:
            if answer in self.normalizations['yes_no']:
                return self.normalizations['yes_no'][answer]
            if answer in ['yes', 'no']:
                return answer
        
        # Color normalization
        if question_type == 'color' or any(color in answer for color in self.normalizations['colors']):
            for variant, standard in self.normalizations['colors'].items():
                if answer == variant or variant in answer:
                    return standard
        
        # Object normalization
        for variant, standard in self.normalizations['objects'].items():
            if answer == variant:
                return standard
        
        # Return first word if answer is too long (common in VQA)
        words = answer.split()
        if len(words) > 2 and question_type != 'general':
            return words[0]
            
        return answer if answer else original_answer

def inference(device, args, test_loader):
    # Building GroundingDINO, LLM, VLM, and CLIP_Count models as multi-agents
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    LLM, VLM = QueryLLM(args), QueryVLM(args)
    clip_count = CLIP_Count.load_from_checkpoint('CLIP_Count/ckpt/clipcount_pretrained.ckpt', strict=False).to(device)
    clip_count.eval()  # Set the model to evaluation mode
    
    # Initialize answer normalizer
    normalizer = AnswerNormalizer()

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

            # Enhanced trigger logic with multiple conditions
            baseline_answer = answer[0].strip()
            
            # Detect if we're using Qwen2.5 model
            is_qwen25 = hasattr(VLM, 'model_name') and 'qwen2.5' in VLM.model_name.lower()
            
            # 1. EXPLICIT FAILURE DETECTION
            if is_qwen25:
                explicit_failure = any(signal in baseline_answer for signal in ['[Answer Failed]'])
                zero_numeric = '[Zero Numeric Answer]' in baseline_answer
            else:
                explicit_failure = any(signal in baseline_answer for signal in ['[Answer Failed]', 'sorry', "I don't know", "I cannot", "unclear"])
                zero_numeric = '[Zero Numeric Answer]' in baseline_answer
            
            nonzero_numeric = '[Non-zero Numeric Answer]' in baseline_answer
            verify_numeric_answer = nonzero_numeric
            
            # 2. SMART ANSWER QUALITY ASSESSMENT
            vague_answer = False
            if is_qwen25:
                # More restrictive for Qwen2.5 - only truly vague answers
                vague_keywords = ['unclear', 'unsure', 'i see something', 'it appears unclear', 'not clear']
                vague_answer = (len(baseline_answer.strip()) == 0 or 
                               baseline_answer.lower().strip() in ['unknown', 'unclear', 'unsure'] or
                               any(keyword in baseline_answer.lower() for keyword in vague_keywords))
            else:
                vague_answer = (len(baseline_answer.strip()) == 0 or 
                               baseline_answer.lower().strip() in ['unknown', 'unsure', 'maybe'])
            
            # 3. REFINED COUNTING QUESTIONS DETECTION  
            is_counting_question = any(phrase in question[0].lower() for phrase in ['how many', 'what number', 'count', 'number of'])
            counting_trigger = False
            if is_counting_question and not zero_numeric and not nonzero_numeric and not explicit_failure:
                try:
                    import re
                    numbers = re.findall(r'\d+', baseline_answer)
                    if numbers:
                        largest_num = max(int(num) for num in numbers)
                        # REFINED: Increase threshold - only for truly difficult counting
                        threshold = 8 if is_qwen25 else 8  # Increase from 3 to 8
                        if largest_num >= threshold:  
                            verify_numeric_answer = True
                            counting_trigger = True
                            if args['inference']['verbose']:
                                print(f'{Colors.WARNING}High count detected: "{baseline_answer}" (>={threshold}), triggering verification{Colors.ENDC}')
                    
                    # REFINED: More selective textual number detection
                    high_textual_numbers = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'twenty', 'thirty']
                    if any(num in baseline_answer.lower() for num in high_textual_numbers):
                        verify_numeric_answer = True
                        counting_trigger = True
                        if args['inference']['verbose']:
                            print(f'{Colors.WARNING}High textual number detected: "{baseline_answer}", triggering verification{Colors.ENDC}')
                    
                    # Keep vague counting words detection
                    vague_counting = ['many', 'several', 'few', 'some', 'lots', 'multiple']
                    if any(word in baseline_answer.lower() for word in vague_counting):
                        verify_numeric_answer = True
                        counting_trigger = True
                        if args['inference']['verbose']:
                            print(f'{Colors.WARNING}Vague counting word detected: "{baseline_answer}", triggering verification{Colors.ENDC}')
                    
                    # REFINED: More restrictive suspicious pattern detection
                    if not counting_trigger and numbers:
                        count = int(numbers[0])
                        # Only trigger if clearly suspicious patterns
                        if ((count >= 10 and len(baseline_answer.split()) > 5) or  # Long explanation for high count
                            (count % 10 == 0 and count >= 20) or  # Round numbers >=20
                            (count > 15)):  # Very high counts
                            verify_numeric_answer = True
                            counting_trigger = True
                            if args['inference']['verbose']:
                                print(f'{Colors.WARNING}Suspicious counting pattern: "{baseline_answer}", triggering verification{Colors.ENDC}')
                except:
                    pass
            
            # 4. SELECTIVE VISUAL COMPLEXITY ASSESSMENT
            question_lower = question[0].lower()
            complex_visual_question = False
            
            # Only trigger complex visual for truly complex questions
            spatial_keywords = ['relationship between', 'behind', 'in front of', 'between', 'compared to']
            has_spatial = any(keyword in question_lower for keyword in spatial_keywords)
            
            # Complex reasoning requiring multiple objects analysis
            complex_keywords = ['different from', 'similar to', 'relationship', 'interaction']
            has_complex_reasoning = any(keyword in question_lower for keyword in complex_keywords)
            
            # Multiple objects but only if answer seems insufficient
            multi_object_keywords = ['people are', 'animals are', 'all the', 'both', 'each of the']
            has_multi_objects = any(keyword in question_lower for keyword in multi_object_keywords)
            
            # CRITICAL FIX: Only trigger attribute questions if answer seems wrong or insufficient
            detailed_attribute = False
            if any(phrase in question_lower for phrase in ['what color', 'what type', 'what kind', 'what material', 'what size']):
                color_question = 'color' in question_lower
                if color_question:
                    # REFINED: Only trigger color questions if answer is clearly inadequate
                    common_colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey', 'orange', 'purple', 'pink', 'silver']
                    answer_has_color = any(color in baseline_answer.lower() for color in common_colors)
                    # REFINED: More restrictive - only trigger if no clear color AND uncertainty
                    if (not answer_has_color and 
                        (len(baseline_answer.split()) > 4 or  # Long non-color answer
                         any(word in baseline_answer.lower() for word in ['appears', 'seems', 'looks', 'unclear']))):
                        detailed_attribute = True
                        if args['inference']['verbose']:
                            print(f'{Colors.WARNING}Color question unclear: "{baseline_answer}"{Colors.ENDC}')
                else:
                    # REFINED: More restrictive attribute triggers  
                    attribute_keywords = ['what type', 'what kind', 'what material', 'what shape', 'what size', 'how big', 'how small']
                    if any(phrase in question_lower for phrase in attribute_keywords):
                        # REFINED: Only trigger if answer is clearly insufficient
                        generic_answers = ['object', 'thing', 'item', 'something', 'material', 'type', 'stuff']
                        if (baseline_answer.lower().strip() in generic_answers or
                            (len(baseline_answer.split()) <= 1 and baseline_answer.lower() not in ['yes', 'no']) or
                            any(word in baseline_answer.lower() for word in ['unclear', 'unknown', 'unsure'])):
                            detailed_attribute = True
                            if args['inference']['verbose']:
                                print(f'{Colors.WARNING}Attribute question insufficient: "{baseline_answer}"{Colors.ENDC}')
            
            # REFINED: More selective complex visual assessment
            long_question = len(question[0].split()) > 15  # Increased threshold from 12 to 15
            
            # REFINED: Only trigger complex visual for truly problematic cases
            if (has_spatial or has_complex_reasoning or 
                (has_multi_objects and len(baseline_answer.split()) <= 1) or  # Only very short answers
                detailed_attribute or  
                (long_question and len(baseline_answer.split()) <= 1 and baseline_answer.lower() not in ['yes', 'no'])):  # Exclude yes/no
                complex_visual_question = True
                if args['inference']['verbose']:
                    triggers = []
                    if has_spatial: triggers.append("spatial_relationship")
                    if has_complex_reasoning: triggers.append("complex_reasoning") 
                    if has_multi_objects: triggers.append("multi_objects_very_short")
                    if detailed_attribute: triggers.append("attribute_insufficient")
                    if long_question: triggers.append("long_question_inadequate")
                    print(f'{Colors.WARNING}Complex visual question detected ({", ".join(triggers)}): "{question[0][:50]}..."{Colors.ENDC}')
            
            # 5. REFINED CONFIDENCE ASSESSMENT
            low_confidence_answer = False
            
            # More restrictive uncertainty detection
            uncertainty_indicators = ['not sure', 'unclear', 'i think', 'unsure', 'unknown']  # Removed common words
            has_uncertainty = any(indicator in baseline_answer.lower() for indicator in uncertainty_indicators)
            
            # More selective short answer detection
            too_short_for_complex = (len(question[0].split()) > 15 and len(baseline_answer.split()) <= 1 and  # Increased from 10 to 15
                                    baseline_answer.lower() not in ['yes', 'no'] and  # Exclude valid yes/no
                                    not any(phrase in question_lower for phrase in ['is this', 'is there', 'is it', 'are there', 'can you see']))
            
            # REFINED: More restrictive generic answers
            truly_generic_answers = ['object', 'thing', 'item', 'something', 'nothing', 'unclear', 'unknown', 'stuff', 'things']
            is_generic = baseline_answer.lower().strip() in truly_generic_answers
            
            if has_uncertainty or too_short_for_complex or is_generic:
                low_confidence_answer = True
                if args['inference']['verbose']:
                    reasons = []
                    if has_uncertainty: reasons.append("uncertainty_words")
                    if too_short_for_complex: reasons.append("inadequate_for_very_complex")
                    if is_generic: reasons.append("truly_generic_answer")
                    print(f'{Colors.WARNING}Low confidence answer detected ({", ".join(reasons)}): "{baseline_answer}"{Colors.ENDC}')
            
            # REFINED: Much more restrictive location triggers
            location_trigger = False
            location_keywords = ['where is', 'where are', 'where does', 'where can', 'what place', 'which place']
            if any(phrase in question_lower for phrase in location_keywords):
                # REFINED: Only trigger if answer is truly vague or non-location
                truly_vague_locations = ['there', 'here', 'somewhere', 'place', 'location', 'area', 'unknown']
                if (baseline_answer.lower().strip() in truly_vague_locations or
                    (len(baseline_answer.strip()) == 0) or
                    (any(word in baseline_answer.lower() for word in ['unclear', 'unknown', 'not sure']) and 
                     len(baseline_answer.split()) <= 2)):
                    location_trigger = True
                    if args['inference']['verbose']:
                        print(f'{Colors.WARNING}Location question truly unclear: "{baseline_answer}"{Colors.ENDC}')

            # 6. SELECTIVE CONTEXTUAL REASONING
            needs_context = False
            # Only trigger for questions that truly need contextual reasoning
            context_keywords = ['why is', 'reason for', 'purpose of', 'why are', 'what is the purpose']
            if any(keyword in question_lower for keyword in context_keywords):
                # Only if answer is inadequate for the why/purpose question
                if len(baseline_answer.split()) <= 2 or baseline_answer.lower().strip() in ['yes', 'no', 'unknown']:
                    needs_context = True
                    if args['inference']['verbose']:
                        print(f'{Colors.WARNING}Contextual reasoning needed: "{question[0][:50]}..."{Colors.ENDC}')
            
            # 7. REFINED TRIGGER DECISION - Include location_trigger but more selective
            match_baseline_failed = (explicit_failure or zero_numeric or vague_answer or 
                                   counting_trigger or complex_visual_question or 
                                   low_confidence_answer or needs_context or location_trigger)
            
            # IMPROVED: Giảm override conditions - chỉ override cho very simple cases
            simple_question_types = ['is this', 'is there', 'is it', 'are there', 'can you see', 'does the', 'do you see']
            is_simple_question = any(phrase in question_lower for phrase in simple_question_types)
            simple_answer = baseline_answer.lower().strip() in ['yes', 'no'] or len(baseline_answer.split()) == 1
            
            # IMPROVED: Không override important question types
            important_question_types = ['how many', 'what color', 'where is', 'what type', 'what kind', 'what material', 'what shape', 'what size']
            is_important_question = any(phrase in question_lower for phrase in important_question_types)
            
            if is_simple_question and simple_answer and not explicit_failure and not zero_numeric and not is_important_question:
                match_baseline_failed = False  # Override trigger for simple Q&A pairs
                if args['inference']['verbose'] and (counting_trigger or complex_visual_question or low_confidence_answer):
                    print(f'{Colors.OKBLUE}Override: Simple question with reasonable answer, skipping multi-agent{Colors.ENDC}')
            
            # Enhanced trigger reasons for verbose output  
            if match_baseline_failed and args['inference']['verbose']:
                trigger_reasons = []
                if explicit_failure: trigger_reasons.append("explicit_failure")
                if zero_numeric: trigger_reasons.append("zero_numeric")
                if vague_answer: trigger_reasons.append("vague_answer")
                if counting_trigger: trigger_reasons.append("counting_verification")
                if complex_visual_question: trigger_reasons.append("complex_visual") 
                if low_confidence_answer: trigger_reasons.append("low_confidence")
                if needs_context: trigger_reasons.append("needs_context")
                if location_trigger: trigger_reasons.append("location_trigger")
                
                if verify_numeric_answer:
                    msg = f"Multi-agent triggered for counting verification. Reasons: {', '.join(trigger_reasons)}"
                else:
                    msg = f"Multi-agent triggered for enhanced analysis. Reasons: {', '.join(trigger_reasons)}"
                print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

            # Use multi-agent approach when triggered
            if match_baseline_failed:
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

            # Apply answer normalization to both baseline and final answers
            normalized_baseline = normalizer.normalize_answer(baseline_answer, question[0])
            normalized_final = normalizer.normalize_answer(final_answer, question[0])
            
            if args['inference']['verbose'] and normalized_final != final_answer:
                print(f'{Colors.OKBLUE}Answer normalized: "{final_answer}" → "{normalized_final}"{Colors.ENDC}')

            # Use VQA evaluation from utils with normalized answers
            gt_answers = [{'answer': target_answer[0]}] * 3  # VQA format with 3 annotators
            vqa_score = grader.vqa_score(normalized_final, gt_answers)
            baseline_score = grader.vqa_score(normalized_baseline, gt_answers)
            
            if args['inference']['verbose']:
                print(f"VQA Score: {vqa_score:.3f} [{'Correct' if vqa_score > 0 else 'Incorrect'}]")

            # Use grader's accumulate_grades for proper tracking with normalized answer
            majority_vote = grader.accumulate_grades(args, [], match_baseline_failed, 
                                                   target_answer=gt_answers, 
                                                   model_answer=normalized_final, 
                                                   answer_type='unknown',
                                                   initial_answer=normalized_baseline)

            # Record response
            response_dict = {
                'image_id': str(image_id[0].item()), 
                'image_path': image_path[0], 
                'question_id': str(question_id[0].item()), 
                'question': question[0], 
                'target_answer': target_answer[0],
                'initial_answer': baseline_answer,
                'normalized_baseline': normalized_baseline,
                'final_answer': final_answer,
                'normalized_final': normalized_final,
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
