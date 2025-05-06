import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import cv2
import base64
import requests
import concurrent.futures
from torchvision.ops import box_convert
import re

import autogen
from autogen.agentchat.assistant_agent import AssistantAgent

# gemini
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image as vertexai_Image

class QueryVLMAutoGen(AssistantAgent):
    def __init__(self, name, args, llm_config=None, system_message="You are a VLM assistant.", **kwargs):
        super().__init__(name=name, llm_config=llm_config, system_message=system_message, **kwargs)
        self.image_cache = {}
        self.image_size = args.get('vlm', {}).get('image_size', 512) # Adjusted to match typical arg structure
        self.min_bbox_size = args['vlm']['min_bbox_size']
        self.args = args
        self.vlm_type = args["model"]

        # Preserve original API key loading and Gemini initialization
        # Ensure "openai_key.txt" is accessible or handle path appropriately
        try:
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read().strip()
        except FileNotFoundError:
            print("Warning: openai_key.txt not found. GPT-4V queries will fail if self.api_key is not set otherwise.")
            self.api_key = None # Or raise an error, or expect it in args

        if self.vlm_type=="gemini":
            print("Using Gemini Pro Vision as VLM, initializing the model")
            # Ensure Google Cloud credentials are set up for Vertex AI
            try:
                self.gemini_pro_vision = GenerativeModel("gemini-1.0-pro-vision")
            except Exception as e:
                print(f"Failed to initialize Gemini Pro Vision: {e}")
                self.gemini_pro_vision = None


    def process_image(self, image, bbox=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            # Ensure bbox is a tensor for box_convert if it's not already
            if not isinstance(bbox, torch.Tensor):
                # Assuming bbox is [cx, cy, w, h] as per original context implies
                # This part might need adjustment based on actual bbox format in self.args
                # For this example, assuming bbox is passed as a list/numpy array [cx, cy, w, h]
                bbox_tensor = torch.tensor([bbox]) 
            else:
                bbox_tensor = bbox

            # Original script had width, height = bbox[1], bbox[2]
            # This implies bbox was expected to be a single bounding box, not a batch
            # If bbox is [cx, cy, w, h]
            width, height = bbox_tensor.item(), bbox_tensor.item()
            xyxy = box_convert(boxes=bbox_tensor, in_fmt="cxcywh", out_fmt="xyxy").squeeze(0)
            x1, y1, x2, y2 = int(xyxy), int(xyxy[3]), int(xyxy[1]), int(xyxy[2])

            if width < self.min_bbox_size:
                x1 = int(max(0, x1 - (self.min_bbox_size - width) / 2))
                x2 = int(min(image.shape[3], x2 + (self.min_bbox_size - width) / 2))
            if height < self.min_bbox_size:
                y1 = int(max(0, y1 - (self.min_bbox_size - height) / 2))
                y2 = int(min(image.shape, y2 + (self.min_bbox_size - height) / 2))
            
            image = image[y1:y2, x1:x2]

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()
        if self.vlm_type == "gpt4":
            image_bytes = base64.b64encode(image_bytes).decode('utf-8')
        
        return image_bytes

    def messages_to_answer_directly_gemini(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            # "list each one by [Object i] where i is the index." \
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + \
                      "', explain what the question wants to ask, what objects or objects with specific attributes is related to the question, " \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, provide your final answer which starts with the notation '[Answer]' at the end. " \
                      "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are four different cases for the final answer part. " \
                      "(Case 1) If you believe your answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Answer]' based on your decision. " \
                      "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                      "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                      "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                      "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                      "Then, step-by-step describe each object in this image that satisfy the descriptions in the question. " \
                      "If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. " \
                      "If there are less or equal to three objects in the image, you should start your final answer with '[Numeric Answer]' instead of '[Answer]', answer your predicted number right after '[Numeric Answer]'. " \
                      "If you believe there are many(larger than three objects) in the image, you should answer '[Non-zero Numeric Answer] [Answer Failed]' instead of '[Answer]', provide your predicted number right after '[Non-zero Numeric Answer]'. Avoid being too confident. " \
                      "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Answer]'. No extra words after '[Answer]'. " \
                      "(Case 4) If you think you can't answer the question directly, or you need more information, or you find that your answer could be wrong, " \
                      "do not make a guess. Instead, explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and answer '[Answer Failed]' instead of '[Answer]'. Keep your answers short."
        else:
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
                      "Begin your final answer with the notation '[Answer]'. " \
                      "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        return message


    def messages_to_answer_directly(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image. " \
                      "Finally, begin your final answer with the notation '[Answer]'. " \
                      "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
                      "(Case 1) If you believe your answer falls into the category of 'yes/no', say 'yes/no' after '[Answer]'. " \
                      "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                      "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                      "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                      "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                      "Begin your answer with '[Numeric Answer]', step-by-step describe each object in this image that satisfy the descriptions in the question, " \
                      "list each one by [Object i] where i is the index, and finally predict the number. " \
                      "If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. " \
                      "If there are many of them, for example, more than three in the image, you should answer '[Non-zero Numeric Answer]' and '[Answer Failed]'. and avoid being too confident. " \
                      "(Case 3) If the answer should be an activity or a noun, say the word after '[Answer]'. Similarly, no extra words after '[Answer]'. " \
                      "(Case 4) If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'. Keep your answers short"
        else:
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
                      "Begin your final answer with the notation '[Answer]'. " \
                      "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        return message


    def message_to_check_if_answer_is_numeric(self, question):
        message = "You are performing a Visual Question Answering task. " \
                  "Given the image and the question '" + question + "', please first verify if the question type is like 'how many' or 'what number of' and asks you to count the number of an object. " \
                  "If not, say '[Not Numeric Answer]' and explain why. " \
                  "Otherwise, find which object you need to count, say '[Numeric Answer]', and predict the number. "
        return message


    def messages_to_query_object_attributes(self, question, phrase=None, verify_numeric_answer=False):
        if verify_numeric_answer:
            message = "Describe the " + phrase + " in each image in one sentence that can help you answer the question '" + question + "' and count the number of " + phrase + " in the image. "
        else:
            # We expect each object to offer a different perspective to solve the question
            # message = "Describe the attributes and the name of the object related to answer the question '" + question + "' in one sentence."
            message = "Describe the attributes and the name of the object in the image in one sentence, " \
                  "including visual attributes like color, shape, size, materials, and clothes if the object is a person, " \
                  "and semantic attributes like type and current status if applicable. " \
                  "Think about what objects you should look at to answer the question '" + question + "' in this specific image, and only focus on these objects." \

            if phrase is not None:
                message += "You need to focus on the " + phrase + " and nearby objects. "

        return message


    def messages_to_reattempt(self, question, obj_descriptions, prev_answer):
        # message = "After a previous attempt to answer the question '" + question + "', the response was not successful, " \
        #           "Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "]. To address this, we've identified additional objects within the image: "                                                                                                                                                                                                                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
        message = "After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
                  "highlighting the need for more detailed object detection and analysis. Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "] " \
                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "

        for i, obj in enumerate(obj_descriptions):
            message += "[Object " + str(i) + "] " + obj + "; "

        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Summarize all the information you have, and then begin your final answer with '[Reattempted Answer]'." \
                       "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
                       "If you believe your answer falls into the category of 'yes/no' or a number, say 'yes/no' or the number after '[Reattempted Answer]'. " \
                       "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                       "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                       "If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                       "step-by-step describe each object in this image that satisfy the descriptions in the question, list each one by [Object i] where i is the index, " \
                       "and finally reevaluated the number after '[Reattempted Answer]'. Objects could be only partially visible." \
                       "If the answer should be an activity or a noun, say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
        else:
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Begin your final answer with '[Reattempted Answer]'."

        return message


    def messages_to_reattempt_gemini(self, question, obj_descriptions, prev_answer):
        # message = "After a previous attempt to answer the question '" + question + "', the response was not successful, " \
        #           "Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "]. To address this, we've identified additional objects within the image: "                                                                                                                                                                                                                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
        message = "You are performing a Visual Question Answering task. After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
                  "highlighting the need for more detailed object detection and analysis. Here is the feedback from that previous attempt for reference: [Previous Failed Answer: " + prev_answer + "] " \
                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "

        for i, obj in enumerate(obj_descriptions):
            message += "[Object " + str(i) + "] " + obj + "; "

        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
            message += "Based on the previous attempt, these descriptions and the image, you need to first list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
                       "Then, given these additional object descriptions that the model previously missed, summarize all the information and re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Finally, provide your own answer to the question: '" + question + "'. Your final answer should starts with notation '[Reattempted Answer]'." \
                       "Your final answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are three different cases for the final answer part. " \
                       "(Case 1) If you believe your final answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Reattempted Answer]'. " \
                       "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                       "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                       "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                       "describe each object in this image that related the descriptions in the question. " \
                       "Finally re-evaluated and say the number after '[Reattempted Answer]'. Objects could be only partially visible." \
                       "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
        else:
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Begin your final answer with '[Reattempted Answer]'."

        return message


    def query_vlm(self, image, question, step='attributes', phrases=None, obj_descriptions=None, prev_answer=None, bboxes=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        responses =

        if step == 'reattempt' or step == 'ask_directly' or bboxes is None or len(bboxes) == 0:
            if self.vlm_type=="gemini":
                response = self._query_gemini_pro_vision(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                         verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            else: # Assuming gpt4
                response = self._query_openai_gpt_4v(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            return [response]

        if len(bboxes) == 1:
            # Original script had bbox = bboxes.squeeze(0) which implies bboxes was a tensor
            # If bboxes is a list of bbox, then bboxes
            bbox = bboxes if isinstance(bboxes, list) else bboxes.squeeze(0)
            phrase = phrases if phrases else None # Assuming phrases correspond to bboxes
            if self.vlm_type == "gemini":
                response = self._query_gemini_pro_vision(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            else: # Assuming gpt4
                response = self._query_openai_gpt_4v(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            responses.append(response)
        else:
            total_num_objects = len(bboxes)
            # The lambda functions in map originally didn't pass all args like verify_numeric_answer, needed_objects, verbose
            # Adding them here for consistency if they were intended to be used in batched calls.
            # However, the original _query_... methods didn't use all of them when phrase/bbox were present.
            # Sticking to original lambda signature for _query calls.
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
                if self.vlm_type == "gemini":
                    # Original lambda for Gemini batch didn't include verify_numeric_answer, needed_objects, verbose
                    # It only passed image, question, step, phrase, bbox, verbose (implicitly from outer scope for verbose)
                    # Replicating that structure.
                    # Note: 'verbose' in the lambda for _query_gemini_pro_vision was not explicitly passed in original,
                    # it would have used the 'verbose' from the outer scope of query_vlm.
                    # For clarity, passing it explicitly if the method signature supports it.
                    # The original _query_gemini_pro_vision takes verbose.
                    batch_responses = list(executor.map(lambda p_bbox, p_phrase: self._query_gemini_pro_vision(image, question, step, phrase=p_phrase, bbox=p_bbox, verbose=verbose), bboxes, phrases))
                else: # Assuming gpt4
                    batch_responses = list(executor.map(lambda p_bbox, p_phrase: self._query_openai_gpt_4v(image, question, step, phrase=p_phrase, bbox=p_bbox, verbose=verbose), bboxes, phrases))
                # Original code had responses.append(response) which would append a list as a single element.
                # Assuming it should be extend.
                responses.extend(batch_responses)
        
        return responses

    def _query_openai_gpt_4v(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        if not self.api_key:
            return "Error: OpenAI API key not configured."
        base64_image = self.process_image(image, bbox)

        messages_text = "" # Renamed from 'messages' to avoid conflict with AutoGen's 'messages' convention
        max_tokens = 0

        if step == 'ask_directly':
            messages_text = self.messages_to_answer_directly(question)
            max_tokens = 400
        elif step == 'check_numeric_answer':
            messages_text = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages_text = self.messages_to_query_object_attributes(question)
            else:
                messages_text = self.messages_to_query_object_attributes(question, phrase)
            max_tokens = 400
        elif step == 'reattempt':
            messages_text = self.messages_to_reattempt(question, obj_descriptions, prev_answer)
            max_tokens = 600
        else:
            raise ValueError('Invalid step')

        completion_text = "" # Default to empty string
        for _ in range(3):
            prompt = {
                "model": "gpt-4-vision-preview", # Or a newer model if preferred
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": messages_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt)
                response.raise_for_status() # Raise an exception for HTTP errors
                response_json = response.json()
            except requests.exceptions.RequestException as e:
                if verbose:
                    print(f"API request failed: {e}")
                completion_text = f"Error: API request failed: {e}"
                continue # Retry
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Failed to decode API response: {e}")
                completion_text = f"Error: Failed to decode API response: {e}"
                continue # Retry


            if 'choices' in response_json and len(response_json['choices']) > 0:
                choice = response_json['choices']
                if choice.get('message') and choice['message'].get('content'):
                    completion_text = choice['message']['content']
                elif choice.get('text'): # Fallback for older completion formats if any
                    completion_text = choice.get('text')
                else:
                    completion_text = "Error: No content in VLM response."

                if verbose:
                    print(f'VLM Response at step {step}: {completion_text}')
            else:
                completion_text = "Error: Invalid response structure from VLM."
                if verbose:
                    print(f'VLM Response at step {step} (invalid structure): {response_json}')


            if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
                break
        
        return completion_text

    def _query_gemini_pro_vision(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        if not self.gemini_pro_vision:
            return "Error: Gemini Pro Vision model not initialized."
        
        byte_image = self.process_image(image, bbox) # process_image now returns bytes for gemini
        vertexai_image_obj = vertexai_Image.from_bytes(byte_image)

        messages_text = "" # Renamed
        max_tokens = 0

        if step == 'ask_directly':
            messages_text = self.messages_to_answer_directly_gemini(question)
            max_tokens = 500
        elif step == 'check_numeric_answer':
            messages_text = self.message_to_check_if_answer_is_numeric(question) # Assuming same prompt structure
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages_text = self.messages_to_query_object_attributes(question) # Assuming same
            else:
                messages_text = self.messages_to_query_object_attributes(question, phrase) # Assuming same
            max_tokens = 400
        elif step == 'reattempt':
            messages_text = self.messages_to_reattempt_gemini(question, obj_descriptions, prev_answer)
            max_tokens = 700
        else:
            raise ValueError('Invalid step')

        completion_text = ""
        for _ in range(3):
            gemini_contents = [messages_text, vertexai_image_obj]
            try:
                response = self.gemini_pro_vision.generate_content(
                    gemini_contents,
                    generation_config={"max_output_tokens": max_tokens},
                    # stream=False # Ensure non-streaming for direct text extraction
                )
                # Accessing response.text directly is common for non-streaming
                completion_text = response.text
            except Exception as e:
                if verbose:
                    print(f'Error querying Gemini VLM at step {step}: {e}')
                # Check for specific safety/block reasons if available in 'e' or response parts
                try:
                    if response.prompt_feedback.block_reason:
                        completion_text = f"Blocked: {response.prompt_feedback.block_reason}"
                        if verbose: print(f"Gemini content blocked: {response.prompt_feedback.block_reason_message}")
                        break # Don't retry if blocked by safety filters
                except: # If response object doesn't have prompt_feedback or text
                    pass
                completion_text = f"Error: Gemini API call failed: {e}" # Fallback error
                continue # Retry

            if verbose:
                print(f'Gemini VLM Response at step {step}: {completion_text}')

            if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
                break
        
        return completion_text