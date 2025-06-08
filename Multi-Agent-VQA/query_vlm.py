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


class QueryVLM:
    def __init__(self, args, image_size=512):
        self.image_cache = {}
        self.image_size = image_size
        self.min_bbox_size = args['vlm']['min_bbox_size']
        self.args = args
        self.vlm_type = args["model"]
        
        # Load VLM provider configuration
        self.vlm_provider = args['vlm'].get('provider', 'openai')
        self.vlm_config = args['vlm'].get(self.vlm_provider, {})
        
        print(f"Using VLM provider: {self.vlm_provider}")
        
        if self.vlm_provider == 'openai':
            # Load OpenAI API key
            api_key_file = self.vlm_config.get('api_key_file', 'openai_key.txt')
            with open(api_key_file, "r") as f:
                self.api_key = f.read().strip()
            self.model_name = self.vlm_config.get('model', 'gpt-4o-mini')
            self.base_url = self.vlm_config.get('base_url', 'https://api.openai.com/v1/chat/completions')
        elif self.vlm_provider in ['vllm_8000', 'vllm_9000']:
            # vLLM server configuration
            self.api_key = self.vlm_config.get('api_key', 'EMPTY')
            self.model_name = self.vlm_config.get('model', 'Qwen/Qwen2.5-VL-3B-Instruct')
            self.base_url = self.vlm_config.get('base_url', 'http://localhost:8000/v1/chat/completions')
            self.max_tokens = self.vlm_config.get('max_tokens', 400)
            self.temperature = self.vlm_config.get('temperature', 0.0)
            print(f"Configured vLLM server: {self.base_url} with model: {self.model_name}")
        else:
            # Fallback to original logic for other providers like gemini
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read().strip()

        # Init gemini model
        if self.vlm_type=="gemini":
            print("Using Gemini Pro Vision as VLM, initializing the model")
            self.gemini_pro_vision = GenerativeModel("gemini-1.0-pro-vision")

    def process_image(self, image, bbox=None):
        # we have to crop the image before converting it to base64
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            width, height = bbox[2], bbox[3]
            xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy")
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # increase the receptive field of each box to include possible nearby objects and contexts
            if width < self.min_bbox_size:
                x1 = int(max(0, x1 - (self.min_bbox_size - width) / 2))
                x2 = int(min(image.shape[1], x2 + (self.min_bbox_size - width) / 2))
            if height < self.min_bbox_size:
                y1 = int(max(0, y1 - (self.min_bbox_size - height) / 2))
                y2 = int(min(image.shape[0], y2 + (self.min_bbox_size - height) / 2))

            # cv2.imwrite('test_images/original_image' + str(bbox) + '.jpg', image)
            image = image[y1:y2, x1:x2]
            # cv2.imwrite('test_images/cropped_image' + str(bbox) + '.jpg', image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()
        # OpenAI models need base64 encoding, Gemini needs raw bytes
        if self.vlm_type != "gemini":
            image_bytes = base64.b64encode(image_bytes).decode('utf-8')

        return image_bytes

    def messages_to_answer_directly(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            message = "You are a Visual Question Answering (VQA) system. " \
                     "Look at the image carefully and answer the question accurately.\n\n" \
                     "ANSWER FORMAT:\n" \
                     "Always start your response with 'Answer: ' followed by your answer.\n\n" \
                     "REGULAR ANSWERS (99% of cases):\n" \
                     "- YES/NO questions: Answer 'yes' or 'no'\n" \
                     "- COUNTING questions: Give the number you count (e.g., '3', 'two')\n" \
                     "- OTHER questions: Give a short, direct answer\n\n" \
                     "SPECIAL CASES (use ONLY when specified):\n" \
                     "- ONLY for counting questions where you see ZERO objects: 'Answer: [Zero Numeric Answer]'\n" \
                     "- ONLY if you absolutely cannot answer: 'Answer: [Answer Failed]'\n\n" \
                     "Examples:\n" \
                     "Question: What is the man doing?\nAnswer: skiing\n" \
                     "Question: Is this a cat?\nAnswer: yes\n" \
                     "Question: How many cars are there?\nAnswer: 2\n" \
                     "Question: How many elephants do you see?\nAnswer: [Zero Numeric Answer] (only if you see zero elephants)"
        else:
            message = "You are a Visual Question Answering (VQA) system. " \
                     "Look at the image carefully and answer the question accurately.\n\n" \
                     "Always start with 'Answer: ' followed by a short, direct answer.\n" \
                     "Only use 'Answer: [Answer Failed]' if you absolutely cannot answer.\n\n" \
                     "Examples:\n" \
                     "Question: What is the man doing?\nAnswer: skiing\n" \
                     "Question: What color is the car?\nAnswer: red"
        return message

    def message_to_check_if_answer_is_numeric(self, question):
        message = f"Is this question asking for a count or number?\n\n" \
                 f"Question: '{question}'\n\n" \
                 f"Answer '[Numeric Answer]' if it asks 'how many' or 'what number of'.\n" \
                 f"Answer '[Not Numeric Answer]' if it asks anything else."
        return message

    def messages_to_query_object_attributes(self, question, phrase=None, verify_numeric_answer=False):
        if verify_numeric_answer:
            message = f"Look at the image and count the {phrase}.\n\n" \
                     f"Question: '{question}'\n\n" \
                     f"Describe what you see and count each {phrase} carefully. " \
                     f"Give a detailed description of each instance you can identify."
        else:
            message = f"Look at the image and describe the objects relevant to answering: '{question}'\n\n" \
                     f"Describe each relevant object including:\n" \
                     f"- Visual appearance (color, shape, size)\n" \
                     f"- Type and current state\n" \
                     f"- Location and relationships with other objects\n\n"

            if phrase is not None:
                message += f"Focus especially on: {phrase}\n"

        return message

    def messages_to_reattempt(self, question, obj_descriptions, prev_answer):
        message = f"The previous answer '{prev_answer}' was incorrect. Let me try again using this object information:\n\n" \
                 f"Question: '{question}'\n\n" \
                 f"Objects detected:\n"
        
        for i, obj in enumerate(obj_descriptions):
            message += f"- Object {i+1}: {obj}\n"

        if self.args['datasets']['dataset'] == 'vqa-v2':
            message += f"\nBased on these objects, answer the question directly:\n" \
                      f"- For YES/NO questions: Answer only 'yes' or 'no'\n" \
                      f"- For COUNTING questions: Give the exact number you can count\n" \
                      f"- For OTHER questions: Give a short, direct answer\n" \
                      f"- If still uncertain, write '[Answer Failed]'\n\n" \
                      f"Answer:"
        else:
            message += f"\nBased on these objects, give a direct answer to: '{question}'\n" \
                      f"If still uncertain, write '[Answer Failed]'\n\n" \
                      f"Answer:"

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
        responses = []

        if step == 'reattempt' or step == 'ask_directly' or bboxes is None or len(bboxes) == 0:
            if self.vlm_type=="gemini":
                response = self._query_gemini_pro_vision(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            else:
                response = self._query_openai_gpt_4v(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            return [response]

        # query on a single object
        if len(bboxes) == 1:
            bbox = bboxes.squeeze(0)
            phrase = phrases[0]
            if self.vlm_type == "gemini":
                response = self._query_gemini_pro_vision(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            else:
                response = self._query_openai_gpt_4v(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            responses.append(response)

        else:
            # process all objects from the same image in a parallel batch
            total_num_objects = len(bboxes)
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
                if self.vlm_type == "gemini":
                    response = list(executor.map(lambda bbox, phrase: self._query_gemini_pro_vision(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                else:
                    response = list(executor.map(lambda bbox, phrase: self._query_openai_gpt_4v(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                responses.append(response)

        return responses


    def _query_openai_gpt_4v(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        # we have to crop the image before converting it to base64
        base64_image = self.process_image(image, bbox)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly(question)
            max_tokens = 30  # Match the successful baseline
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question)
            else:
                messages = self.messages_to_query_object_attributes(question, phrase)
            max_tokens = 400
        elif step == 'reattempt':
            messages = self.messages_to_reattempt(question, obj_descriptions, prev_answer)
            max_tokens = 600
        else:
            raise ValueError('Invalid step')

        # Use provider-specific max_tokens if available
        if hasattr(self, 'max_tokens') and self.vlm_provider in ['vllm_8000', 'vllm_9000']:
            max_tokens = self.max_tokens

        # Retry if GPT response is like "I'm sorry, I cannot assist with this request"
        completion_text = ""  # Initialize completion_text
        for _ in range(3):
            # Form the prompt including the image.
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            # print('Prompt: ', messages)
            
            if self.vlm_provider == 'openai':
                # OpenAI API format
                prompt = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": messages},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        },
                        {"role": "assistant", "content": "Answer:"}
                    ],
                    "max_tokens": max_tokens
                }
            elif self.vlm_provider in ['vllm_8000', 'vllm_9000']:
                # vLLM API format (match 1_model.py exactly)
                prompt = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": messages},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": question}
                            ]
                        },
                        {"role": "assistant", "content": "Answer:"}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": self.temperature
                }
            else:
                # Default format (fallback)
                prompt = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": messages},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    "max_tokens": max_tokens
                }

            # Send request to API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            try:
                response = requests.post(self.base_url, headers=headers, json=prompt)
                
                if verbose:
                    print(f'API Response Status: {response.status_code}')
                    print(f'Using provider: {self.vlm_provider}, URL: {self.base_url}')
                    
                response_json = response.json()
                if verbose:
                    print(f'API Response JSON keys: {list(response_json.keys())}')
                    if 'error' in response_json:
                        print(f'API Error: {response_json["error"]}')
                        
                # Process the response
                # Check if the response is valid and contains the expected data
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    completion_text = response_json['choices'][0].get('message', {}).get('content', '')

                    if verbose:
                        print(f'VLM Response at step {step}: {completion_text}')
                        
                    # Process answer based on step and new format
                    if step == 'ask_directly':
                        # Simple processing like 1_model.py - just extract after "Answer:"
                        if "Answer:" in completion_text:
                            answer = completion_text.split("Answer:", 1)[-1].strip()
                            # Remove any trailing explanation or newlines  
                            answer = answer.split('\n')[0].strip()
                            if answer.lower().startswith("answer:"):
                                answer = answer.split(":", 1)[-1].strip()
                            completion_text = answer
                        else:
                            # Take first line as answer if no "Answer:" format
                            clean_answer = completion_text.strip().split('\n')[0].strip()
                            completion_text = clean_answer
                            
                    # Handle reattempt format
                    elif step == 'reattempt':
                        # Extract answer after "Answer:" 
                        if "Answer:" in completion_text:
                            answer_part = completion_text.split("Answer:", 1)[-1].strip()
                            # Take first line as the answer
                            answer = answer_part.split('\n')[0].strip()
                            if answer:
                                completion_text = answer
                        # Fallback: look for [Reattempted Answer] format (legacy)
                        elif "[Reattempted Answer]" in completion_text:
                            reattempt_match = re.search(r'\[Reattempted Answer\]\s*(.*?)(?:\n|$)', completion_text, re.DOTALL)
                            if reattempt_match:
                                answer = reattempt_match.group(1).strip()
                                completion_text = answer if answer else completion_text
                        # If no clear format, take first substantial line
                        else:
                            lines = completion_text.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith(('Based on', 'Looking at', 'The', 'I can see')):
                                    completion_text = line
                                    break
                else:
                    completion_text = ""
                    if verbose:
                        print(f'No choices in response or empty choices. Response: {response_json}')
                        
            except Exception as e:
                if verbose:
                    print(f'Request error: {e}')
                    print(f'Raw response: {response.text[:200] if "response" in locals() else "No response"}')
                # Don't automatically trigger [Answer Failed] on exception - retry instead
                completion_text = ""
                continue

            if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
                break
            
        # If all retries failed and completion_text is empty, only then consider it failed
        if not completion_text and step == 'ask_directly':
            completion_text = "[Answer Failed]"

        return completion_text


