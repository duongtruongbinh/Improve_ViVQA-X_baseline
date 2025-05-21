# Derived from QueryVLM.messages_to_answer_directly (assuming non-Gemini general version)
# The {question} will be part of the user message to the agent.
INITIAL_VLM_SYSTEM_PROMPT_VQA_V2 = """
You are performing a Visual Question Answering task.
You will be given an image and a question.
First, explain what the question wants to ask, what objects or objects with specific attributes are related to the question, 
you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. 
Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image. 
Finally, begin your final answer with the notation '[Answer]'. 
The correct answer could be a 'yes/no', a number, or other open-ended response. 
(Case 1) If you believe your answer falls into the category of 'yes/no', say 'yes/no' after '[Answer]'. 
Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'.
For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. 
(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', 
pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible.
Begin your answer with '[Numeric Answer]', step-by-step describe each object in this image that satisfy the descriptions in the question, 
list each one by [Object i] where i is the index, and finally predict the number. 
If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. 
If there are many of them, for example, more than three in the image, you should answer '[Non-zero Numeric Answer]' and '[Answer Failed]'. and avoid being too confident. 
(Case 3) If the answer should be an activity or a noun, say the word after '[Answer]'. Similarly, no extra words after '[Answer]'. 
(Case 4) If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, 
do not make a guess, but please explain why and what you need to solve the question,
like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'. Keep your answers short.
"""

INITIAL_VLM_SYSTEM_PROMPT_DEFAULT = """
You are performing a Visual Question Answering task.
You will be given an image and a question.
Please first explain what the question wants to ask, what objects or objects with specific attributes
you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. 
Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image.
Begin your final answer with the notation '[Answer]'. 
If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, 
do not make a guess, but please explain why and what you need to solve the question,
like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'.
"""

# Derived from QueryLLM.messages_to_extract_needed_objects (non-numeric verification part)
# The {question} and {previous_response} will be part of the user message.
FAILURE_ANALYSIS_SYSTEM_PROMPT = """
Based on the response provided by a large vision-language model (VLM) for a visual question answering task, 
it appears that the model encountered difficulties in generating an accurate answer for the given question. 
The model has provided an explanation (previous response) for its inability to respond correctly, which might suggest that certain objects 
important to answer the question were not detected or considered in the image. 
Your task is to analyze the model's explanation carefully in conjunction with the original question.
Identify critical objects, attributes, or relationships that were likely missed or misinterpreted.
For questions asking about specific objects (e.g., 'What is the color of the car?'), list the objects 'Car' directly if it seems missed.
For questions seeking objects with certain attributes (e.g., 'Which object has a bright color?'), list the attributes with the word 'objects' (e.g., 'bright-colored objects').
Make sure to include the subject and the object of the question if they seem critical and potentially overlooked.
Ignore objects irrelevant to the question even if they are mentioned in the model explanation.
If the VLM's failure seems to be related to counting, explicitly state 'Numeric reattempt needed for: [item to count]'.
Otherwise, list the needed objects/attributes for a general reattempt as a single line: 'General reattempt, focus on: Object1 . Object2 . Attribute3 objects .'
If no specific objects or attributes can be identified from the explanation but the answer failed, suggest focusing on objects mentioned in the original question.
"""
# The user message to FailureAnalysisAgent will be:
# f"Original Question: '{question}'\nFailed VLM Response: '{previous_response}'"


# Derived from QueryVLM.messages_to_query_object_attributes (non-numeric, no phrase from tool)
# The {question} and {textually_specified_objects} will be part of the user message.
# This prompt is for getting descriptions of objects *without* bounding boxes, based on text.
OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS = """
You will be given an image, an original VQA question, and a list of objects/attributes (derived cứu textually) that are considered important for answering the question.
Your task is to carefully examine the image and provide a concise, relevant description for each of the specified objects/attributes as they appear in the image.
Focus on visual attributes like color, shape, size, materials, relative location, and current status if applicable, especially those details that might help answer the original VQA question.
For example, if asked to describe 'the red car on the left', your description should confirm its presence, color, and location, and any other notable feature.
Structure your response clearly, addressing each specified item.
"""
# The user message to ObjectAttributeAgent will be:
# f"Original Question: '{question}'\nFocus on these textually identified items from the image: '{textually_specified_objects}'. Provide their descriptions."

# Derived from QueryVLM.messages_to_reattempt (adapted for no tools, using textual descriptions)
# The {question}, {previous_failed_answer}, and {new_object_descriptions} will be part of the user message.
REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS = """
You are performing a reattempt at a Visual Question Answering task.
You will receive the original image, the original question, the previous failed answer, and new textual descriptions of relevant objects/attributes identified in the image.
Your task is to synthesize all this information to provide a new, more accurate answer to the original question.
First, explain your reasoning step-by-step, considering the new descriptions and how they address the shortcomings of the previous answer.
List any geometric, possessive, or semantic relations among the described objects/attributes that are crucial for answering the question.
Then, begin your final answer with '[Reattempted Answer]'.
The correct answer could be a 'yes/no', a number, or other open-ended response. 
(Case 1) If you believe your answer falls into the category of 'yes/no' or a number, say 'yes/no' or the number after '[Reattempted Answer]'. 
Understand that the question may not capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'.
(Case 2) If the question asks you to count the number of an object (e.g. 'how many'), and new descriptions provide clarity,
step-by-step describe each object that satisfies the question based on new context, and then re-evaluate the number after '[Reattempted Answer]'. Objects could be only partially visible.
Your previous attempt might have struggled with counting. Please be meticulous. List each instance you count before giving the total. For example: "I see: 1. a cat on the mat. 2. a cat on the chair. [Reattempted Answer] 2".
(Case 3) If the answer should be an activity or a noun, say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'.
If, even with the new information, you still cannot confidently answer, explain why and use '[Answer Failed]'.
"""

REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS = """
You are performing a reattempt at a Visual Question Answering task.
You will receive the original image, the original question, the previous failed answer, and new textual descriptions of relevant objects/attributes identified in the image.
Your task is to synthesize all this information to provide a new, more accurate answer to the original question.
Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects that are crucial for answering the question.
Given these additional object descriptions, please re-attempt to answer the visual question step by step.
Begin your final answer with '[Reattempted Answer]'.
If, even with the new information, you still cannot confidently answer, explain why and use '[Answer Failed]'.
"""


# Derived from QueryLLM.messages_to_grade_the_answer
# The {grader_id}, {question}, {target_answer}, {model_answer} will be part of the user message or formatted into it.
GRADING_SYSTEM_PROMPT_TEMPLATE = """
Please grade the following answer provided by a large vision-language model (VLM) for a visual question answering task in one to two sentences.
The VLM was asked a question, given an image. We have a target (correct) answer and the VLM's provided answer.
Please understand that the target answer provided by the dataset might be artificially short. 
Therefore, as long as the target answer is mentioned in or consistent with the VLM's answer, it should be graded as '[Grader {grader_id}] [Correct]'. 
If the VLM's answer contains the target answer but has additional information not mentioned by the target answer, it is still '[Correct]'.
If the question involves multiple conditions and the target answer is 'no', grade the VLM's answer as '[Grader {grader_id}] [Correct]' as long as it correctly finds that one ofต้นฉบับthe conditions is not met.
If the answer is a number, verify if the number is correct.
Partially correct answers or synonyms are still '[Grader {grader_id}] [Correct]'. For example, 'brown' and 'black' might be considered synonyms in some contexts if the distinction is not critical for the question.
Otherwise, if the VLM's answer misses the targeted information or is clearly wrong, grade the answer as '[Grader {grader_id}] [Incorrect]'.
Focus on the part after '[Answer]' or '[Reattempted Answer]' in the model's response.
Reason your grading step by step but keep it short. Your entire response should start with '[Grader {grader_id}] [Correct]' or '[Grader {grader_id}] [Incorrect]'.
"""
# User message to GraderAgent will be:
# f"Grader ID: {grader_id}\nQuestion: '{question}'\nTarget Answer: '{target_answer}'\nModel's Answer: '{model_answer}'"