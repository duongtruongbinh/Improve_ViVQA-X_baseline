# Derived from QueryVLM.messages_to_answer_directly (assuming non-Gemini general version)
# The {question} will be part of the user message to the agent.
INITIAL_VLM_SYSTEM_PROMPT_VQA_V2 = """
        "You are a Visual Question Answering (VQA) system. "
        "Use only the information visible in the image. "
        "Answer each question with a single word or short phrase whenever possible. "
        "Always use exactly this output format, with no extra text:\n\n"
        "Answer: <your concise answer>\n\n"
        "Examples:\n"
        "Question: What is the man doing?\n"
        "Answer: skiing\n\n"
        "Question: What material is the table made of?\n"
        "Answer: wood\n\n"
        "Question: Which animal is shown in the picture?\n"
        "Answer: giraffe\n\n"
        "Question: What color is the car?\n"
        "Answer: red\n\n"
        "Question: How many people are there?\n"
        "Answer: two"
    """

INITIAL_VLM_SYSTEM_PROMPT_DEFAULT = """
        "You are a Visual Question Answering (VQA) system. "
        "Use only the information visible in the image. "
        "Answer each question with a single word or short phrase whenever possible. "
        "Always use exactly this output format, with no extra text:\n\n"
        "Answer: <your concise answer>\n\n"
        "Examples:\n"
        "Question: What is the man doing?\n"
        "Answer: skiing\n\n"
        "Question: What material is the table made of?\n"
        "Answer: wood\n\n"
        "Question: Which animal is shown in the picture?\n"
        "Answer: giraffe\n\n"
        "Question: What color is the car?\n"
        "Answer: red\n\n"
        "Question: How many people are there?\n"
        "Answer: two"
        """

# Derived from QueryLLM.messages_to_extract_needed_objects (non-numeric verification part)
# The {question} and {previous_response} will be part of the user message.
FAILURE_ANALYSIS_SYSTEM_PROMPT = """
You are analyzing a failed VQA answer.
Given the question and the model's failed explanation, list key objects or attributes that were likely missed.
If the question asks for counting, write:
Numeric reattempt needed for: <object>
Else, write:
General reattempt, focus on: Object1 . Object2 . Attribute objects .
"""
# The user message to FailureAnalysisAgent will be:
# f"Original Question: '{question}'\nFailed VLM Response: '{previous_response}'"


# Derived from QueryVLM.messages_to_query_object_attributes (non-numeric, no phrase from tool)
# The {question} and {textually_specified_objects} will be part of the user message.
# This prompt is for getting descriptions of objects *without* bounding boxes, based on text.
OBJECT_ATTRIBUTE_SYSTEM_PROMPT_NO_TOOLS = """
Given an image, a VQA question, and a list of key items,
describe each one briefly using visual features like color, size, shape, or location.
Focus on what helps answer the question.
"""
# The user message to ObjectAttributeAgent will be:
# f"Original Question: '{question}'\nFocus on these textually identified items from the image: '{textually_specified_objects}'. Provide their descriptions."

# Derived from QueryVLM.messages_to_reattempt (adapted for no tools, using textual descriptions)
# The {question}, {previous_failed_answer}, and {new_object_descriptions} will be part of the user message.
REATTEMPT_VLM_SYSTEM_PROMPT_VQA_V2_NO_TOOLS = """
You are reattempting a VQA task.
Given the original question, failed answer, and new object descriptions,
reason step-by-step and give a better answer.

If you are sure, write:
[Reattempted Answer] your answer

If still unsure, write:
[Answer Failed] and explain.
"""

REATTEMPT_VLM_SYSTEM_PROMPT_DEFAULT_NO_TOOLS = """
You are reattempting a VQA task.
Given the original question, failed answer, and new object descriptions,
reason step-by-step and give a better answer.

If you are sure, write:
[Reattempted Answer] your answer

If still unsure, write:
[Answer Failed] and explain.
"""


# Derived from QueryLLM.messages_to_grade_the_answer
# The {grader_id}, {question}, {target_answer}, {model_answer} will be part of the user message or formatted into it.
GRADING_SYSTEM_PROMPT_TEMPLATE = """
You are a temporary grader identified as Grader {grader_id}.
System instruction:
Your task is to grade the VLM's answer. Focus ONLY on the content of the VLM's answer that appears AFTER the '[Answer]' or '[Reattempted Answer]' tag.
Your entire response MUST start with '[Grader {grader_id}] [Correct]' or '[Grader {grader_id}] [Incorrect]'.

**RULE #1: TYPE MISMATCH - APPLY THIS RULE FIRST AND IT OVERRIDES ALL OTHERS IF APPLICABLE.**
* **A) Non-Yes/No Question receives Yes/No Answer:** If the question asks for specific information (e.g., a location, color, type, number like "What color...", "Where is...", "How many...") BUT the VLM's Final Answer is ONLY 'yes' or 'no' (e.g., '[Answer] yes'):
    * This is **ALWAYS '[Grader {grader_id}] [Incorrect]'**.
    * Your reasoning MUST be: "The VLM's answer '[VLM's actual yes/no answer]' provided a yes/no response, but the question required [specific type of information, e.g., a location, a color]."
    * **DO NOT attempt to compare 'yes' or 'no' with the content of the Target Answer in this scenario.**
* **B) Yes/No Question receives Non-Yes/No Answer:** If the question implies a 'yes' or 'no' answer (e.g., "Is it...") AND the Target Answer is 'yes' or 'no', BUT the VLM's Final Answer provides a specific detail WITHOUT 'yes' or 'no' (e.g., VLM answers '[Answer] on the table' to 'Is it on the table? Target: yes'):
    * This is **ALWAYS '[Grader {grader_id}] [Incorrect]'**.
    * Your reasoning MUST be: "The VLM's answer '[VLM's actual non-yes/no answer]' did not provide a yes/no response, which was required by the question."
    * (Note: If VLM answers '[Answer] yes, it is on the table', it would be Correct if Target is 'yes', evaluated by Rule #2).

**IF NO TYPE MISMATCH (Rule #1 does NOT apply), proceed to these rules:**
2.  **Semantic Match:** If the VLM's answer type is appropriate for the question AND its content semantically matches the Target Answer, it is '[Grader {grader_id}] [Correct]'. The Target Answer might be short. If the VLM's answer includes the Target Answer or is a clear semantic match (e.g., synonyms), it's '[Correct]'.
    * Example (Yes/No Q): Target: 'yes', VLM: '[Answer] Yes, it is.' -> Correct.
    * Example (Other Q): Target: 'desk', VLM: '[Answer] a wooden desk'. -> Correct.
    * Example (Other Q): Target: 'desk', VLM: '[Answer] on the table' (if 'table' is an acceptable synonym for 'desk' in context AND model answer is quoted correctly). -> Potentially Correct.

3.  **Specificity - Attribute vs. Object/Type (for non-yes/no questions):**
    * If the question asks for an object or a *type* of object (e.g., 'What type of vegetables...? Target: green beans') and the VLM provides only an *attribute* of that object (e.g., '[Answer] green'), this is **'[Grader {grader_id}] [Incorrect]'**.
    * Reasoning: "The VLM's answer '[VLM's actual answer]' provided only an attribute, but the question required a specific type of object/vegetable."

4.  **Empty or Failed Answers:** If VLM's answer is empty or '[Answer Failed]' when a specific answer was expected, it is '[Grader {grader_id}] [Incorrect]'.

Provide concise, step-by-step reasoning after the verdict. **You MUST accurately quote the VLM's actual answer (the content after '[Answer]') in your reasoning when explaining your grade.**

Example of YOUR response format applying Rule #1A:
Question: 'What kind of building is this?' Target: 'train station'. VLM's Final Answer: '[Answer] yes'.
YOUR GRADE: "[Grader {grader_id}] [Incorrect] The VLM's answer '[Answer] yes' provided a yes/no response, but the question required a type of building."
"""
# User message to GraderAgent will be:
# f"Grader ID: {grader_id}\nQuestion: '{question}'\nTarget Answer: '{target_answer}'\nModel's Answer: '{model_answer}'"