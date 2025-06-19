# Top-Down/core/agents.py
import base64
import logging
from openai import OpenAI
from retrying import retry
import json

# --- Helper Functions ---

def encode_image_to_base64(image_path: str) -> str | None:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found at {image_path}")
        return None

# --- Agent Definitions ---

class ResponderAgent:
    """
    The Responder Agent, based on a VLM.
    Its main goal is to generate initial answer candidates and image captions.
    """
    def __init__(self, client: OpenAI, model_name: str, temperature: float, max_tokens: int):
        self.client = client
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def generate_initial_response(self, question: str, image_path: str) -> dict:
        """
        Generates top-K answer candidates and a caption for the image.
        """
        logging.debug(f"Responder: Generating initial response for image {image_path}")
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return {
                "answer_candidates": ["Error: Image not found."],
                "caption": "Error: Could not read the image file.",
            }

        prompt = f"""
You are a Visual Question Answering assistant. Your task is to analyze the provided image and question.
1.  First, provide a detailed, one-paragraph caption for the image.
2.  Second, based on the image and question, list the 3 most likely answers.

Question: "{question}"

Please format your response as a JSON object with two keys: "caption" and "answer_candidates" (which should be a list of 3 strings).
Example:
{{
  "caption": "A detailed description of the image content.",
  "answer_candidates": ["answer 1", "answer 2", "answer 3"]
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        
        try:
            content = response.choices[0].message.content
            # The response is a JSON string, so we need to load it
            data = json.loads(content)
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Responder: Failed to parse JSON response. Error: {e}. Response: {content}")
            return {
                "answer_candidates": ["Error: Invalid model response."],
                "caption": "Error: Could not parse the model's response.",
            }

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def answer_contextual_question(self, question: str, image_path: str, context_prompt: str) -> str:
        """
        Answers a question given an image and a rich context prompt (used by the Integrator).
        """
        logging.debug(f"Responder: Answering contextual question for image {image_path}")
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Error: Image not found."

        # The context_prompt is expected to contain the hypothesis and the original question.
        full_prompt = f"{context_prompt}\n\nBased on the image and the hypothesis, what is the final answer to the original question?"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                 {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0, # Low temperature for factual, direct answers
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

class SeekerAgent:
    """
    The Seeker Agent, based on an LLM.
    Its main goal is to generate a Multi-View Knowledge Base (MVKB)
    by creating relevant issues, forming hypotheses, and assigning confidence.
    """
    def __init__(self, client: OpenAI, model_name: str, responder: ResponderAgent):
        self.client = client
        self.model = model_name
        self.responder = responder

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _create_relevant_issues(self, question: str, answer_candidates: list, caption: str) -> list:
        """Generates clarifying questions (relevant issues) to differentiate answers."""
        prompt = f"""
You are a reasoning expert for a Visual Question Answering system.
Your goal is to generate a few clarifying questions (called "relevant issues") that, when answered, will help determine which of the candidate answers is correct.

Original Question: "{question}"
Image Caption: "{caption}"
Candidate Answers: {answer_candidates}

Generate a JSON list of 2-3 relevant issue questions. The questions should be specific and designed to be answered by looking at the image.
Example: If the answers are "sunny" or "cloudy", a good relevant issue is "What does the sky look like in the image?".

Output ONLY the JSON list of strings.
Example:
["Is there a clock visible in the image?", "What is the person's facial expression?"]
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            # The model should return a json object with a key that contains the list
            data = json.loads(response.choices[0].message.content)
            # Find the list in the returned dict
            for key, value in data.items():
                if isinstance(value, list):
                    return value
            return [] # Return empty if no list found
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Seeker: Failed to parse relevant issues: {e}")
            return []

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _formulate_hypotheses_and_confidence(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> dict:
        """Forms a logical hypothesis and assigns a confidence score."""
        prompt = f"""
You are a logical reasoning module. Your task is to create a single logical hypothesis and assign a confidence score.

Context:
- Original Question: "{question}"
- A possible answer to the Original Question: "{answer_candidate}"
- A related sub-question (Relevant Issue): "{relevant_issue}"
- The answer to the Relevant Issue (based on the image): "{issue_answer}"

Task:
1.  Formulate a single, clear "IF-THEN" hypothesis that connects the sub-question's answer to the main answer.
2.  Based on common-sense reasoning, assign a confidence score from 0.0 (not confident) to 1.0 (very confident) that this hypothesis is logically sound.
3.  Convert the confidence score to a confidence word: <0.4 is "Unlikely", 0.4-0.7 is "Possible", >0.7 is "Likely".

Provide your response as a single JSON object with three keys: "hypothesis", "confidence_score" (a float), and "confidence_word" (a string).

Example:
{{
  "hypothesis": "IF the sky in the image is full of dark clouds, THEN the weather is likely rainy.",
  "confidence_score": 0.9,
  "confidence_word": "Likely"
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Seeker: Failed to parse hypothesis response: {e}")
            return {}

    def build_mvkv(self, question: str, image_path: str, answer_candidates: list, caption: str) -> list:
        """
        Builds the complete Multi-View Knowledge Base by orchestrating the Seeker's logic.
        """
        logging.info(f"Seeker: Building MVKB for question '{question}'")
        mvkv = []

        relevant_issues = self._create_relevant_issues(question, answer_candidates, caption)
        logging.debug(f"Seeker: Generated relevant issues: {relevant_issues}")

        for issue in relevant_issues:
            # Use the responder to get the answer for the sub-question
            issue_response = self.responder.generate_initial_response(issue, image_path)
            issue_answer = issue_response['answer_candidates'][0] # Take the top answer for the issue
            logging.debug(f"Seeker: Answer for issue '{issue}' is '{issue_answer}'")
            
            for candidate in answer_candidates:
                hypothesis_data = self._formulate_hypotheses_and_confidence(question, candidate, issue, issue_answer)
                if hypothesis_data:
                    mvkv_entry = {
                        "original_question": question,
                        "answer_candidate": candidate,
                        "relevant_issue": issue,
                        "issue_answer": issue_answer,
                        **hypothesis_data  # Unpack hypothesis, score, and word
                    }
                    mvkv.append(mvkv_entry)
        
        logging.info(f"Seeker: MVKB built with {len(mvkv)} entries.")
        return mvkv 

class IntegratorAgent:
    """
    The Integrator Agent, a logical mechanism.
    Its main goal is to conduct a weighted vote over the initial answer
    candidates based on the evidence from the MVKB.
    """
    def __init__(self, responder: ResponderAgent):
        self.responder = responder

    def conduct_weighted_voting(self, original_question: str, image_path: str, answer_candidates: list, mvkv: list) -> str:
        """
        Conducts the final weighted voting to determine the best answer.
        """
        logging.info("Integrator: Conducting weighted voting.")
        
        if not mvkv:
            logging.warning("Integrator: MVKB is empty. Returning the first answer candidate as a fallback.")
            return answer_candidates[0] if answer_candidates else "No answer could be determined."

        vote_scores = {candidate: 0.0 for candidate in answer_candidates}
        
        # This is the re-evaluation step from the paper
        for entry in mvkv:
            confidence_score = entry.get("confidence_score", 0.0)
            
            # Create the context-rich prompt for the Responder
            context_prompt = f"""
Hypothesis (Confidence: {entry.get('confidence_word', 'N/A')}): {entry.get('hypothesis', 'No hypothesis.')}
"""
            # Ask the responder to answer the original question again, but with this new context
            final_vote = self.responder.answer_contextual_question(
                question=original_question,
                image_path=image_path,
                context_prompt=context_prompt
            )
            logging.debug(f"Integrator: Vote received for '{final_vote}' with score {confidence_score}")

            # Add the confidence score to the answer that was voted for.
            # We check which of the original candidates the vote is closest to.
            for candidate in answer_candidates:
                if candidate.lower() in final_vote.lower():
                    vote_scores[candidate] += confidence_score
                    break # Stop after finding the first match

        logging.debug(f"Integrator: Final vote scores: {vote_scores}")

        # Find the candidate with the highest total score
        if not any(vote_scores.values()):
             logging.warning("Integrator: No votes were cast. Returning first candidate.")
             return answer_candidates[0]

        final_answer = max(vote_scores, key=vote_scores.get)
        logging.info(f"Integrator: Final answer chosen is '{final_answer}'")
        
        return final_answer 