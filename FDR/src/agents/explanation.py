"""
ExplanationAgent for MVKB-X Pipeline
Explanation Generation Agent for MVKB-X pipeline.
Generates natural language explanations from MVKB and final answers.
"""

import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI

from .base import BaseAgent


class ExplanationAgent(BaseAgent):
    """
    Explanation Generation Agent for MVKB-X pipeline.
    Generates natural language explanations from MVKB and final answers.
    """
    
    def __init__(self, client: OpenAI = None, use_vllm: bool = True):
        super().__init__(use_vllm)
        
        # Initialize backend
        self._initialize_backend()

    def generate_explanation(self, question: str, final_answer: str, caption: str, mvkb_entries: list, confidence_breakdown: dict) -> str:
        """
        Generate explanation from MVKB entries and voting results.
        Adapted from top_down_baseline Step 6 logic.
        """
        if not self.client:
            return "Error: No backend available for explanation generation"
        
        # Find key hypothesis with highest confidence for final answer
        key_hypothesis = self._find_key_hypothesis(mvkb_entries, final_answer)
        
        # Build explanation prompt
        explanation_prompt = self._build_explanation_prompt()
        
        # Generate explanation
        user_input = f"Question: {question}\n"
        user_input += f"Image Caption: {caption}\n"
        
        if key_hypothesis['hypothesis']:
            confidence_level = self._map_confidence_level(key_hypothesis['confidence_score'])
            user_input += f"Key Hypothesis: {key_hypothesis['hypothesis']} ({confidence_level})\n"
        else:
            user_input += f"Key Hypothesis: No specific hypothesis available for this answer.\n"
        
        user_input += f"Final Answer: {final_answer}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": explanation_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                max_tokens=256
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Clean up explanation format
            if explanation.startswith("Explanation:"):
                explanation = explanation.replace("Explanation:", "").strip()
            
            logging.info(f"ExplanationAgent: Generated explanation for '{final_answer}'")
            return explanation
            
        except Exception as e:
            logging.error(f"ExplanationAgent: Failed to generate explanation: {e}")
            return f"Error generating explanation: {str(e)}"

    def _find_key_hypothesis(self, mvkb_entries: list, final_answer: str) -> dict:
        """Find hypothesis with highest confidence for the final answer"""
        best_hypothesis = {"hypothesis": None, "confidence_score": 0.0}
        
        for entry in mvkb_entries:
            if entry.get("answer_candidate") == final_answer:
                confidence = entry.get("confidence_score", 0.0)
                if confidence > best_hypothesis["confidence_score"]:
                    best_hypothesis = {
                        "hypothesis": entry.get("hypothesis"),
                        "confidence_score": confidence
                    }
        
        return best_hypothesis

    def _map_confidence_level(self, confidence_score: float) -> str:
        """Map confidence score to confidence word"""
        if confidence_score > 0.8:
            return "Very Likely"
        elif confidence_score > 0.6:
            return "Likely"
        elif confidence_score > 0.4:
            return "Possible"
        else:
            return "Unlikely"

    def _build_explanation_prompt(self) -> str:
        """Build system prompt for explanation generation"""
        return """You are a reasoning and explanation generation expert. Your task is to craft a single, concise sentence that synthesizes all provided information to explain WHY a specific answer was chosen for a visual question.

You will be given:
1. **Question:** The original question.
2. **Image Caption:** The visual context.
3. **Key Hypothesis:** The core logical rule that was applied.
4. **Final Answer:** The chosen answer.

Your explanation must be a single, fluid sentence that directly answers "Why was this answer chosen?". It should seamlessly weave together the visual evidence from the caption and the logic from the hypothesis.

**Output Format:**
Provide only the explanation sentence, starting with "The answer is..."

Example: The answer is "Fry" because the dish's glossy appearance, as seen in the image, directly matches the very likely hypothesis that a glossy sheen indicates a frying preparation method.""" 