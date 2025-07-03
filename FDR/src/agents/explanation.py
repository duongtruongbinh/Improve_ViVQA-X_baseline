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
        
        confidence_level = self._map_confidence_level(key_hypothesis['confidence_score'])
        hypothesis = f"{key_hypothesis['hypothesis']} ({confidence_level})"
        # Build explanation prompt
        explanation_prompt = self._build_explanation_prompt(question, caption, hypothesis, final_answer)
        
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
                    {"role": "user", "content": explanation_prompt}
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

    def _build_explanation_prompt(self, question: str, caption: str, key_hypothesis: str, final_answer: str) -> str:
        """Build system prompt for explanation generation"""
        return f"""You are an expert at explaining visual question answering results. Your task is to synthesize the following information into a natural, conversational explanation.

--- PROVIDED INFORMATION ---

ORIGINAL QUESTION: {question}

IMAGE CONTEXT: {caption}

KEY REASONING: {key_hypothesis}

FINAL ANSWER: {final_answer}

--- YOUR TASK ---
- Create an explanation that is natural and conversational.
- Remove any robotic or template-like language from the Key Reasoning.
- Keep the explanation concise (1-2 sentences maximum).
- Maintain the core logic from the Key Reasoning but make it sound human.
- Use appropriate confidence language based on the reasoning (e.g., "(Likely)" might become "suggests that...", while "(Very Likely)" might become "clearly shows...").

Natural explanation:"""  