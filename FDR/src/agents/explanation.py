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

    def generate_explanation_from_synthesis(self, question: str, synthesis_result: dict, caption: str, evidence_set: list) -> str:
        """
        NEW METHOD: Generate explanation using causal trace from Synthesizer Logic Engine.
        
        Args:
            question: Original question
            synthesis_result: Result from SynthesizerEngine.synthesize()
            caption: Image caption from verifier
            evidence_set: Evidence used in synthesis
        
        Returns:
            Natural language explanation
        """
        if not self.client:
            return "Error: No backend available for explanation generation"
        
        status = synthesis_result.get('status', 'UNKNOWN')
        answer = synthesis_result.get('answer')
        causal_trace = synthesis_result.get('causal_trace', [])
        
        try:
            if status == "CONCLUSIVE" and causal_trace:
                # Generate explanation from logical reasoning
                return self._generate_conclusive_explanation(question, answer, causal_trace, evidence_set, caption)
            
            elif status == "CONTRADICTORY":
                # Generate explanation for contradictory evidence
                contradictory_answers = synthesis_result.get('contradictory_answers', [])
                return self._generate_contradictory_explanation(question, contradictory_answers, causal_trace, caption)
            
            elif status == "INCONCLUSIVE":
                # Generate explanation for insufficient evidence
                return self._generate_inconclusive_explanation(question, evidence_set, caption)
            
            else:
                # Fallback for unknown status
                return self._generate_fallback_explanation(question, answer, caption)
                
        except Exception as e:
            logging.error(f"ExplanationAgent: Failed to generate explanation from synthesis: {e}")
            return f"Error generating explanation: {str(e)}"

    def _generate_conclusive_explanation(self, question: str, answer: str, causal_trace: list, evidence_set: list, caption: str) -> str:
        """Generate explanation for conclusive synthesis result"""
        
        # Extract logical reasoning from causal trace
        trace = causal_trace[0]  # First (primary) trace
        hypothesis_id = trace['hypothesis_id']
        triggered_evidence = trace['triggered_by_evidence']
        
        # Get details of triggered evidence
        evidence_details = []
        for evidence_id in triggered_evidence:
            evidence = next((e for e in evidence_set if e['evidence_id'] == evidence_id), None)
            if evidence:
                evidence_details.append({
                    'id': evidence_id,
                    'text': evidence['issue_text'],
                    'answer': evidence['answer'],
                    'source': evidence.get('source', 'Unknown')
                })
        
        # Build detailed prompt for explanation
        explanation_prompt = self._build_conclusive_explanation_prompt()
        
        user_input = f"""Question: {question}
Answer: {answer}
Image Context: {caption}

Logical Reasoning Chain:
- Hypothesis: {hypothesis_id}
- Triggered by evidence: {triggered_evidence}

Evidence Details:
{self._format_evidence_details(evidence_details)}

Generate a natural explanation of why this answer was chosen based on the logical reasoning."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": explanation_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2,  # Lower temperature for more consistent explanations
            max_tokens=200
        )
        
        explanation = response.choices[0].message.content.strip()
        
        # Clean up explanation format
        if explanation.startswith("Explanation:"):
            explanation = explanation.replace("Explanation:", "").strip()
        
        # Ensure explanation starts with "The answer is..."
        if not explanation.lower().startswith("the answer is"):
            explanation = f"The answer is \"{answer}\" because {explanation.lower()}"
        
        logging.info(f"ExplanationAgent: Generated conclusive explanation for '{answer}'")
        return explanation

    def _generate_contradictory_explanation(self, question: str, contradictory_answers: list, causal_trace: list, caption: str) -> str:
        """Generate explanation for contradictory synthesis result"""
        
        explanation_prompt = self._build_contradictory_explanation_prompt()
        
        user_input = f"""Question: {question}
Image Context: {caption}

Contradictory Analysis Result:
- Multiple possible answers found: {', '.join(contradictory_answers)}
- Number of conflicting logical rules: {len(causal_trace)}

Generate an explanation for why the analysis reached contradictory conclusions."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": explanation_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        explanation = response.choices[0].message.content.strip()
        logging.info(f"ExplanationAgent: Generated contradictory explanation")
        return explanation

    def _generate_inconclusive_explanation(self, question: str, evidence_set: list, caption: str) -> str:
        """Generate explanation for inconclusive synthesis result"""
        
        explanation_prompt = self._build_inconclusive_explanation_prompt()
        
        user_input = f"""Question: {question}
Image Context: {caption}

Analysis Result:
- Evidence available: {len(evidence_set)} items
- No logical rules were satisfied
- Insufficient evidence for definitive conclusion

Generate an explanation for why no definitive answer could be determined."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": explanation_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=120
        )
        
        explanation = response.choices[0].message.content.strip()
        logging.info(f"ExplanationAgent: Generated inconclusive explanation")
        return explanation

    def _generate_fallback_explanation(self, question: str, answer: str, caption: str) -> str:
        """Generate fallback explanation when synthesis status is unknown"""
        return f"The answer \"{answer}\" was determined based on visual analysis of the image. {caption}"

    def _format_evidence_details(self, evidence_details: list) -> str:
        """Format evidence details for prompt"""
        formatted = []
        for evidence in evidence_details:
            formatted.append(f"- {evidence['id']}: {evidence['text']} → {evidence['answer']} (from {evidence['source']})")
        return "\n".join(formatted)

    def _build_conclusive_explanation_prompt(self) -> str:
        """Build system prompt for conclusive explanations"""
        return """You are an expert at explaining logical reasoning in clear, natural language. Your task is to create a single, concise explanation for why a specific answer was chosen based on logical analysis.

You will be given:
1. The original question
2. The chosen answer  
3. Image context
4. The logical reasoning chain (hypothesis + evidence)

Your explanation must:
- Start with "The answer is [answer] because..."
- Explain the logical reasoning in natural language
- Reference the visual evidence when relevant
- Be clear and easy to understand
- Be exactly one sentence

Focus on explaining WHY the logical reasoning led to this specific answer, making the causal connection clear."""

    def _build_contradictory_explanation_prompt(self) -> str:
        """Build system prompt for contradictory explanations"""
        return """You are an expert at explaining when logical analysis finds conflicting evidence. Your task is to explain why multiple contradictory answers were found.

Your explanation must:
- Acknowledge the contradictory findings
- Explain what this means in practical terms
- Suggest what additional information might help
- Be clear and professional
- Be 1-2 sentences maximum

Focus on explaining why the analysis couldn't reach a single conclusion."""

    def _build_inconclusive_explanation_prompt(self) -> str:
        """Build system prompt for inconclusive explanations"""
        return """You are an expert at explaining when logical analysis cannot reach a definitive conclusion. Your task is to explain why no clear answer could be determined.

Your explanation must:
- Explain why the evidence was insufficient
- Be honest about the limitations
- Suggest what might help reach a conclusion
- Be clear and professional
- Be 1-2 sentences maximum

Focus on explaining why no logical rules were satisfied by the available evidence."""

    def generate_explanation(self, question: str, final_answer: str, caption: str, mvkb_entries: list, confidence_breakdown: dict) -> str:
        """
        LEGACY METHOD: Generate explanation from MVKB entries and voting results.
        Maintained for backward compatibility.
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
            
            logging.info(f"ExplanationAgent: Generated legacy explanation for '{final_answer}'")
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