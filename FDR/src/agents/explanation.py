"""
ExplanationAgent for the FDR Pipeline.
This agent is solely responsible for generating the final natural language
explanation based on the results from the Synthesizer.
"""

import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI

from .base import BaseAgent

class ExplanationAgent(BaseAgent):
    """
    Generates natural, faithful explanations for the VQA results.
    It uses an LLM to transform the logical trace from the Synthesizer
    into a human-readable explanation.
    """

    def __init__(self, agent_config: Dict[str, Any], client: Optional[OpenAI], model_name: Optional[str]):
        """Initializes the ExplanationAgent."""
        super().__init__(agent_config, client, model_name)
        self.logger = logging.getLogger(__name__)

    def generate_explanation(self, question: str, answer: str, mvkb: dict, voting_result: dict) -> str:
        """
        Generate a natural language explanation from the synthesis results.
        
        Args:
            question: The original question.
            answer: The final answer determined by the synthesizer.
            mvkb: The Multi-View Knowledge Base from the strategist.
            voting_result: The result dictionary from the synthesizer.

        Returns:
            A natural language explanation string.
        """
        if not voting_result:
            return "Unable to generate an explanation due to a synthesizer failure."

        try:
            # Step 1: Build context from the synthesizer's results
            context = self._build_narrative_context(voting_result, mvkb)
            
            # Step 2: Create a logical template from the context
            template = self._create_narrative_template(question, context)
            
            # Step 3: Use the LLM to refine the template into a natural explanation
            explanation = self._refine_with_llm(template, question, answer)
            
            self.logger.debug(f"ExplanationAgent generated: {explanation}")
            return explanation

        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            # Fallback to a simple explanation if generation fails
            status = voting_result.get('status', 'UNKNOWN')
            if status == 'CONCLUSIVE':
                return f"The answer is {answer} based on logical analysis of the image."
            else:
                return f"The answer is {answer} based on visual analysis of the image."

    def _build_narrative_context(self, voting_result: dict, mvkb: dict) -> dict:
        """Builds a structured context for explanation from the voting and MVKB."""
        status = voting_result.get('status', 'UNKNOWN')
        answer = voting_result.get('answer')
        causal_trace = voting_result.get('causal_trace', [])
        evidence_set = mvkb.get('evidence_set', [])
        
        context = {
            'answer': answer,
            'status': status,
            'reasoning_description': None,
            'evidence_descriptions': [],
            'confidence_level': 'moderate'
        }

        if causal_trace:
            winning_hypothesis = causal_trace[0]
            context['reasoning_description'] = winning_hypothesis.get('reasoning_description', '')
            
            triggered_evidence_ids = winning_hypothesis.get('triggered_by_evidence', [])
            evidence_map = {e.get('evidence_id'): e for e in evidence_set}
            
            for evidence_id in triggered_evidence_ids:
                if evidence_id in evidence_map:
                    evidence = evidence_map[evidence_id]
                    description = evidence.get('issue_description', evidence.get('issue_text', ''))
                    context['evidence_descriptions'].append(description)

            confidence = winning_hypothesis.get('confidence_source', 0.5)
            if confidence >= 0.8:
                context['confidence_level'] = 'high'
            elif confidence >= 0.6:
                context['confidence_level'] = 'moderate'
            else:
                context['confidence_level'] = 'low'
        
        return context

    def _create_narrative_template(self, question: str, context: dict) -> str:
        """Creates a structured, logical explanation template from the context."""
        status = context['status']
        answer = context['answer']
        reasoning = context.get('reasoning_description')
        evidence_descriptions = context.get('evidence_descriptions', [])

        if status == 'CONCLUSIVE':
            if reasoning and evidence_descriptions:
                evidence_text = "; ".join(evidence_descriptions)
                return f"The answer is '{answer}' because: {reasoning}. This is supported by evidence: {evidence_text}."
            else:
                return f"The answer is '{answer}' based on a clear logical analysis of the image."
        elif status == 'CONCLUSIVE_AFTER_CONFLICT':
            return f"After resolving conflicting possibilities, the answer is '{answer}'. The reasoning is: {reasoning or 'visual analysis confirms this'}"
        elif status == 'CONCLUSIVE_BY_FALLBACK':
            return f"The answer is '{answer}' based on initial visual analysis, as a definitive conclusion could not be reached from the evidence."
        else:
            return f"The answer is '{answer}' based on an analysis of the image."

    def _refine_with_llm(self, template: str, question: str, answer: str) -> str:
        """Uses the configured LLM to make the explanation template more natural."""
        if not self.client:
            self.logger.warning("No LLM client for ExplanationAgent; returning raw template.")
            return template

        prompt = f"""You are an expert at explaining VQA results. Your task is to transform a structured explanation into a natural, conversational one.

Original Question: "{question}"
Final Answer: "{answer}"
Structured Explanation: "{template}"

Please rewrite the structured explanation to be more human-readable and fluent. Keep it concise (1-2 sentences).

Natural Explanation:"""

        try:
            # Use the _make_request method from BaseAgent for retry logic
            explanation = self._make_request(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return explanation if explanation else template

        except Exception as e:
            self.logger.error(f"LLM-based explanation refinement failed: {e}")
            return template # Fallback to the structured template 