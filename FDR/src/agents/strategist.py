"""
StrategistAgent for FDR Pipeline
The Strategist Agent (formerly SeekerAgent), based on an LLM.
Strategic reasoning agent responsible for Multi-View Knowledge Base (MVKB) construction,
hypothesis generation, confidence assessment, and explanation generation.

Enhanced for Design V2: Integrated explanation generation capabilities.
"""

import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from retrying import retry
import json

from .base import BaseAgent

# Import the new prompt management system
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "prompts"))
    from prompt_manager import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
    logging.info("✅ PromptManager imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ PromptManager not available: {e}")
    PROMPT_MANAGER_AVAILABLE = False


class StrategistAgent(BaseAgent):
    """
    The Strategist Agent (formerly SeekerAgent), based on an LLM.
    Strategic reasoning agent responsible for Multi-View Knowledge Base (MVKB) construction,
    hypothesis generation, confidence assessment, and explanation generation.
    
    Enhanced for Design V2: Integrated explanation generation capabilities.
    """
    
    def __init__(self, client: OpenAI = None, model_name: str = None, verifier=None, use_vllm: bool = True):
        super().__init__(use_vllm, model_name)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend
        self._initialize_backend()
        
        self.verifier = verifier
        
        # Initialize prompt manager
        if PROMPT_MANAGER_AVAILABLE:
            try:
                prompts_dir = Path(__file__).parent.parent / "prompts"
                self.prompt_manager = PromptManager(prompts_dir, enable_hot_reload=True)
                self.use_templates = True
                logging.info("✅ Strategist: Template system initialized")
            except Exception as e:
                logging.warning(f"⚠️ Strategist: Template system fallback: {e}")
                self.prompt_manager = None
                self.use_templates = False
        else:
            self.prompt_manager = None
            self.use_templates = False

    def build_mvkb(self, question: str, image_path: str, answer_candidates: list, caption: str) -> list:
        """
        Builds a Multi-View Knowledge Base (MVKB) by decomposing the question,
        formulating hypotheses, and gathering evidence. This version is guided
        by a structured reasoning process learned from dataset examples.
        
        Returns the enhanced format with reasoning_description and issue_description
        as per Design V2 for improved explanation generation.
        """
        logging.info("Strategist: Building MVKB with structured reasoning.")
        
        # Step 1: Generate a structured reasoning plan (issues + hypothesis) from the LLM
        reasoning_plan = self._generate_reasoning_plan(question, caption, answer_candidates)
        
        if not reasoning_plan:
            logging.error("Strategist: Failed to generate a reasoning plan.")
            return []

        # Step 2: Verify each relevant issue using the VerifierAgent
        evidence_set = []
        relevant_issues = reasoning_plan.get("relevant_issues", [])
        
        for issue in relevant_issues:
            issue_id = issue.get("issue_id", "unknown_issue")
            issue_text = issue.get("question_text", "")
            
            if not issue_text:
                continue

            # Verifier answers the sub-question
            issue_answer = self.verifier.answer_contextual_question(
                question=issue_text,
                image_path=image_path,
                context_prompt="Answer this question based on the image:"
            )
            
            # Enhanced evidence format with issue_description for better explanation generation
            evidence_set.append({
                "evidence_id": issue_id,
                "issue_text": issue_text,
                "issue_description": self._generate_issue_description(issue_text, issue_answer),
                "answer": issue_answer.strip(),
                "confidence": 0.85,  # Default confidence for verified evidence
                "source": "VerifierAgent"
            })
            logging.debug(f"Strategist: Verified issue '{issue_id}' -> Answer: '{issue_answer.strip()}'")

        # Step 3: Enhanced hypothesis with reasoning_description
        hypothesis = reasoning_plan.get("hypothesis", {})
        if hypothesis:
            # Add reasoning_description for explanation generation
            hypothesis["reasoning_description"] = self._generate_reasoning_description(
                hypothesis, question, answer_candidates
            )
            hypothesis["confidence_source"] = 0.8  # Default confidence for generated hypotheses
        
        # Step 4: Combine the evidence and hypothesis into the final format for the Synthesizer
        mvkb_payload = {
            "evidence_set": evidence_set,
            "hypothesis_set": [hypothesis] if hypothesis else []
        }
        
        logging.info("Strategist: Successfully built enhanced MVKB payload with descriptions.")
        return mvkb_payload

    def generate_explanation(self, question: str, synthesis_result: dict, caption: str, evidence_set: list) -> str:
        """
        Generate natural language explanation from synthesis results.
        This integrates explanation generation directly into the Strategist.
        """
        if not synthesis_result:
            return "Unable to generate explanation due to synthesis failure."
        
        try:
            # Step 1: Build narrative context from synthesis result
            context = self._build_narrative_context(synthesis_result, evidence_set)
            
            # Step 2: Create narrative template
            template = self._create_narrative_template(question, synthesis_result, context)
            
            # Step 3: Generate natural explanation using LLM
            explanation = self._generate_with_llm(template, question, synthesis_result, caption)
            
            self.logger.debug(f"Strategist explanation generated: {explanation}")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Strategist explanation generation failed: {e}")
            
            # Fallback to simple explanation
            status = synthesis_result.get('status', 'UNKNOWN')
            answer = synthesis_result.get('answer', 'unknown')
            
            if status == 'CONCLUSIVE':
                return f"The answer is {answer} based on logical analysis of the image."
            elif status == 'CONCLUSIVE_AFTER_CONFLICT':
                return f"The answer is {answer} after resolving multiple possibilities from the visual evidence."
            elif status == 'CONCLUSIVE_BY_FALLBACK':
                return f"The answer is {answer} based on initial visual analysis."
            else:
                return f"The answer is {answer} based on image analysis."

    def _build_narrative_context(self, synthesis_result: Dict[str, Any], evidence_set: List[Dict]) -> Dict[str, Any]:
        """
        Build narrative context from causal trace and evidence.
        """
        causal_trace = synthesis_result.get('causal_trace', [])
        status = synthesis_result.get('status', 'UNKNOWN')
        answer = synthesis_result.get('answer')
        
        context = {
            'answer': answer,
            'status': status,
            'reasoning_description': None,
            'evidence_descriptions': [],
            'confidence_level': 'moderate'
        }
        
        # Extract reasoning from causal trace
        if causal_trace:
            # Get the first (or winning) hypothesis
            winning_hypothesis = causal_trace[0]
            context['reasoning_description'] = winning_hypothesis.get('reasoning_description', '')
            
            # Get evidence descriptions
            triggered_evidence_ids = winning_hypothesis.get('triggered_by_evidence', [])
            evidence_map = {e.get('evidence_id'): e for e in evidence_set}
            
            for evidence_id in triggered_evidence_ids:
                if evidence_id in evidence_map:
                    evidence = evidence_map[evidence_id]
                    description = evidence.get('issue_description', evidence.get('issue_text', ''))
                    context['evidence_descriptions'].append(description)
            
            # Determine confidence level
            confidence = winning_hypothesis.get('confidence_source', 0.5)
            if confidence >= 0.8:
                context['confidence_level'] = 'high'
            elif confidence >= 0.6:
                context['confidence_level'] = 'moderate'
            else:
                context['confidence_level'] = 'low'
        
        return context
    
    def _create_narrative_template(self, question: str, synthesis_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Create a narrative template based on synthesis status.
        """
        status = synthesis_result.get('status')
        answer = context['answer']
        reasoning = context['reasoning_description']
        evidence_descriptions = context['evidence_descriptions']
        
        if status == 'CONCLUSIVE':
            # Strong logical conclusion
            if reasoning and evidence_descriptions:
                evidence_text = "; ".join(evidence_descriptions)
                template = f"The answer is '{answer}' based on logical reasoning: {reasoning}. " + \
                          f"This conclusion is supported by evidence: {evidence_text}."
            else:
                template = f"The answer is '{answer}' based on clear logical analysis of the image."
                
        elif status == 'CONCLUSIVE_AFTER_CONFLICT':
            # Resolved conflict  
            template = f"The answer is '{answer}' after resolving multiple possibilities. " + \
                      f"The reasoning: {reasoning or 'visual analysis confirms this as the most likely answer'}."
                      
        elif status == 'CONCLUSIVE_BY_FALLBACK':
            # Best guess fallback
            template = f"The answer is '{answer}' based on initial visual analysis, " + \
                      "as a definitive logical conclusion could not be reached from the available evidence."
                      
        else:
            # Generic template
            template = f"The answer is '{answer}' based on analysis of the image."
        
        return template
    
    def _generate_with_llm(self, template: str, question: str, synthesis_result: Dict[str, Any], caption: str) -> str:
        """
        Use LLM to transform template into natural, fluent explanation.
        """
        if not self.client:
            return template
            
        status = synthesis_result.get('status', 'UNKNOWN')
        answer = synthesis_result.get('answer')
        
        # Build prompt for LLM
        prompt = f"""You are an expert at explaining visual question answering results. Transform the following logical explanation into a natural, conversational explanation.

ORIGINAL QUESTION: {question}

IMAGE CONTEXT: {caption}

LOGICAL ANALYSIS RESULT: {template}

REASONING STATUS: {status}

YOUR TASK:
- Rewrite the explanation to be natural and conversational
- Remove any robotic or template-like language
- Keep the explanation concise (1-2 sentences maximum)
- Maintain the logical reasoning but make it sound human
- Use appropriate confidence language based on the reasoning status

Now transform this explanation:
{template}

Natural explanation:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistent, focused explanations
                max_tokens=100,   # Keep explanations concise
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Ensure the explanation starts appropriately
            if not any(explanation.lower().startswith(phrase) for phrase in 
                      ['the answer', 'this', 'the image', 'based on', 'the']):
                explanation = f"The answer is {answer}. {explanation}"
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"LLM explanation generation failed: {e}")
            # Fallback to template
            return template

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _generate_reasoning_plan(self, question: str, caption: str, answer_candidates: list) -> Optional[Dict]:
        """
        Generates a structured reasoning plan (issues, hypothesis) using a dedicated prompt.
        This is the core of the new "Learning-Guided" strategy.
        """
        if not self.client:
            logging.error("Strategist: LLM client not available.")
            return None
        
        prompt = self._build_reasoning_prompt(question, caption, answer_candidates)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"} # Force JSON output
            )
            
            content = response.choices[0].message.content.strip()
            plan = json.loads(content)
            
            logging.info("Strategist: Successfully generated reasoning plan.")
            logging.debug(f"Reasoning Plan: {plan}")
            return plan

        except Exception as e:
            logging.error(f"Strategist: Failed to generate or parse reasoning plan: {e}")
            return None

    def _build_reasoning_prompt(self, question: str, caption: str, answer_candidates: list) -> str:
        """Builds the prompt for the LLM to generate a reasoning plan."""
        
        return f"""You are an expert reasoning agent for Visual Question Answering. Your task is to create a logical plan to answer a question based on an image.

Follow this two-step process:

**Step 1: Decompose the Main Question**
Break down the main question into 2-3 smaller, factual, and verifiable sub-questions ("Relevant Issues"). These issues should act as building blocks of evidence. Each issue must have a unique `issue_id` and a `question_text`.

**Step 2: Formulate a Logical Hypothesis**
Create a single, clear logical rule ("Hypothesis"). This rule must use the answers to your "Relevant Issues" to logically deduce the final answer.

**IMPORTANT: JSON Structure Rules**
- The `IF` clause in the hypothesis **MUST** contain a list of objects.
- Each object in the `IF` list **MUST** have two keys: `issue_id` (matching an ID from "Relevant Issues") and `answer_is` (the expected answer for that issue).
- The `THEN` clause **MUST** contain an object with a single key: `final_answer`.

**EXAMPLE 1:**
- **Main Question**: "What does the weather seem to be like?"
- **Image Caption**: "A photo shows people in heavy coats walking on a snowy sidewalk."
- **Answer Candidates**: ["Sunny", "Rainy", "Cold", "Warm"]
- **Your Output (JSON Object):**
{{
  "relevant_issues": [
    {{"issue_id": "issue_01", "question_text": "Are the people in the photo wearing heavy coats or winter jackets?"}},
    {{"issue_id": "issue_02", "question_text": "Is there visible snow or ice on the ground?"}}
  ],
  "hypothesis": {{
    "hypothesis_id": "H_Weather_Cold",
    "IF": [
      {{"issue_id": "issue_01", "answer_is": "Yes"}},
      {{"issue_id": "issue_02", "answer_is": "Yes"}}
    ],
    "THEN": {{"final_answer": "Cold"}}
  }}
}}

---

**YOUR TASK:**
Apply this reasoning process to the following task. Provide your output as a single, strictly formatted JSON object that follows all the rules.

- **Main Question**: "{question}"
- **Image Caption**: "{caption}"
- **Answer Candidates**: {json.dumps(answer_candidates)}

**Your JSON Output:**
"""

    def _formulate_hypotheses_and_confidence(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> dict:
        """DEPRECATED - Replaced by _generate_reasoning_plan"""
        pass

    def _get_issue_confidence(self, issue_response: dict) -> float:
        """DEPRECATED - Replaced by structured reasoning plan"""
        pass

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _create_relevant_issues(self, question: str, answer_candidates: list, caption: str) -> list:
        """Create relevant sub-questions/issues for MVKB construction using templates."""
        if not self.client:
            logging.error("No working backend available for relevant issues creation")
            return ["What objects are visible?", "What is the main subject?"]
        
        try:
            # Try template-based approach first
            if self.use_templates and self.prompt_manager:
                prompt = self.prompt_manager.render(
                    'agents/strategist/fdr_strategist_mvkb_creation.jinja',
                    question=question,
                    answer_candidates=answer_candidates,
                    caption=caption
                )
                logging.debug("✅ Using template-based prompt for relevant issues")
            else:
                # Fallback to hardcoded prompt
                prompt = f"""Given this visual question and potential answers, create 2-3 relevant sub-questions that would help determine the correct answer.

Main Question: {question}
Potential Answers: {', '.join(answer_candidates)}
Image Description: {caption}

Create specific, focused sub-questions that:
1. Address different aspects of the visual scene
2. Help distinguish between the answer candidates
3. Can be answered by looking at the image

Format: Return only the sub-questions, one per line."""
                logging.debug("⚠️ Using fallback hardcoded prompt for relevant issues")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response if using templates
            if self.use_templates and self.prompt_manager and content.startswith('{'):
                import json
                try:
                    parsed = json.loads(content)
                    issues = parsed.get('relevant_issues', [])
                    logging.debug("✅ Successfully parsed JSON response from template")
                except json.JSONDecodeError:
                    logging.warning("⚠️ Failed to parse JSON, falling back to line parsing")
                    issues = [line.strip() for line in content.split('\n') if line.strip()]
            else:
                # Parse line-by-line for fallback mode
                issues = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Limit to 3 issues for efficiency
            issues = issues[:3]
            
            if not issues:
                # Fallback issues
                issues = [
                    "What is the main object or subject in the image?",
                    "What specific details are visible that relate to the question?",
                    "What is the context or setting of the image?"
                ]
            
            logging.debug(f"Generated relevant issues: {issues}")
            return issues
            
        except Exception as e:
            logging.error(f"Relevant issues creation failed: {e}")
            return ["What objects are visible?", "What is the main subject?", "What are the key details?"]

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _calculate_issue_confidence(self, issue: str, issue_answer: str, issue_response: dict) -> float:
        """Calculate confidence for an issue answer based on various factors"""
        confidence = 0.5  # Base confidence
        
        try:
            # Factor 1: Answer length and specificity (short, specific answers often more confident)
            if issue_answer and len(issue_answer.strip()) > 0:
                answer_words = len(issue_answer.split())
                if answer_words == 1:
                    confidence += 0.2  # Single word answers often more confident
                elif answer_words <= 3:
                    confidence += 0.1  # Short answers
                else:
                    confidence -= 0.1  # Very long answers might be less confident
            
            # Factor 2: Answer type detection
            answer_lower = issue_answer.lower().strip()
            
            # High confidence for clear yes/no answers
            if answer_lower in ['yes', 'no']:
                confidence += 0.3
            
            # High confidence for numbers
            elif answer_lower.isdigit():
                confidence += 0.3
            
            # Medium confidence for common objects/colors
            elif any(word in answer_lower for word in ['person', 'people', 'man', 'woman', 'child', 
                                                      'car', 'truck', 'bike', 'dog', 'cat',
                                                      'red', 'blue', 'green', 'black', 'white',
                                                      'big', 'small', 'large']):
                confidence += 0.2
            
            # Factor 3: Multiple candidates indicate uncertainty
            candidates = issue_response.get('answer_candidates', [])
            if len(candidates) > 1:
                confidence -= 0.1  # Multiple candidates = less certain
            
            # Factor 4: Error indicators reduce confidence
            if any(word in answer_lower for word in ['error', 'unknown', 'unclear', 'uncertain']):
                confidence -= 0.3
            
            # Clamp confidence to [0.1, 0.9] range
            confidence = max(0.1, min(0.9, confidence))
            
        except Exception as e:
            logging.warning(f"Issue confidence calculation failed: {e}")
            confidence = 0.5
        
        return confidence 

    def _generate_issue_description(self, issue_text: str, issue_answer: str) -> str:
        """
        Generate a human-readable description of what the issue is checking for.
        This enhances explanation generation by providing context.
        """
        # Extract key concepts from the issue text for description
        if "how many" in issue_text.lower():
            return f"Counting the number of objects or elements in the image: {issue_answer}"
        elif "what color" in issue_text.lower():
            return f"Identifying the color of the specified object: {issue_answer}"
        elif "is there" in issue_text.lower() or "are there" in issue_text.lower():
            return f"Verifying the presence of specific elements: {issue_answer}"
        elif "what room" in issue_text.lower() or "where" in issue_text.lower():
            return f"Determining the location or setting: {issue_answer}"
        elif "what method" in issue_text.lower() or "how" in issue_text.lower():
            return f"Identifying the method or process: {issue_answer}"
        else:
            # General case
            key_words = [word for word in issue_text.split() if len(word) > 3 and word.lower() not in 
                        ['what', 'where', 'when', 'how', 'which', 'does', 'are', 'is', 'the', 'this', 'that']]
            if key_words:
                focus = " ".join(key_words[:3])  # Take first 3 meaningful words
                return f"Analyzing {focus} in the image: {issue_answer}"
            else:
                return f"Visual analysis question: {issue_answer}"

    def _generate_reasoning_description(self, hypothesis: dict, question: str, answer_candidates: list) -> str:
        """
        Generate a natural language description of the logical reasoning behind the hypothesis.
        This provides context for the explanation generator.
        """
        if not hypothesis or 'IF' not in hypothesis or 'THEN' not in hypothesis:
            return "Logical reasoning based on visual evidence analysis"
        
        if_conditions = hypothesis['IF']
        then_result = hypothesis['THEN'].get('final_answer', 'unknown')
        
        # Build reasoning description
        if len(if_conditions) == 1:
            condition = if_conditions[0]
            issue_id = condition.get('issue_id', 'evidence')
            expected_answer = condition.get('answer_is', 'confirmed')
            return f"If the visual analysis confirms {expected_answer}, then the answer is {then_result}"
        
        elif len(if_conditions) == 2:
            cond1 = if_conditions[0]
            cond2 = if_conditions[1]
            ans1 = cond1.get('answer_is', 'confirmed')
            ans2 = cond2.get('answer_is', 'confirmed')
            return f"If both visual checks confirm {ans1} and {ans2}, then the answer is {then_result}"
        
        else:
            # Multiple conditions
            answers = [cond.get('answer_is', 'confirmed') for cond in if_conditions]
            conditions_text = ", ".join(answers)
            return f"If multiple visual analyses confirm {conditions_text}, then the answer is {then_result}" 