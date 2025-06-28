"""
StrategistAgent for MVKB-X Pipeline
The Strategist Agent (formerly SeekerAgent), based on an LLM.
Strategic reasoning agent responsible for Multi-View Knowledge Base (MVKB) construction,
hypothesis generation, and confidence assessment.
"""

import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from retrying import retry

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
    hypothesis generation, and confidence assessment.
    """
    
    def __init__(self, client: OpenAI = None, model_name: str = None, verifier=None, use_vllm: bool = True):
        super().__init__(use_vllm, model_name)
        
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
        Builds the complete Multi-View Knowledge Base by orchestrating strategic reasoning.
        Returns MVKB entries with structured format for Synthesizer.
        """
        logging.info(f"Strategist: Building MVKB for question '{question}'")
        mvkb = []

        relevant_issues = self._create_relevant_issues(question, answer_candidates, caption)
        logging.debug(f"Strategist: Generated relevant issues: {relevant_issues}")

        for issue in relevant_issues:
            # Use verifier to get answer for sub-question
            issue_response = self.verifier.generate_initial_response(issue, image_path)
            issue_answer = issue_response['answer_candidates'][0]
            
            # Calculate issue confidence based on answer quality and consistency
            issue_confidence = self._calculate_issue_confidence(issue, issue_answer, issue_response)
            logging.debug(f"Strategist: Answer for issue '{issue}' is '{issue_answer}' (confidence: {issue_confidence:.3f})")
            
            for candidate in answer_candidates:
                hypothesis_data = self._formulate_hypotheses_and_confidence(question, candidate, issue, issue_answer)
                if hypothesis_data:
                    # Use hypothesis confidence_score as the primary confidence metric
                    # This is more accurate than generic issue_confidence
                    primary_confidence = hypothesis_data.get("confidence_score", 0.5)
                    
                    # Combine issue confidence with hypothesis confidence for more robust scoring
                    combined_confidence = (primary_confidence + issue_confidence) / 2.0
                    
                    mvkb_entry = {
                        "original_question": question,
                        "answer_candidate": candidate,
                        "relevant_issue": issue,
                        "issue_answer": issue_answer,
                        "issue_confidence": combined_confidence,  # Use calculated confidence
                        "hypothesis": hypothesis_data.get("hypothesis"),
                        "confidence_score": primary_confidence,  # Keep original hypothesis confidence
                        "confidence_word": hypothesis_data.get("confidence_word")
                    }
                    mvkb.append(mvkb_entry)
        
        logging.info(f"Strategist: MVKB built with {len(mvkb)} entries.")
        return mvkb

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
    def _formulate_hypotheses_and_confidence(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> dict:
        """Formulate hypothesis and confidence for a specific answer candidate using templates."""
        if not self.client:
            logging.error("No working backend available for hypothesis formulation")
            return {}
        
        try:
            # Try template-based approach first
            if self.use_templates and self.prompt_manager:
                prompt = self.prompt_manager.render(
                    'agents/strategist/fdr_strategist_hypothesis.jinja',
                    question=question,
                    answer_candidate=answer_candidate,
                    relevant_issue=relevant_issue,
                    issue_answer=issue_answer
                )
                logging.debug("✅ Using template-based prompt for hypothesis formulation")
            else:
                # Fallback to hardcoded prompt
                prompt = f"""Given this information, formulate a hypothesis about whether the answer candidate is correct.

Original Question: {question}
Answer Candidate: {answer_candidate}
Relevant Issue: {relevant_issue}
Issue Answer: {issue_answer}

Task:
1. Create a logical hypothesis connecting the issue answer to the answer candidate
2. Assign a confidence score (0.0 to 1.0) based on how well the issue answer supports the candidate
3. Provide a confidence word (Very Likely, Likely, Possible, Unlikely)

Format your response as:
Hypothesis: [Your logical reasoning]
Confidence Score: [0.0-1.0]
Confidence Word: [Very Likely/Likely/Possible/Unlikely]"""
                logging.debug("⚠️ Using fallback hardcoded prompt for hypothesis formulation")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response if using templates
            if self.use_templates and self.prompt_manager and content.startswith('{'):
                import json
                try:
                    parsed = json.loads(content)
                    result = {
                        "hypothesis": parsed.get("hypothesis", ""),
                        "confidence_score": float(parsed.get("confidence_score", 0.5)),
                        "confidence_word": parsed.get("confidence_word", "Possible")
                    }
                    logging.debug("✅ Successfully parsed JSON response from hypothesis template")
                    return result
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"⚠️ Failed to parse JSON hypothesis response: {e}")
                    # Fall through to legacy parsing
            
            # Legacy parsing for fallback mode
            hypothesis = ""
            confidence_score = 0.5
            confidence_word = "Possible"
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Hypothesis:'):
                    hypothesis = line.replace('Hypothesis:', '').strip()
                elif line.startswith('Confidence Score:'):
                    try:
                        score_str = line.replace('Confidence Score:', '').strip()
                        confidence_score = float(score_str)
                        confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
                    except ValueError:
                        confidence_score = 0.5
                elif line.startswith('Confidence Word:'):
                    confidence_word = line.replace('Confidence Word:', '').strip()
            
            # Validate confidence word
            valid_words = ["Very Likely", "Likely", "Possible", "Unlikely"]
            if confidence_word not in valid_words:
                # Map confidence score to word
                if confidence_score >= 0.8:
                    confidence_word = "Very Likely"
                elif confidence_score >= 0.6:
                    confidence_word = "Likely"
                elif confidence_score >= 0.4:
                    confidence_word = "Possible"
                else:
                    confidence_word = "Unlikely"
            
            result = {
                "hypothesis": hypothesis or f"The issue answer '{issue_answer}' suggests that '{answer_candidate}' may be correct.",
                "confidence_score": confidence_score,
                "confidence_word": confidence_word
            }
            
            logging.debug(f"Hypothesis for '{answer_candidate}': {result}")
            return result
            
        except Exception as e:
            logging.error(f"Hypothesis formulation failed: {e}")
            return {
                "hypothesis": f"Analysis suggests '{answer_candidate}' as a potential answer.",
                "confidence_score": 0.5,
                "confidence_word": "Possible"
            } 

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