"""
SynthesizerAgent for MVKB-X Pipeline
The Synthesizer Agent (formerly IntegratorAgent), implementing Algorithm 2.
Algorithmic agent for weighted voting, evidence synthesis, and final decision making.
"""

import logging
from typing import Dict, List, Any

from .base import BaseAgent


class SynthesizerAgent:
    """
    The Synthesizer Agent (formerly IntegratorAgent), implementing Algorithm 2.
    Algorithmic agent for weighted voting, evidence synthesis, and final decision making.
    """
    
    def __init__(self, verifier):
        self.verifier = verifier

    def conduct_weighted_voting(self, original_question: str, image_path: str, answer_candidates: list, mvkb: list) -> dict:
        """
        Algorithm 2: The Synthesizer Agent
        Conducts weighted voting based on MVKB entries and returns result with explanation data.
        """
        logging.info("Synthesizer: Conducting weighted voting per Algorithm 2.")
        logging.debug(f"Answer candidates: {answer_candidates}")
        
        if not mvkb:
            logging.warning("Synthesizer: MVKB is empty. Returning first answer candidate as fallback.")
            return {
                "final_answer": answer_candidates[0] if answer_candidates else "No answer determined",
                "voting_pool": {},
                "mvkb_entries": [],
                "confidence_breakdown": {}
            }

        # Algorithm 2 Implementation
        voting_pool = {candidate: [] for candidate in answer_candidates}
        
        # Step 2-8: Process each MVKB entry
        for i, entry in enumerate(mvkb):
            hypothesis = entry.get("hypothesis", "")
            confidence_word = entry.get("confidence_word", "N/A")
            issue_confidence = entry.get("issue_confidence", 0.5)
            
            logging.debug(f"Processing MVKB entry {i+1}: confidence={issue_confidence:.3f}")
            
            # Step 3: Concatenate H, Φ, I, Q for contextual question
            context_prompt = f"""
Hypothesis (Confidence: {confidence_word}): {hypothesis}
"""
            
            # Step 4: Get Q* from Verifier (top-1 answer)
            contextual_answer = self.verifier.answer_contextual_question(
                question=original_question,
                image_path=image_path,
                context_prompt=context_prompt
            )
            
            logging.debug(f"Contextual answer: '{contextual_answer}'")
            
            # Step 5-7: Improved matching logic for answer candidates
            matched_candidate = self._match_answer_to_candidates(contextual_answer, answer_candidates)
            
            if matched_candidate:
                voting_pool[matched_candidate].append(issue_confidence)
                logging.debug(f"Synthesizer: Vote for '{matched_candidate}' with confidence {issue_confidence:.3f}")
            else:
                logging.debug(f"Synthesizer: No match found for contextual answer '{contextual_answer}' among candidates {answer_candidates}")

        # Step 9: Vote final answer with highest score
        vote_scores = {candidate: sum(scores) for candidate, scores in voting_pool.items()}
        
        logging.debug(f"Vote scores: {vote_scores}")
        
        if not any(vote_scores.values()):
            logging.warning("Synthesizer: No votes cast. Returning first candidate.")
            final_answer = answer_candidates[0]
        else:
            final_answer = max(vote_scores, key=vote_scores.get)
        
        logging.info(f"Synthesizer: Final answer chosen: '{final_answer}' with score {vote_scores.get(final_answer, 0):.3f}")
        
        return {
            "final_answer": final_answer,
            "voting_pool": voting_pool,
            "vote_scores": vote_scores,
            "mvkb_entries": mvkb,
            "confidence_breakdown": {
                "total_votes": sum(vote_scores.values()),
                "winning_score": vote_scores.get(final_answer, 0),
                "score_distribution": vote_scores
            }
        }

    def _match_answer_to_candidates(self, contextual_answer: str, answer_candidates: list) -> str:
        """
        Improved matching logic to handle short answers accurately.
        Returns the best matching candidate or None if no match found.
        """
        if not contextual_answer or not answer_candidates:
            return None
        
        contextual_answer = contextual_answer.lower().strip()
        logging.debug(f"Matching '{contextual_answer}' against candidates: {answer_candidates}")
        
        # Method 1: Exact match (highest priority)
        for candidate in answer_candidates:
            if candidate.lower().strip() == contextual_answer:
                logging.debug(f"Exact match found: '{candidate}'")
                return candidate
        
        # Method 2: Check if contextual answer starts with candidate (for short answers)
        for candidate in answer_candidates:
            candidate_lower = candidate.lower().strip()
            if contextual_answer.startswith(candidate_lower):
                # Additional check: make sure it's a word boundary
                if len(contextual_answer) == len(candidate_lower) or contextual_answer[len(candidate_lower)] in [' ', '.', ',', '!', '?', ';', ':']:
                    logging.debug(f"Start match found: '{candidate}'")
                    return candidate
        
        # Method 3: Word-level matching with priority for longer matches
        best_match = None
        best_score = 0
        
        for candidate in answer_candidates:
            candidate_lower = candidate.lower().strip()
            candidate_words = candidate_lower.split()
            
            # For single word candidates, check if it appears as a complete word
            if len(candidate_words) == 1:
                import re
                # Use word boundary regex to find complete word matches
                pattern = r'\b' + re.escape(candidate_lower) + r'\b'
                if re.search(pattern, contextual_answer):
                    # Score based on position (earlier matches get higher scores)
                    match_pos = contextual_answer.find(candidate_lower)
                    score = 10 - (match_pos / len(contextual_answer)) * 5  # Earlier = higher score
                    if score > best_score:
                        best_score = score
                        best_match = candidate
                        logging.debug(f"Word boundary match found: '{candidate}' (score: {score:.2f})")
            else:
                # Multi-word candidates: check how many words match
                matching_words = 0
                for word in candidate_words:
                    if word in contextual_answer:
                        matching_words += 1
                
                match_ratio = matching_words / len(candidate_words)
                score = match_ratio * 5  # Multi-word scoring
                
                if score > best_score and match_ratio > 0.5:  # At least 50% of words must match
                    best_score = score
                    best_match = candidate
                    logging.debug(f"Multi-word match found: '{candidate}' (score: {score:.2f}, ratio: {match_ratio:.2f})")
        
        # Method 4: Fallback - simple substring match for very short answers
        if not best_match:
            for candidate in answer_candidates:
                candidate_lower = candidate.lower().strip()
                if len(candidate_lower) <= 3 and candidate_lower in contextual_answer:
                    # But avoid false positives for common short words
                    if candidate_lower not in ['a', 'an', 'the', 'is', 'are', 'it', 'in', 'on', 'at', 'to']:
                        logging.debug(f"Substring match found: '{candidate}'")
                        return candidate
        
        if best_match:
            logging.debug(f"Best match selected: '{best_match}' (score: {best_score:.2f})")
        else:
            logging.debug(f"No match found for '{contextual_answer}'")
        
        return best_match 