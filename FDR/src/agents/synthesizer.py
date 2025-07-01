"""
Synthesizer Logic Engine for FDR Pipeline
A deterministic, algorithmic engine for evidence synthesis and logical conclusion.

Design Principles:
1. Deterministic: Same input always produces same output
2. Verifiable: Logic simple enough for manual audit
3. Modular: Single responsibility - logical synthesis only
4. Stateless: No memory between queries
"""

import logging
from typing import Dict, List, Any, Optional


class SynthesizerEngine:
    """
    Synthesizer Logic Engine - A deterministic algorithm for evidence synthesis.
    
    This is NOT an LLM. It's a pure logical engine that:
    - Takes evidence_set and hypothesis_set as input
    - Applies logical rules to derive conclusions
    - Returns deterministic results with causal traces
    """
    
    def __init__(self):
        """Initialize the Synthesizer Engine."""
        self.logger = logging.getLogger(__name__)
    
    def synthesize(self, evidence_set: List[Dict], hypothesis_set: List[Dict], answer_candidates: List[str]) -> Dict[str, Any]:
        """
        Main synthesis function with multi-tiered decision logic to always return an answer.
        
        Args:
            evidence_set: List of evidence objects from Verifier.
            hypothesis_set: List of causal rules from Strategist.
            answer_candidates: List of initial answer candidates from Verifier for fallback.
        
        Returns:
            Dict with keys: status, answer, causal_trace
        """
        self.logger.info("Synthesizer Engine: Starting multi-tiered logical synthesis")
        
        # Step 1: Build evidence map for fast lookup
        evidence_map = self._build_evidence_map(evidence_set)
        
        # Step 2: Evaluate all hypotheses
        triggered_hypotheses = []
        for hypothesis in hypothesis_set:
            if self._validate_hypothesis_format(hypothesis) and self._check_hypothesis_conditions(hypothesis, evidence_map):
                triggered_hypotheses.append(hypothesis)
                self.logger.debug(f"Hypothesis {hypothesis['hypothesis_id']} triggered -> {hypothesis['THEN']['final_answer']}")
        
        # Step 3: Multi-tiered decision making
        result = self._evaluate_and_decide(triggered_hypotheses, answer_candidates)
        
        self.logger.info(f"Synthesis complete. Status: {result['status']}, Answer: {result.get('answer', 'None')}")
        return result
    
    def _build_evidence_map(self, evidence_set: List[Dict]) -> Dict[str, str]:
        """Build a map from evidence_id to answer for fast lookup."""
        evidence_map = {}
        
        for evidence in evidence_set:
            if 'evidence_id' in evidence and 'answer' in evidence:
                evidence_map[evidence['evidence_id']] = evidence['answer']
            else:
                self.logger.warning(f"Invalid evidence format: {evidence}")
        
        return evidence_map
    
    def _validate_hypothesis_format(self, hypothesis: Dict) -> bool:
        """Validate that hypothesis has required structure."""
        required_keys = ['hypothesis_id', 'IF', 'THEN']
        
        if not all(key in hypothesis for key in required_keys):
            return False
        
        if not isinstance(hypothesis['IF'], list):
            return False
        
        if 'final_answer' not in hypothesis['THEN']:
            return False
        
        # Validate each condition in IF clause
        for condition in hypothesis['IF']:
            if not isinstance(condition, dict):
                return False
            if 'evidence_id' not in condition or 'answer_is' not in condition:
                return False
        
        return True
    
    def _check_hypothesis_conditions(self, hypothesis: Dict, evidence_map: Dict[str, str]) -> bool:
        """Check if all conditions in hypothesis IF clause are met."""
        conditions_met = True
        
        for condition in hypothesis['IF']:
            evidence_id = condition['evidence_id']
            required_answer = condition['answer_is']
            
            # Check if evidence exists and matches required answer
            if evidence_id not in evidence_map:
                self.logger.debug(f"Evidence {evidence_id} not found in evidence map")
                conditions_met = False
                break
            
            if evidence_map[evidence_id] != required_answer:
                self.logger.debug(f"Evidence {evidence_id}: got '{evidence_map[evidence_id]}', required '{required_answer}'")
                conditions_met = False
                break
        
        return conditions_met
    
    def _evaluate_and_decide(self, triggered_hypotheses: List[Dict], answer_candidates: List[str]) -> Dict[str, Any]:
        """
        Multi-tiered decision logic to always produce an answer.
        Tier 1: Pure Logic -> Tier 2: Conflict Resolution -> Tier 3: Best-Guess Fallback
        
        Enhanced for Design V2: Includes reasoning_description and evidence details in causal_trace.
        """
        # Build enhanced causal trace with metadata for explanation generation
        causal_trace = []
        for h in triggered_hypotheses:
            trace_entry = {
                "hypothesis_id": h['hypothesis_id'],
                "reasoning_description": h.get('reasoning_description', 'Logical reasoning based on evidence'),
                "triggered_by_evidence": [c['evidence_id'] for c in h['IF']],
                "confidence_source": h.get('confidence_source', 0.5)
            }
            causal_trace.append(trace_entry)
        
        # --- Tier 1: Pure Logic ---
        if len(triggered_hypotheses) > 0:
            unique_conclusions = {h['THEN']['final_answer'] for h in triggered_hypotheses}
            if len(unique_conclusions) == 1:
                final_answer = unique_conclusions.pop()
                self.logger.info(f"Tier 1 (CONCLUSIVE): Single logical conclusion found: {final_answer}")
                return {"status": "CONCLUSIVE", "answer": final_answer, "causal_trace": causal_trace}
        
            # --- Tier 2: Conflict Resolution using Confidence ---
            self.logger.warning(f"Tier 2 (CONFLICT RESOLUTION): Multiple conclusions triggered: {unique_conclusions}. Resolving with confidence.")
            best_hypothesis = max(triggered_hypotheses, key=lambda h: h.get('confidence_source', 0))
            final_answer = best_hypothesis['THEN']['final_answer']
            max_confidence = best_hypothesis.get('confidence_source', 0)
            
            # Check for a true tie in confidence
            ties = [h for h in triggered_hypotheses if h.get('confidence_source', 0) == max_confidence]
            if len({h['THEN']['final_answer'] for h in ties}) > 1:
                self.logger.error(f"Unresolvable conflict with tied confidence {max_confidence}. Falling back.")
                # Fall through to Tier 3
            else:
                self.logger.info(f"Conflict resolved. Answer '{final_answer}' chosen with confidence {max_confidence:.2f}.")
                
                # Enhanced causal trace for conflict resolution - only include winning hypothesis
                winning_trace = [trace for trace in causal_trace 
                               if trace["hypothesis_id"] == best_hypothesis['hypothesis_id']]
                
                return {
                    "status": "CONCLUSIVE_AFTER_CONFLICT",
                    "answer": final_answer,
                    "causal_trace": winning_trace
                }

        # --- Tier 3: Best-Guess Fallback ---
        if answer_candidates:
            final_answer = answer_candidates[0]
            self.logger.warning(f"Tier 3 (FALLBACK): Using first answer candidate '{final_answer}'.")
            return {
                "status": "CONCLUSIVE_BY_FALLBACK",
                "answer": final_answer,
                "causal_trace": []
            }
        else:
            # Absolute last resort
            final_answer = "Unavailable"
            self.logger.error("Tier 3 (FALLBACK): No answer candidates available. Returning 'Unavailable'.")
            return {
                "status": "CONCLUSIVE_BY_FALLBACK",
                "answer": final_answer,
                "causal_trace": []
            }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "status": "ERROR",
            "answer": None,
            "causal_trace": [],
            "error": error_message
        }


# Backward compatibility wrapper for existing codebase
class SynthesizerAgent:
    """
    Backward compatibility wrapper for the new SynthesizerEngine.
    Maintains the same interface as the old SynthesizerAgent.
    """
    
    def __init__(self, verifier=None):
        """Initialize with optional verifier (maintained for compatibility)."""
        self.engine = SynthesizerEngine()
        self.verifier = verifier  # Kept for compatibility but not used in new logic
        self.logger = logging.getLogger(__name__)
    
    def conduct_weighted_voting(self, original_question: str, image_path: str, 
                              answer_candidates: list, mvkb: list) -> dict:
        """
        Legacy interface maintained for backward compatibility.
        
        Note: This method expects the new format in mvkb:
        - evidence_set: List of evidence from Verifier
        - hypothesis_set: List of logical rules from Strategist
        """
        self.logger.warning("Legacy conduct_weighted_voting called. Consider migrating to new synthesize() method.")
        
        # Extract evidence_set and hypothesis_set from mvkb
        # This assumes Strategist has formatted mvkb correctly
        evidence_set = []
        hypothesis_set = []
        
        if isinstance(mvkb, dict):
            evidence_set = mvkb.get('evidence_set', [])
            hypothesis_set = mvkb.get('hypothesis_set', [])
        elif isinstance(mvkb, list) and len(mvkb) > 0:
            # Handle case where mvkb is still in old format
            self.logger.warning("Old MVKB format detected. Attempting conversion...")
            # For now, return inconclusive - this needs Strategist to provide proper format
            return {
                "final_answer": answer_candidates[0] if answer_candidates else "Unknown",
                "voting_pool": {},
                "vote_scores": {},
                "mvkb_entries": mvkb,
                "confidence_breakdown": {},
                "synthesizer_status": "LEGACY_FORMAT_ERROR",
                "note": "Please update Strategist to provide evidence_set and hypothesis_set format"
            }
        
        # Use new engine
        result = self.engine.synthesize(evidence_set, hypothesis_set, answer_candidates)
        
        # Convert to legacy format for backward compatibility
        return {
            "final_answer": result.get("answer", answer_candidates[0] if answer_candidates else "Unknown"),
            "voting_pool": {},  # Not applicable in new logic
            "vote_scores": {},  # Not applicable in new logic
            "mvkb_entries": evidence_set,
            "confidence_breakdown": {
                "synthesizer_status": result["status"],
                "causal_trace": result.get("causal_trace", [])
            },
            "synthesizer_result": result  # Full new format result
        }
    
    def synthesize(self, evidence_set: List[Dict], hypothesis_set: List[Dict], answer_candidates: List[str] = None) -> Dict[str, Any]:
        """
        New interface for the logical synthesis engine.
        
        Args:
            evidence_set: Evidence from Verifier.
            hypothesis_set: Logical rules from Strategist.
            answer_candidates: List of initial answer candidates for fallback.
        
        Returns:
            A dictionary with status, answer, and causal_trace.
        """
        if answer_candidates is None:
            answer_candidates = [] # Ensure it's a list
        return self.engine.synthesize(evidence_set, hypothesis_set, answer_candidates) 