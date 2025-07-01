"""
Synthesizer Logic Engine for MVKB-X Pipeline
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
    
    def synthesize(self, evidence_set: List[Dict], hypothesis_set: List[Dict]) -> Dict[str, Any]:
        """
        Main synthesis function implementing the logical engine.
        
        Args:
            evidence_set: List of evidence objects from Verifier
                Format: [{"evidence_id": "E01", "issue_text": "...", "answer": "Yes", "confidence": 0.88}, ...]
            
            hypothesis_set: List of causal rules from Strategist  
                Format: [{"hypothesis_id": "H1", "IF": [{"evidence_id": "E01", "answer_is": "Yes"}], "THEN": {"final_answer": "Fry"}}, ...]
        
        Returns:
            Dict with keys: status, answer, causal_trace
            status: "CONCLUSIVE" | "INCONCLUSIVE" | "CONTRADICTORY"
        """
        self.logger.info("Synthesizer Engine: Starting logical synthesis")
        self.logger.debug(f"Evidence set: {evidence_set}")
        self.logger.debug(f"Hypothesis set: {hypothesis_set}")
        
        # Input validation
        if not isinstance(evidence_set, list) or not isinstance(hypothesis_set, list):
            self.logger.error("Invalid input types: evidence_set and hypothesis_set must be lists")
            return self._create_error_response("Invalid input types")
        
        # Step 1: Convert evidence_set to map for fast lookup
        evidence_map = self._build_evidence_map(evidence_set)
        self.logger.debug(f"Evidence map: {evidence_map}")
        
        # Step 2: Process each hypothesis to check conditions
        triggered_conclusions = []
        causal_trace = []
        
        for hypothesis in hypothesis_set:
            if not self._validate_hypothesis_format(hypothesis):
                self.logger.warning(f"Invalid hypothesis format: {hypothesis}")
                continue
                
            conditions_met = self._check_hypothesis_conditions(hypothesis, evidence_map)
            
            if conditions_met:
                conclusion = hypothesis['THEN']['final_answer']
                triggered_conclusions.append(conclusion)
                
                trace_entry = {
                    "hypothesis_id": hypothesis['hypothesis_id'],
                    "triggered_by_evidence": [c['evidence_id'] for c in hypothesis['IF']]
                }
                causal_trace.append(trace_entry)
                
                self.logger.debug(f"Hypothesis {hypothesis['hypothesis_id']} triggered -> {conclusion}")
        
        # Step 3: Check consistency of conclusions
        result = self._evaluate_conclusions(triggered_conclusions, causal_trace)
        
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
    
    def _evaluate_conclusions(self, triggered_conclusions: List[str], causal_trace: List[Dict]) -> Dict[str, Any]:
        """Evaluate the consistency of triggered conclusions."""
        unique_conclusions = set(triggered_conclusions)
        
        if len(unique_conclusions) == 0:
            # No hypothesis was triggered
            self.logger.info("No hypotheses triggered - INCONCLUSIVE")
            return {
                "status": "INCONCLUSIVE",
                "answer": None,
                "causal_trace": [],
                "explanation": "No logical rules were satisfied by the available evidence"
            }
        
        elif len(unique_conclusions) > 1:
            # Contradictory conclusions
            self.logger.warning(f"Contradictory conclusions found: {unique_conclusions}")
            return {
                "status": "CONTRADICTORY", 
                "answer": None,
                "causal_trace": causal_trace,
                "contradictory_answers": list(unique_conclusions),
                "explanation": f"Multiple contradictory conclusions reached: {list(unique_conclusions)}"
            }
        
        else:
            # Single consistent conclusion
            final_answer = unique_conclusions.pop()
            self.logger.info(f"Conclusive result: {final_answer}")
            return {
                "status": "CONCLUSIVE",
                "answer": final_answer,
                "causal_trace": causal_trace,
                "explanation": f"Logical conclusion reached: {final_answer}"
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
        result = self.engine.synthesize(evidence_set, hypothesis_set)
        
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
    
    def synthesize(self, evidence_set: List[Dict], hypothesis_set: List[Dict]) -> Dict[str, Any]:
        """
        New interface for the logical synthesis engine.
        
        Args:
            evidence_set: Evidence from Verifier in format:
                [{"evidence_id": "E01", "issue_text": "...", "answer": "Yes", "confidence": 0.88}, ...]
            
            hypothesis_set: Logical rules from Strategist in format:
                [{"hypothesis_id": "H1", "IF": [{"evidence_id": "E01", "answer_is": "Yes"}], 
                  "THEN": {"final_answer": "Answer"}}, ...]
        
        Returns:
            {"status": "CONCLUSIVE|INCONCLUSIVE|CONTRADICTORY", "answer": "...", "causal_trace": [...]}
        """
        return self.engine.synthesize(evidence_set, hypothesis_set) 