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


# ===== V3 WEIGHTED VOTING SYNTHESIZER ENGINE =====
class SynthesizerV3Engine:
    """
    Synthesizer V3 Logic Engine - Weighted Voting Algorithm
    
    Key Differences from V2:
    - Weighted voting instead of winner-takes-all
    - Confidence breakdown for all answer candidates
    - Comprehensive evidence utilization
    - Enhanced transparency for explanation generation
    """
    
    def __init__(self):
        """Initialize the V3 Synthesizer Engine."""
        self.logger = logging.getLogger(__name__)
    
    def synthesize(self, evidence_set: List[Dict], hypothesis_set: List[Dict], answer_candidates: List[str]) -> Dict[str, Any]:
        """
        V3 Main synthesis function with weighted voting algorithm.
        
        Args:
            evidence_set: List of evidence objects from Verifier.
            hypothesis_set: List of causal rules from Strategist.
            answer_candidates: List of initial answer candidates from Verifier for fallback.
        
        Returns:
            Dict with keys: status, final_answer, final_confidence, confidence_breakdown
        """
        self.logger.info("Synthesizer V3 Engine: Starting weighted voting synthesis")
        
        # DEBUG: Log input data
        self.logger.debug(f"Evidence set received: {evidence_set}")
        self.logger.debug(f"Hypothesis set received: {hypothesis_set}")
        self.logger.debug(f"Answer candidates: {answer_candidates}")
        
        # Step 1: Build evidence map for fast lookup
        evidence_map = self._build_evidence_map(evidence_set)
        self.logger.debug(f"Built evidence map: {evidence_map}")
        
        # Step 2: Initialize voting pool
        voting_pool = {}
        voting_trace = {}
        
        # Step 3: Evaluate all hypotheses with weighted voting
        for i, hypothesis in enumerate(hypothesis_set):
            self.logger.debug(f"Evaluating hypothesis {i}: {hypothesis}")
            
            is_valid = self._validate_hypothesis_format(hypothesis)
            if not is_valid:
                self.logger.warning(f"❌ Hypothesis {i} has invalid format")
                continue
                
            is_triggered, relevant_evidences = self._check_hypothesis_conditions_v3(hypothesis, evidence_set)
            
            if is_triggered:
                # Calculate weighted score
                weighted_score = self._calculate_weighted_score(hypothesis, relevant_evidences)
                candidate_answer = hypothesis['THEN']['final_answer']
                
                # Add to voting pool
                if candidate_answer not in voting_pool:
                    voting_pool[candidate_answer] = 0.0
                    voting_trace[candidate_answer] = []
                
                voting_pool[candidate_answer] += weighted_score
                voting_trace[candidate_answer].append({
                    "hypothesis_id": hypothesis.get('hypothesis_id', f'H_{i}'),
                    "weighted_score": weighted_score,
                    "evidence_support": [e.get('evidence_id') for e in relevant_evidences],
                    "reasoning_description": hypothesis.get('reasoning_description', 'Logical reasoning based on evidence')
                })
                
                self.logger.info(f"✅ Hypothesis {hypothesis.get('hypothesis_id', i)} -> {candidate_answer} (score: {weighted_score:.3f})")
            else:
                self.logger.warning(f"❌ Hypothesis {hypothesis.get('hypothesis_id', i)} conditions NOT met")
        
        # Step 4: Decision making and result formatting
        return self._make_final_decision_v3(voting_pool, voting_trace, answer_candidates)
    
    def _build_evidence_map(self, evidence_set: List[Dict]) -> Dict[str, Any]:
        """Build evidence map with both answer and confidence."""
        evidence_map = {}
        
        for evidence in evidence_set:
            if 'evidence_id' in evidence and 'answer' in evidence:
                evidence_map[evidence['evidence_id']] = {
                    'answer': evidence['answer'],
                    'confidence': evidence.get('confidence', 0.5),
                    'source': evidence.get('source', 'unknown')
                }
            else:
                self.logger.warning(f"Invalid evidence format: {evidence}")
        
        return evidence_map
    
    def _validate_hypothesis_format(self, hypothesis: Dict) -> bool:
        """Validate hypothesis format - same as V2."""
        if not isinstance(hypothesis, dict):
            self.logger.warning("Hypothesis Validation FAIL: Input is not a dictionary.")
            return False

        required_keys = ['hypothesis_id', 'IF', 'THEN']
        for key in required_keys:
            if key not in hypothesis:
                self.logger.warning(f"Hypothesis Validation FAIL: Missing required key '{key}'. Found keys: {list(hypothesis.keys())}")
                return False
        
        if not isinstance(hypothesis['IF'], list):
            self.logger.warning("Hypothesis Validation FAIL: 'IF' clause is not a list.")
            return False
        
        if not isinstance(hypothesis['THEN'], dict):
            self.logger.warning("Hypothesis Validation FAIL: 'THEN' clause is not a dictionary.")
            return False

        if 'final_answer' not in hypothesis['THEN']:
            self.logger.warning("Hypothesis Validation FAIL: Missing 'final_answer' in 'THEN' clause.")
            return False
        
        return True
    
    def _check_hypothesis_conditions_v3(self, hypothesis: Dict, evidence_set: List[Dict]) -> tuple[bool, List[Dict]]:
        """
        Check hypothesis conditions and return relevant evidence objects.
        
        Returns:
            tuple: (conditions_met: bool, relevant_evidences: List[Dict])
        """
        conditions = hypothesis.get('IF', [])
        if not conditions:
            return False, []
        
        evidence_map = {e.get('evidence_id'): e for e in evidence_set}
        relevant_evidences = []
        
        for i, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                self.logger.warning(f"Condition {i} is not a valid dictionary.")
                return False, []

            evidence_id = condition.get('evidence_id') or condition.get('issue_id')
            required_answer_raw = condition.get('answer_is')

            if not evidence_id or required_answer_raw is None:
                self.logger.warning(f"Condition {i} malformed. Missing keys.")
                return False, []

            required_answer = str(required_answer_raw).strip().lower()
            
            if evidence_id not in evidence_map:
                self.logger.debug(f"Evidence '{evidence_id}' not found.")
                return False, []
            
            evidence = evidence_map[evidence_id]
            actual_answer = str(evidence.get('answer', '')).strip().lower()
            
            # Flexible matching logic
            if actual_answer == required_answer or required_answer in actual_answer:
                relevant_evidences.append(evidence)
                continue
            else:
                self.logger.debug(f"Evidence '{evidence_id}' mismatch: got '{actual_answer}', required '{required_answer}'.")
                return False, []
        
        return True, relevant_evidences
    
    def _calculate_weighted_score(self, hypothesis: Dict, relevant_evidences: List[Dict]) -> float:
        """
        Calculate weighted score according to V3 algorithm:
        weighted_score = H.confidence_source * min([E.confidence for E in relevant_evidences])
        """
        hypothesis_confidence = hypothesis.get('confidence_source', 0.5)
        
        if not relevant_evidences:
            return 0.0
        
        evidence_confidences = [e.get('confidence', 0.5) for e in relevant_evidences]
        min_evidence_confidence = min(evidence_confidences)
        
        weighted_score = hypothesis_confidence * min_evidence_confidence
        
        self.logger.debug(f"Weighted score calculation: {hypothesis_confidence} * {min_evidence_confidence} = {weighted_score}")
        return weighted_score
    
    def _make_final_decision_v3(self, voting_pool: Dict[str, float], voting_trace: Dict[str, List], answer_candidates: List[str]) -> Dict[str, Any]:
        """
        Make final decision based on voting pool and create V3 output format.
        """
        if not voting_pool:
            # Fallback mechanism
            self.logger.warning("V3 Fallback: No hypotheses triggered")
            final_answer = answer_candidates[0] if answer_candidates else "Unavailable"
            return {
                "status": "CONCLUSIVE_BY_FALLBACK", 
                "final_answer": final_answer,
                "final_confidence": 0.0,
                "confidence_breakdown": {}
            }
        
        # Find winner
        final_answer = max(voting_pool.keys(), key=lambda k: voting_pool[k])
        final_confidence = voting_pool[final_answer]
        
        # Create confidence breakdown
        confidence_breakdown = {}
        for answer, total_score in voting_pool.items():
            confidence_breakdown[answer] = {
                "total_score": total_score,
                "contributors": voting_trace[answer]
            }
        
        self.logger.info(f"V3 Decision: '{final_answer}' with confidence {final_confidence:.3f}")
        
        return {
            "status": "CONCLUSIVE",
            "final_answer": final_answer, 
            "final_confidence": final_confidence,
            "confidence_breakdown": confidence_breakdown
        }


# ===== V2 SYNTHESIZER ENGINE (EXISTING) =====
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
        
        # DEBUG: Log input data
        self.logger.debug(f"Evidence set received: {evidence_set}")
        self.logger.debug(f"Hypothesis set received: {hypothesis_set}")
        self.logger.debug(f"Answer candidates: {answer_candidates}")
        
        # Step 1: Build evidence map for fast lookup
        evidence_map = self._build_evidence_map(evidence_set)
        self.logger.debug(f"Built evidence map: {evidence_map}")
        
        # Step 2: Evaluate all hypotheses
        triggered_hypotheses = []
        for i, hypothesis in enumerate(hypothesis_set):
            self.logger.debug(f"Evaluating hypothesis {i}: {hypothesis}")
            
            is_valid = self._validate_hypothesis_format(hypothesis)
            self.logger.debug(f"Hypothesis {i} format valid: {is_valid}")
            
            if is_valid:
                conditions_met = self._check_hypothesis_conditions(hypothesis, evidence_map)
                self.logger.debug(f"Hypothesis {i} conditions met: {conditions_met}")
                
                if conditions_met:
                    triggered_hypotheses.append(hypothesis)
                    self.logger.info(f"✅ Hypothesis {hypothesis.get('hypothesis_id', i)} triggered -> {hypothesis['THEN']['final_answer']}")
                else:
                    self.logger.warning(f"❌ Hypothesis {hypothesis.get('hypothesis_id', i)} conditions NOT met")
            else:
                self.logger.warning(f"❌ Hypothesis {i} has invalid format")
        
        self.logger.info(f"Total triggered hypotheses: {len(triggered_hypotheses)}")
        
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
        """Validate that hypothesis has required structure with improved logging."""
        if not isinstance(hypothesis, dict):
            self.logger.warning("Hypothesis Validation FAIL: Input is not a dictionary.")
            return False

        required_keys = ['hypothesis_id', 'IF', 'THEN']
        for key in required_keys:
            if key not in hypothesis:
                self.logger.warning(f"Hypothesis Validation FAIL: Missing required key '{key}'. Found keys: {list(hypothesis.keys())}")
                return False
        
        if not isinstance(hypothesis['IF'], list):
            self.logger.warning("Hypothesis Validation FAIL: 'IF' clause is not a list.")
            return False
        
        if not isinstance(hypothesis['THEN'], dict):
            self.logger.warning("Hypothesis Validation FAIL: 'THEN' clause is not a dictionary.")
            return False

        if 'final_answer' not in hypothesis['THEN']:
            self.logger.warning("Hypothesis Validation FAIL: Missing 'final_answer' in 'THEN' clause.")
            return False
        
        # Validate each condition in IF clause
        for i, condition in enumerate(hypothesis['IF']):
            if not isinstance(condition, dict):
                self.logger.warning(f"Hypothesis Validation FAIL: Condition {i} in 'IF' is not a dictionary.")
                return False
            
            # Flexible check for condition keys
            id_key_found = 'evidence_id' in condition or 'issue_id' in condition
            answer_key_found = 'answer_is' in condition or 'answer' in condition

            if not id_key_found or not answer_key_found:
                self.logger.warning(f"Hypothesis Validation FAIL: Condition {i} is missing ID ('evidence_id' or 'issue_id') or ANSWER ('answer_is' or 'answer'). Found keys: {list(condition.keys())}")
                return False
        
        self.logger.debug(f"Hypothesis {hypothesis.get('hypothesis_id')} format validation PASSED.")
        return True
    
    def _check_hypothesis_conditions(self, hypothesis: Dict, evidence_map: Dict[str, str]) -> bool:
        """
        Check if all conditions in hypothesis IF clause are met.
        This version is more robust against malformed conditions.
        """
        conditions = hypothesis.get('IF', [])
        if not conditions:
            self.logger.warning(f"Hypothesis {hypothesis.get('hypothesis_id')} has no IF conditions to check.")
            return False

        for i, condition in enumerate(conditions):
            # --- Defensive Key Check ---
            # Ensure the condition itself is a dictionary and has the required keys
            if not isinstance(condition, dict):
                self.logger.warning(f"Condition {i} in hypothesis {hypothesis.get('hypothesis_id')} is not a valid dictionary. Skipping.")
                return False # A malformed condition invalidates the hypothesis

            evidence_id = condition.get('evidence_id') or condition.get('issue_id')
            required_answer_raw = condition.get('answer_is')

            if not evidence_id or required_answer_raw is None:
                self.logger.warning(
                    f"Condition {i} in hypothesis {hypothesis.get('hypothesis_id')} is malformed. "
                    f"Missing 'evidence_id'/'issue_id' or 'answer_is'. Keys found: {list(condition.keys())}. Skipping."
                )
                return False # A malformed condition invalidates the hypothesis

            # --- Logic Check ---
            required_answer = str(required_answer_raw).strip().lower()
            
            if evidence_id not in evidence_map:
                self.logger.debug(f"Condition FAIL: Evidence '{evidence_id}' not found in evidence map for hypothesis {hypothesis.get('hypothesis_id')}.")
                return False
            
            actual_answer = str(evidence_map[evidence_id]).strip().lower()
            
            # Flexible matching logic
            if actual_answer == required_answer or required_answer in actual_answer:
                self.logger.debug(f"Condition PASS: Evidence '{evidence_id}' value ('{actual_answer}') matches required ('{required_answer}').")
                continue # Go to the next condition
            else:
                self.logger.debug(f"Condition FAIL: Evidence '{evidence_id}' value mismatch. Got '{actual_answer}', required '{required_answer}'.")
                return False # One failed condition invalidates the entire hypothesis
        
        # If the loop completes without returning False, all conditions were met
        return True
    
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
                "triggered_by_evidence": [c.get('evidence_id') or c.get('issue_id') for c in h['IF']],
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
    Unified wrapper supporting both V2 and V3 SynthesizerEngine.
    Maintains backward compatibility while enabling V3 weighted voting.
    """
    
    def __init__(self, verifier=None, engine_version="v2"):
        """
        Initialize with configurable engine version.
        
        Args:
            verifier: Optional verifier (maintained for compatibility)
            engine_version: Either "v2" (winner-takes-all) or "v3" (weighted voting)
        """
        self.engine_version = engine_version.lower()
        
        if self.engine_version == "v3":
            self.engine = SynthesizerV3Engine()
            self.logger = logging.getLogger(__name__)
            self.logger.info("🚀 SynthesizerAgent initialized with V3 Engine (weighted voting)")
        elif self.engine_version == "v2":
            self.engine = SynthesizerEngine()
            self.logger = logging.getLogger(__name__)
            self.logger.info("🚀 SynthesizerAgent initialized with V2 Engine (winner-takes-all)")
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Unknown engine version '{engine_version}'. Defaulting to V2.")
            self.engine_version = "v2"
            self.engine = SynthesizerEngine()
        
        self.verifier = verifier  # Kept for compatibility but not used in new logic
    
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
        Unified interface for logical synthesis supporting both V2 and V3 engines.
        
        Args:
            evidence_set: Evidence from Verifier.
            hypothesis_set: Logical rules from Strategist.
            answer_candidates: List of initial answer candidates for fallback.
        
        Returns:
            Unified dictionary format:
            - For V2: {"status", "answer", "causal_trace", "engine_version"}
            - For V3: {"status", "answer", "causal_trace", "engine_version", "final_confidence", "confidence_breakdown"}
        """
        if answer_candidates is None:
            answer_candidates = [] # Ensure it's a list
        
        # Get result from engine
        result = self.engine.synthesize(evidence_set, hypothesis_set, answer_candidates)
        
        # Unify output format
        if self.engine_version == "v3":
            # V3 engine returns: {status, final_answer, final_confidence, confidence_breakdown}
            unified_result = {
                "status": result.get("status"),
                "answer": result.get("final_answer"),  # Standardize key name
                "final_confidence": result.get("final_confidence", 0.0),
                "confidence_breakdown": result.get("confidence_breakdown", {}),
                "engine_version": "v3",
                "causal_trace": []  # V3 doesn't use causal_trace format
            }
        else:
            # V2 engine returns: {status, answer, causal_trace}
            unified_result = {
                "status": result.get("status"),
                "answer": result.get("answer"),
                "causal_trace": result.get("causal_trace", []),
                "engine_version": "v2",
                "final_confidence": None,
                "confidence_breakdown": {}
            }
        
        self.logger.info(f"Synthesis completed using {self.engine_version.upper()} engine. Answer: {unified_result['answer']}")
        return unified_result 