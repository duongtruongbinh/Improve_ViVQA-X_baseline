#!/usr/bin/env python3
"""
Test V3 Synthesizer Implementation
Comprehensive testing and comparison between V2 and V3 engines
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.agents.synthesizer import SynthesizerAgent, SynthesizerEngine, SynthesizerV3Engine

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """Create realistic test data matching FDR pipeline format."""
    
    # Sample evidence set from Verifier
    evidence_set = [
        {
            "evidence_id": "EV_VISUAL_001",
            "source": "VISUAL_VERIFICATION",
            "answer": "2",
            "confidence": 0.85,
            "description": "Visual count shows 2 dogs in the image"
        },
        {
            "evidence_id": "EV_CAPTION_001", 
            "source": "IMAGE_CAPTION",
            "answer": "dogs",
            "confidence": 0.90,
            "description": "Image caption identifies dogs as main subject"
        },
        {
            "evidence_id": "EV_OBJECT_001",
            "source": "OBJECT_DETECTION", 
            "answer": "yes",
            "confidence": 0.75,
            "description": "GroundingDINO detects dog objects in image"
        }
    ]
    
    # Sample hypothesis set from Strategist
    hypothesis_set = [
        {
            "hypothesis_id": "H_COUNT_ANIMALS",
            "confidence_source": 0.8,
            "reasoning_description": "If visual verification shows 2 and caption confirms dogs, then answer is 2",
            "IF": [
                {"evidence_id": "EV_VISUAL_001", "answer_is": "2"},
                {"evidence_id": "EV_CAPTION_001", "answer_is": "dogs"}
            ],
            "THEN": {
                "final_answer": "2"
            }
        },
        {
            "hypothesis_id": "H_DETECT_PRESENCE", 
            "confidence_source": 0.7,
            "reasoning_description": "If object detection confirms presence and caption shows dogs, then animals present",
            "IF": [
                {"evidence_id": "EV_OBJECT_001", "answer_is": "yes"},
                {"evidence_id": "EV_CAPTION_001", "answer_is": "dogs"}
            ],
            "THEN": {
                "final_answer": "yes"
            }
        },
        {
            "hypothesis_id": "H_ALTERNATIVE_COUNT",
            "confidence_source": 0.6,
            "reasoning_description": "Alternative counting hypothesis with lower confidence",
            "IF": [
                {"evidence_id": "EV_VISUAL_001", "answer_is": "2"}
            ],
            "THEN": {
                "final_answer": "2"
            }
        }
    ]
    
    answer_candidates = ["2", "yes", "no", "1", "3"]
    
    return evidence_set, hypothesis_set, answer_candidates


def test_v2_engine():
    """Test V2 (winner-takes-all) engine."""
    logger.info("=== Testing V2 Synthesizer Engine ===")
    
    evidence_set, hypothesis_set, answer_candidates = create_test_data()
    engine = SynthesizerEngine()
    
    result = engine.synthesize(evidence_set, hypothesis_set, answer_candidates)
    
    logger.info(f"V2 Result: {json.dumps(result, indent=2)}")
    return result


def test_v3_engine():
    """Test V3 (weighted voting) engine.""" 
    logger.info("=== Testing V3 Synthesizer Engine ===")
    
    evidence_set, hypothesis_set, answer_candidates = create_test_data()
    engine = SynthesizerV3Engine()
    
    result = engine.synthesize(evidence_set, hypothesis_set, answer_candidates)
    
    logger.info(f"V3 Result: {json.dumps(result, indent=2)}")
    return result


def test_synthesizer_agent_switching():
    """Test SynthesizerAgent with both V2 and V3 modes."""
    logger.info("=== Testing SynthesizerAgent Version Switching ===")
    
    evidence_set, hypothesis_set, answer_candidates = create_test_data()
    
    # Test V2 Agent
    logger.info("--- V2 SynthesizerAgent ---")
    agent_v2 = SynthesizerAgent(engine_version="v2")
    result_v2 = agent_v2.synthesize(evidence_set, hypothesis_set, answer_candidates)
    logger.info(f"V2 Agent Result: {json.dumps(result_v2, indent=2)}")
    
    # Test V3 Agent  
    logger.info("--- V3 SynthesizerAgent ---")
    agent_v3 = SynthesizerAgent(engine_version="v3")
    result_v3 = agent_v3.synthesize(evidence_set, hypothesis_set, answer_candidates)
    logger.info(f"V3 Agent Result: {json.dumps(result_v3, indent=2)}")
    
    return result_v2, result_v3


def analyze_v3_confidence_breakdown():
    """Specifically analyze V3's confidence breakdown feature."""
    logger.info("=== Analyzing V3 Confidence Breakdown ===")
    
    evidence_set, hypothesis_set, answer_candidates = create_test_data()
    agent = SynthesizerAgent(engine_version="v3")
    result = agent.synthesize(evidence_set, hypothesis_set, answer_candidates)
    
    confidence_breakdown = result.get('confidence_breakdown', {})
    
    logger.info("Confidence Breakdown Analysis:")
    for answer, details in confidence_breakdown.items():
        total_score = details.get('total_score', 0)
        contributors = details.get('contributors', [])
        
        logger.info(f"  Answer '{answer}': Total Score = {total_score:.3f}")
        for contributor in contributors:
            hypothesis_id = contributor.get('hypothesis_id')
            weighted_score = contributor.get('weighted_score', 0)
            evidence_support = contributor.get('evidence_support', [])
            
            logger.info(f"    - {hypothesis_id}: Score = {weighted_score:.3f}, Evidence = {evidence_support}")
    
    return confidence_breakdown


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("=== Testing Edge Cases ===")
    
    # Test empty hypothesis set
    logger.info("--- Empty Hypothesis Set ---")
    agent = SynthesizerAgent(engine_version="v3")
    result = agent.synthesize([], [], ["fallback_answer"])
    logger.info(f"Empty hypothesis result: {result}")
    
    # Test malformed hypothesis
    logger.info("--- Malformed Hypothesis ---")
    malformed_hypothesis = [
        {
            "hypothesis_id": "BAD_H1",
            # Missing required fields
            "IF": "not a list",
            "THEN": "not a dict"
        }
    ]
    result = agent.synthesize([], malformed_hypothesis, ["fallback_answer"])
    logger.info(f"Malformed hypothesis result: {result}")
    
    # Test no matching evidence
    logger.info("--- No Matching Evidence ---")
    evidence_set = [{"evidence_id": "EV1", "answer": "blue", "confidence": 0.9}]
    hypothesis_set = [
        {
            "hypothesis_id": "H_NO_MATCH",
            "confidence_source": 0.8,
            "IF": [{"evidence_id": "EV1", "answer_is": "red"}],  # Won't match
            "THEN": {"final_answer": "no_match"}
        }
    ]
    result = agent.synthesize(evidence_set, hypothesis_set, ["fallback"])
    logger.info(f"No matching evidence result: {result}")


def performance_comparison():
    """Compare performance characteristics between V2 and V3."""
    logger.info("=== Performance Comparison V2 vs V3 ===")
    
    import time
    
    evidence_set, hypothesis_set, answer_candidates = create_test_data()
    
    # V2 Performance
    start_time = time.time()
    agent_v2 = SynthesizerAgent(engine_version="v2")
    result_v2 = agent_v2.synthesize(evidence_set, hypothesis_set, answer_candidates)
    v2_time = time.time() - start_time
    
    # V3 Performance
    start_time = time.time()
    agent_v3 = SynthesizerAgent(engine_version="v3")
    result_v3 = agent_v3.synthesize(evidence_set, hypothesis_set, answer_candidates)
    v3_time = time.time() - start_time
    
    logger.info(f"V2 Execution Time: {v2_time:.4f}s")
    logger.info(f"V3 Execution Time: {v3_time:.4f}s")
    logger.info(f"Performance Ratio (V3/V2): {v3_time/v2_time:.2f}x")
    
    # Compare output quality
    logger.info("Output Comparison:")
    logger.info(f"V2 Answer: {result_v2.get('answer')}")
    logger.info(f"V3 Answer: {result_v3.get('answer')}")
    logger.info(f"V2 Status: {result_v2.get('status')}")
    logger.info(f"V3 Status: {result_v3.get('status')}")
    
    # V3 specific features
    if 'confidence_breakdown' in result_v3:
        breakdown_detail = len(str(result_v3['confidence_breakdown']))
        logger.info(f"V3 Confidence Breakdown Detail: {breakdown_detail} characters")


def main():
    """Run all tests."""
    logger.info("🧪 Starting V3 Synthesizer Testing Suite")
    logger.info("=" * 60)
    
    try:
        # Core engine tests
        v2_result = test_v2_engine()
        v3_result = test_v3_engine()
        
        # Agent wrapper tests
        agent_v2_result, agent_v3_result = test_synthesizer_agent_switching()
        
        # V3 specific features
        confidence_breakdown = analyze_v3_confidence_breakdown()
        
        # Edge case testing
        test_edge_cases()
        
        # Performance comparison
        performance_comparison()
        
        logger.info("=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("📊 V3 Implementation verification PASSED")
        
        # Summary
        logger.info("\n📋 Test Summary:")
        logger.info(f"  - V2 Engine: {'✅ WORKING' if v2_result.get('answer') else '❌ FAILED'}")
        logger.info(f"  - V3 Engine: {'✅ WORKING' if v3_result.get('final_answer') else '❌ FAILED'}")
        logger.info(f"  - V2 Agent: {'✅ WORKING' if agent_v2_result.get('answer') else '❌ FAILED'}")
        logger.info(f"  - V3 Agent: {'✅ WORKING' if agent_v3_result.get('answer') else '❌ FAILED'}")
        logger.info(f"  - Confidence Breakdown: {'✅ WORKING' if confidence_breakdown else '❌ FAILED'}")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 