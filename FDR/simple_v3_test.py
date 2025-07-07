#!/usr/bin/env python3
"""
Simple V3 Test - Quick verification of V3 vs V2 functionality
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.agents.synthesizer import SynthesizerAgent


def create_sample_data():
    """Create sample data for testing."""
    
    evidence_set = [
        {
            "evidence_id": "EV1",
            "source": "VISUAL",
            "answer": "2",
            "confidence": 0.85
        },
        {
            "evidence_id": "EV2", 
            "source": "CAPTION",
            "answer": "dogs",
            "confidence": 0.90
        }
    ]
    
    hypothesis_set = [
        {
            "hypothesis_id": "H1",
            "confidence_source": 0.8,
            "reasoning_description": "Count hypothesis",
            "IF": [
                {"evidence_id": "EV1", "answer_is": "2"},
                {"evidence_id": "EV2", "answer_is": "dogs"}
            ],
            "THEN": {
                "final_answer": "2"
            }
        },
        {
            "hypothesis_id": "H2",
            "confidence_source": 0.7,
            "reasoning_description": "Alternative hypothesis", 
            "IF": [
                {"evidence_id": "EV1", "answer_is": "2"}
            ],
            "THEN": {
                "final_answer": "2"
            }
        }
    ]
    
    answer_candidates = ["1", "2", "3"]
    
    return evidence_set, hypothesis_set, answer_candidates


def main():
    print("🧪 Simple V3 Test")
    print("=" * 40)
    
    evidence_set, hypothesis_set, answer_candidates = create_sample_data()
    
    print("📊 Test Data:")
    print(f"  Evidence: {len(evidence_set)} items")
    print(f"  Hypotheses: {len(hypothesis_set)} items")
    print(f"  Candidates: {answer_candidates}")
    print()
    
    # Test V2
    print("🔹 Testing V2 Engine:")
    try:
        agent_v2 = SynthesizerAgent(engine_version="v2")
        result_v2 = agent_v2.synthesize(evidence_set, hypothesis_set, answer_candidates)
        print(f"  Answer: {result_v2.get('answer')}")
        print(f"  Status: {result_v2.get('status')}")
        print(f"  Engine: {result_v2.get('engine_version')}")
        print(f"  Causal Trace: {len(result_v2.get('causal_trace', []))} entries")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()
    
    # Test V3
    print("🔹 Testing V3 Engine:")
    try:
        agent_v3 = SynthesizerAgent(engine_version="v3")
        result_v3 = agent_v3.synthesize(evidence_set, hypothesis_set, answer_candidates)
        print(f"  Answer: {result_v3.get('answer')}")
        print(f"  Status: {result_v3.get('status')}")
        print(f"  Engine: {result_v3.get('engine_version')}")
        print(f"  Confidence: {result_v3.get('final_confidence')}")
        
        # Show confidence breakdown
        breakdown = result_v3.get('confidence_breakdown', {})
        print(f"  Confidence Breakdown:")
        for answer, details in breakdown.items():
            total_score = details.get('total_score', 0)
            contributors = details.get('contributors', [])
            print(f"    '{answer}': Score={total_score:.3f}, Contributors={len(contributors)}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("✅ Test completed!")


if __name__ == "__main__":
    main() 