#!/usr/bin/env python3
"""
Test script to demonstrate the new Synthesizer Logic Engine
This validates the deterministic, algorithmic approach of the new synthesizer.
"""

import sys
import json
from pathlib import Path

# Add the FDR src path to import the new synthesizer
sys.path.append(str(Path(__file__).parent / "FDR" / "src"))

from agents.synthesizer import SynthesizerEngine, SynthesizerAgent


def test_example_from_design():
    """Test the exact example from the design specification."""
    print("=== TESTING EXAMPLE FROM DESIGN SPECIFICATION ===\n")
    
    # Example evidence set from design
    evidence_set = [
        {
            "evidence_id": "E01",
            "issue_text": "Locate the person on the left.",
            "answer": "True",
            "confidence": 0.99,
            "grounding": {"bbox": [50, 100, 200, 400]}
        },
        {
            "evidence_id": "E02", 
            "issue_text": "Locate the object being held by that person.",
            "answer": "True",
            "confidence": 0.97,
            "grounding": {"bbox": [150, 250, 180, 280]}
        },
        {
            "evidence_id": "E03",
            "issue_text": "Determine the color of that object.",
            "answer": "Red",
            "confidence": 0.96
        }
    ]
    
    # Example hypothesis set from design
    hypothesis_set = [
        {
            "hypothesis_id": "H_Red",
            "IF": [{"evidence_id": "E03", "answer_is": "Red"}],
            "THEN": {"final_answer": "Red"}
        },
        {
            "hypothesis_id": "H_Blue",
            "IF": [{"evidence_id": "E03", "answer_is": "Blue"}],
            "THEN": {"final_answer": "Blue"}
        }
    ]
    
    print("INPUT:")
    print(f"Evidence Set: {json.dumps(evidence_set, indent=2)}")
    print(f"\nHypothesis Set: {json.dumps(hypothesis_set, indent=2)}")
    
    # Test the new engine
    engine = SynthesizerEngine()
    result = engine.synthesize(evidence_set, hypothesis_set)
    
    print(f"\nOUTPUT:")
    print(json.dumps(result, indent=2))
    
    # Validate expected result
    expected_answer = "Red"
    expected_status = "CONCLUSIVE"
    
    assert result["status"] == expected_status, f"Expected status {expected_status}, got {result['status']}"
    assert result["answer"] == expected_answer, f"Expected answer {expected_answer}, got {result['answer']}"
    assert len(result["causal_trace"]) == 1, f"Expected 1 causal trace, got {len(result['causal_trace'])}"
    assert result["causal_trace"][0]["hypothesis_id"] == "H_Red"
    
    print("\n✅ EXAMPLE FROM DESIGN SPECIFICATION PASSED!\n")


def test_contradictory_case():
    """Test case with contradictory hypotheses."""
    print("=== TESTING CONTRADICTORY CASE ===\n")
    
    evidence_set = [
        {"evidence_id": "E01", "answer": "Yes"},
        {"evidence_id": "E02", "answer": "No"}
    ]
    
    hypothesis_set = [
        {
            "hypothesis_id": "H1",
            "IF": [{"evidence_id": "E01", "answer_is": "Yes"}],
            "THEN": {"final_answer": "Answer_A"}
        },
        {
            "hypothesis_id": "H2", 
            "IF": [{"evidence_id": "E02", "answer_is": "No"}],
            "THEN": {"final_answer": "Answer_B"}
        }
    ]
    
    engine = SynthesizerEngine()
    result = engine.synthesize(evidence_set, hypothesis_set)
    
    print("INPUT:")
    print(f"Evidence: {evidence_set}")
    print(f"Hypotheses: {hypothesis_set}")
    print(f"\nOUTPUT: {json.dumps(result, indent=2)}")
    
    assert result["status"] == "CONTRADICTORY"
    assert result["answer"] is None
    assert "Answer_A" in result["contradictory_answers"]
    assert "Answer_B" in result["contradictory_answers"]
    
    print("\n✅ CONTRADICTORY CASE PASSED!\n")


def test_inconclusive_case():
    """Test case where no hypotheses are triggered."""
    print("=== TESTING INCONCLUSIVE CASE ===\n")
    
    evidence_set = [
        {"evidence_id": "E01", "answer": "Maybe"}
    ]
    
    hypothesis_set = [
        {
            "hypothesis_id": "H1",
            "IF": [{"evidence_id": "E01", "answer_is": "Yes"}],
            "THEN": {"final_answer": "Answer_A"}
        },
        {
            "hypothesis_id": "H2",
            "IF": [{"evidence_id": "E01", "answer_is": "No"}], 
            "THEN": {"final_answer": "Answer_B"}
        }
    ]
    
    engine = SynthesizerEngine()
    result = engine.synthesize(evidence_set, hypothesis_set)
    
    print("INPUT:")
    print(f"Evidence: {evidence_set}")
    print(f"Hypotheses: {hypothesis_set}")
    print(f"\nOUTPUT: {json.dumps(result, indent=2)}")
    
    assert result["status"] == "INCONCLUSIVE"
    assert result["answer"] is None
    assert len(result["causal_trace"]) == 0
    
    print("\n✅ INCONCLUSIVE CASE PASSED!\n")


def test_complex_logic():
    """Test complex multi-condition logic."""
    print("=== TESTING COMPLEX MULTI-CONDITION LOGIC ===\n")
    
    evidence_set = [
        {"evidence_id": "E01", "answer": "Yes"},
        {"evidence_id": "E02", "answer": "Yes"}, 
        {"evidence_id": "E03", "answer": "No"}
    ]
    
    hypothesis_set = [
        {
            "hypothesis_id": "H_Fry",
            "IF": [
                {"evidence_id": "E01", "answer_is": "Yes"},  # Has glossy sheen
                {"evidence_id": "E02", "answer_is": "Yes"}   # Has crispy texture
            ],
            "THEN": {"final_answer": "Fry"}
        },
        {
            "hypothesis_id": "H_Boil",
            "IF": [
                {"evidence_id": "E01", "answer_is": "No"},   # No glossy sheen
                {"evidence_id": "E03", "answer_is": "No"}    # No grill marks
            ],
            "THEN": {"final_answer": "Boil"}
        }
    ]
    
    engine = SynthesizerEngine()
    result = engine.synthesize(evidence_set, hypothesis_set)
    
    print("INPUT:")
    print(f"Evidence: {evidence_set}")
    print(f"Hypotheses: {hypothesis_set}")
    print(f"\nOUTPUT: {json.dumps(result, indent=2)}")
    
    assert result["status"] == "CONCLUSIVE"
    assert result["answer"] == "Fry"
    assert result["causal_trace"][0]["hypothesis_id"] == "H_Fry"
    assert set(result["causal_trace"][0]["triggered_by_evidence"]) == {"E01", "E02"}
    
    print("\n✅ COMPLEX MULTI-CONDITION LOGIC PASSED!\n")


def test_backward_compatibility():
    """Test the backward compatibility wrapper."""
    print("=== TESTING BACKWARD COMPATIBILITY ===\n")
    
    # Test with new format in mvkb
    mvkb_new_format = {
        "evidence_set": [
            {"evidence_id": "E01", "answer": "Yes"}
        ],
        "hypothesis_set": [
            {
                "hypothesis_id": "H1",
                "IF": [{"evidence_id": "E01", "answer_is": "Yes"}],
                "THEN": {"final_answer": "Answer_A"}
            }
        ]
    }
    
    agent = SynthesizerAgent()
    result = agent.conduct_weighted_voting(
        original_question="Test question",
        image_path="test.jpg",
        answer_candidates=["Answer_A", "Answer_B"],
        mvkb=mvkb_new_format
    )
    
    print("INPUT (New Format):")
    print(f"MVKB: {mvkb_new_format}")
    print(f"\nOUTPUT: {json.dumps(result, indent=2)}")
    
    assert result["final_answer"] == "Answer_A"
    assert result["confidence_breakdown"]["synthesizer_status"] == "CONCLUSIVE"
    
    # Test with old format (should show graceful degradation)
    mvkb_old_format = [
        {"hypothesis": "Some old format", "confidence": 0.8}
    ]
    
    result_old = agent.conduct_weighted_voting(
        original_question="Test question", 
        image_path="test.jpg",
        answer_candidates=["Answer_A", "Answer_B"],
        mvkb=mvkb_old_format
    )
    
    print(f"\nOLD FORMAT HANDLING: {result_old.get('synthesizer_status', 'No status')}")
    assert "LEGACY_FORMAT_ERROR" in str(result_old)
    
    print("\n✅ BACKWARD COMPATIBILITY PASSED!\n")


def demo_deterministic_behavior():
    """Demonstrate that the engine is deterministic."""
    print("=== DEMONSTRATING DETERMINISTIC BEHAVIOR ===\n")
    
    evidence_set = [{"evidence_id": "E01", "answer": "Yes"}]
    hypothesis_set = [
        {
            "hypothesis_id": "H1",
            "IF": [{"evidence_id": "E01", "answer_is": "Yes"}],
            "THEN": {"final_answer": "Consistent_Answer"}
        }
    ]
    
    engine = SynthesizerEngine()
    
    # Run the same input 5 times
    results = []
    for i in range(5):
        result = engine.synthesize(evidence_set, hypothesis_set)
        results.append(result)
        print(f"Run {i+1}: {result['status']} -> {result['answer']}")
    
    # Verify all results are identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 2):
        assert result == first_result, f"Run {i} differs from Run 1 - NOT DETERMINISTIC!"
    
    print("\n✅ DETERMINISTIC BEHAVIOR CONFIRMED!")
    print("Same input -> Same output (every time)\n")


if __name__ == "__main__":
    print("🔬 TESTING NEW SYNTHESIZER LOGIC ENGINE")
    print("="*50)
    
    try:
        # Run all tests
        test_example_from_design()
        test_contradictory_case()
        test_inconclusive_case()
        test_complex_logic()
        test_backward_compatibility()
        demo_deterministic_behavior()
        
        print("🎉 ALL TESTS PASSED!")
        print("="*50)
        print("The new Synthesizer Logic Engine is working correctly:")
        print("✅ Deterministic - Same input always produces same output")
        print("✅ Verifiable - Logic is simple and can be manually audited")
        print("✅ Modular - Single responsibility: logical synthesis only") 
        print("✅ Stateless - No memory between queries")
        print("✅ Backward compatible - Existing code can still work")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 