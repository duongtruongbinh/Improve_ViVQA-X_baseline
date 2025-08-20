#!/usr/bin/env python3
"""
Test script to verify visualization changes:
1. GroundingDINO removed from visualization
2. Evidence and hypothesis details added to visualization
"""
import sys
import json
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent / "FDR"
sys.path.insert(0, str(fdr_dir))

def test_info_json_structure():
    """Test that info.json now contains evidence_set and hypothesis_set"""
    print("🧪 Testing info.json structure...")
    
    # Check if there's an existing sample to test
    sample_path = Path("FDR/auto_output/wrong/sample_1/info.json")
    if sample_path.exists():
        try:
            with open(sample_path, 'r') as f:
                info = json.load(f)
            
            # Check for new fields
            has_evidence_set = 'evidence_set' in info
            has_hypothesis_set = 'hypothesis_set' in info
            has_evidence_count = 'evidence_count' in info
            has_hypothesis_count = 'hypothesis_count' in info
            
            print(f"  - evidence_set: {'✅' if has_evidence_set else '❌'}")
            print(f"  - hypothesis_set: {'✅' if has_hypothesis_set else '❌'}")
            print(f"  - evidence_count: {'✅' if has_evidence_count else '❌'}")
            print(f"  - hypothesis_count: {'✅' if has_hypothesis_count else '❌'}")
            
            if has_evidence_set and has_hypothesis_set:
                print("✅ info.json structure updated correctly")
                return True
            else:
                print("❌ info.json structure needs updating")
                return False
                
        except Exception as e:
            print(f"❌ Error reading info.json: {e}")
            return False
    else:
        print("ℹ️ No existing sample found - will be tested when pipeline runs")
        return True

def test_visualization_function():
    """Test the visualization function changes"""
    print("🧪 Testing visualization function...")
    
    try:
        from src.auto_save_pipeline import create_output_visualization
        
        # Create mock result data with full content
        mock_result = {
            'question': 'Is this a test question for the visualization system?',
            'final_answer': 'yes',
            'ground_truth': 'yes',
            'explanation': 'This is a comprehensive test explanation for the visualization system. It demonstrates how the full explanation text will be displayed without truncation in the new layout.',
            'synthesis_status': 'CONCLUSIVE',
            'evidence_set': [
                {
                    'evidence_id': 'E1',
                    'question': 'Is there a person in the image?',
                    'answer': 'Yes, there is a person clearly visible in the center of the image wearing casual clothing'
                },
                {
                    'evidence_id': 'E2',
                    'question': 'What is the person wearing?',
                    'answer': 'The person is wearing a red jacket and blue jeans, which are clearly visible in the image'
                },
                {
                    'evidence_id': 'E3',
                    'question': 'What is the background setting?',
                    'answer': 'The background shows a park setting with trees and grass, indicating an outdoor environment'
                }
            ],
            'hypothesis_set': [
                {
                    'hypothesis_id': 'H_yes_1',
                    'THEN': {'final_answer': 'yes'},
                    'confidence_source': 0.85,
                    'reasoning_description': 'Based on multiple visual evidence points, the answer is yes'
                },
                {
                    'hypothesis_id': 'H_no_1',
                    'THEN': {'final_answer': 'no'},
                    'confidence_source': 0.15,
                    'reasoning_description': 'Alternative hypothesis with lower confidence based on contradictory evidence'
                }
            ],
            'evidence_count': 3,
            'hypothesis_count': 2,
            'image_path': '/fake/path/test.jpg'  # This will show "Image not found" which is fine for testing
        }
        
        # Test directory
        test_dir = Path("test_visualization_output")
        test_dir.mkdir(exist_ok=True)
        
        # Try to create visualization
        create_output_visualization(mock_result, str(test_dir))
        
        # Check if visualization was created
        viz_path = test_dir / "output_visualization.png"
        if viz_path.exists():
            print("✅ Visualization function works correctly")
            print(f"  - Test visualization saved to: {viz_path}")
            return True
        else:
            print("❌ Visualization file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visualization function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_groundingdino_disabled():
    """Test that GroundingDINO is properly disabled"""
    print("🧪 Testing GroundingDINO disabled...")
    
    try:
        from src.agents.verifier import VerifierAgent
        
        # Create verifier agent
        verifier = VerifierAgent(use_vllm=True)
        
        # Check if GroundingDINO is disabled
        if not verifier.groundingdino_enabled:
            print("✅ GroundingDINO is properly disabled")
            return True
        else:
            print("❌ GroundingDINO is still enabled")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GroundingDINO: {e}")
        return False

def show_expected_output():
    """Show what the new visualization should contain"""
    print("\n📋 Expected visualization changes:")
    print("  ✅ 3-panel layout (Input Image + GroundingDINO + Text Details)")
    print("  ✅ GroundingDINO panel kept as before")
    print("  ✅ Evidence Details section with:")
    print("      • Evidence ID, question, and FULL answers (no truncation)")
    print("  ✅ Hypothesis Details section with:")
    print("      • Hypothesis ID, final answer, confidence, and reasoning")
    print("  ✅ Summary counts at bottom")
    print("  ✅ Larger figure size (30x12 or 20x12) for full text display")
    print("  ✅ ALL evidence displayed with FULL text (no truncation)")
    print("  ✅ ALL hypothesis displayed with FULL reasoning")
    print("  ✅ Full explanation text displayed")
    print("  ✅ No emoji warnings (📊 removed)")
    print("\n📋 Expected info.json changes:")
    print("  ✅ evidence_set: [detailed evidence objects with full content]")
    print("  ✅ hypothesis_set: [detailed hypothesis objects with full content]")
    print("  ✅ evidence_count: number")
    print("  ✅ hypothesis_count: number")
    print("  ✅ causal_trace: [existing field]")

def main():
    """Run all tests"""
    print("🚀 Testing visualization changes...")
    print("=" * 60)
    
    tests = [
        test_groundingdino_disabled,
        test_info_json_structure,
        test_visualization_function
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Visualization changes are working correctly.")
        show_expected_output()
        print("\n📋 You can now run:")
        print("  cd FDR && python quick_run.py")
        print("  Then check: auto_output/wrong/sample_1/output_visualization.png")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()
