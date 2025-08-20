#!/usr/bin/env python3
"""
Test script to verify GroundingDINO is disabled and evidence/hypothesis display works
"""
import sys
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent / "FDR"
sys.path.insert(0, str(fdr_dir))

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

def test_config_updated():
    """Test that config file has GroundingDINO disabled"""
    print("🧪 Testing config updated...")
    
    try:
        import yaml
        
        config_path = Path("FDR/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        groundingdino_enabled = config.get('agents_config', {}).get('verifier', {}).get('enable_groundingdino', True)
        
        if not groundingdino_enabled:
            print("✅ Config has GroundingDINO disabled")
            return True
        else:
            print("❌ Config still has GroundingDINO enabled")
            return False
            
    except Exception as e:
        print(f"❌ Error testing config: {e}")
        return False

def test_evidence_hypothesis_structure():
    """Test that evidence and hypothesis structure is available"""
    print("🧪 Testing evidence/hypothesis structure...")
    
    try:
        # Test the summary structure that would be returned
        mock_summary = {
            'sample_details': [
                {
                    'question_id': 'test_001',
                    'evidence_count': 3,
                    'hypothesis_count': 2,
                    'evidence_set': [
                        {'evidence_id': 'E1', 'answer': 'Yes, there is a person in the image'},
                        {'evidence_id': 'E2', 'answer': 'The person is wearing red clothes'}
                    ],
                    'hypothesis_set': [
                        {'hypothesis_id': 'H1', 'THEN': {'final_answer': 'yes'}},
                        {'hypothesis_id': 'H2', 'THEN': {'final_answer': 'no'}}
                    ]
                }
            ]
        }
        
        # Test the display function
        from quick_run import _show_evidence_hypothesis_summary
        print("Testing evidence/hypothesis display:")
        _show_evidence_hypothesis_summary(mock_summary)
        
        print("✅ Evidence/hypothesis structure works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing evidence/hypothesis structure: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing FDR changes...")
    print("=" * 50)
    
    tests = [
        test_config_updated,
        test_groundingdino_disabled,
        test_evidence_hypothesis_structure
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Changes are working correctly.")
        print("\n📋 You can now run:")
        print("  cd FDR && python quick_run.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()
