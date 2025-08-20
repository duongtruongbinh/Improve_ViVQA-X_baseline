#!/usr/bin/env python3
# FDR/test_auto_save.py - Test script for Auto Save Pipeline
import os
import sys
import json
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

def test_auto_save_pipeline():
    """Test the auto save pipeline with minimal samples"""
    print("🧪 Testing Auto Save Pipeline...")
    
    try:
        from src.auto_save_pipeline import run_auto_save_pipeline
        
        # Test with minimal samples
        summary = run_auto_save_pipeline(
            use_vllm=False,  # Use OpenAI for testing
            max_correct=1,
            max_wrong=1,
            output_base_dir="test_output"
        )
        
        if summary:
            print("✅ Test completed successfully!")
            print(f"📊 Results: {summary}")
            
            # Check if files were created
            test_dir = "test_output"
            if os.path.exists(test_dir):
                print(f"📁 Test output directory created: {test_dir}")
                
                # Check correct samples
                correct_dir = os.path.join(test_dir, "correct")
                if os.path.exists(correct_dir):
                    correct_samples = len([d for d in os.listdir(correct_dir) if d.startswith("sample_")])
                    print(f"✅ Correct samples: {correct_samples}")
                
                # Check wrong samples
                wrong_dir = os.path.join(test_dir, "wrong")
                if os.path.exists(wrong_dir):
                    wrong_samples = len([d for d in os.listdir(wrong_dir) if d.startswith("sample_")])
                    print(f"❌ Wrong samples: {wrong_samples}")
                
                # Check summary file
                summary_file = os.path.join(test_dir, "summary.json")
                if os.path.exists(summary_file):
                    print(f"📋 Summary file created: {summary_file}")
                    
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    print(f"📊 Summary data: {summary_data}")
            
            return True
        else:
            print("❌ Test failed: No summary returned")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_output():
    """Clean up test output directory"""
    test_dir = "test_output"
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    print("🚀 Starting Auto Save Pipeline Test")
    
    # Run test
    success = test_auto_save_pipeline()
    
    if success:
        print("\n🎉 Test completed successfully!")
        
        # Ask if user wants to keep test output
        response = input("\nDo you want to keep the test output? (y/n): ").lower()
        if response != 'y':
            cleanup_test_output()
    else:
        print("\n❌ Test failed!")
        cleanup_test_output() 