#!/usr/bin/env python3
# FDR/quick_run.py - Quick run script for Auto Save Pipeline
import os
import sys
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

def quick_run():
    """Quick run with default settings: 3 correct, 3 wrong samples"""
    print("🚀 Quick Run - Auto Save FDR Pipeline")
    print("📊 Target: 3 correct, 3 wrong samples")
    print("🔧 Backend: vLLM (default)")
    print("📁 Output: auto_output")
    print("-" * 50)
    
    try:
        from src.auto_save_pipeline import run_auto_save_pipeline
        
        # Run with default settings
        summary = run_auto_save_pipeline(
            use_vllm=True,
            max_correct=3,
            max_wrong=3,
            output_base_dir="auto_output"
        )
        
        if summary:
            print("\n🎉 Quick run completed successfully!")
            print(f"📊 Results:")
            print(f"  - Total processed: {summary['total_processed']}")
            print(f"  - Correct samples: {summary['correct_samples']}")
            print(f"  - Wrong samples: {summary['wrong_samples']}")
            print(f"  - Target reached: {summary['target_reached']}")
            print(f"📁 Output directory: auto_output")
            
            # Show next steps
            print(f"\n📋 Next steps:")
            print(f"  - View results: python view_results.py")
            print(f"  - List samples: python view_results.py --list")
            print(f"  - Show sample: python view_results.py --show correct:1")
            print(f"  - Show images: python view_results.py --images correct:1")
            print(f"  - Analyze: python view_results.py --analyze")
            
            return summary
        else:
            print("❌ Quick run failed: No summary returned")
            return None
            
    except Exception as e:
        print(f"❌ Quick run failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    quick_run() 