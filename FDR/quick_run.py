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
    #print("� GroundingDINO: Disabled")
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

            # Show evidence and hypothesis details
            _show_evidence_hypothesis_summary(summary)

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

def _show_evidence_hypothesis_summary(summary):
    """Show evidence and hypothesis details from the summary"""
    print(f"\n🔍 Evidence & Hypothesis Summary:")

    # Get sample details if available
    sample_details = summary.get('sample_details', [])
    if not sample_details:
        print("  - No detailed sample information available")
        return

    total_evidence = 0
    total_hypothesis = 0

    for i, sample in enumerate(sample_details[:3]):  # Show first 3 samples
        evidence_count = sample.get('evidence_count', 0)
        hypothesis_count = sample.get('hypothesis_count', 0)

        total_evidence += evidence_count
        total_hypothesis += hypothesis_count

        print(f"  Sample {i+1}:")
        print(f"    - Evidence: {evidence_count} items")
        print(f"    - Hypothesis: {hypothesis_count} items")

        # Show evidence details if available
        evidence_set = sample.get('evidence_set', [])
        if evidence_set:
            print(f"    - Evidence details:")
            for j, evidence in enumerate(evidence_set[:2]):  # Show first 2 evidence
                evidence_id = evidence.get('evidence_id', f'E{j+1}')
                answer = evidence.get('answer', 'N/A')[:50]  # Truncate long answers
                print(f"      • {evidence_id}: {answer}...")

        # Show hypothesis details if available
        hypothesis_set = sample.get('hypothesis_set', [])
        if hypothesis_set:
            print(f"    - Hypothesis details:")
            for j, hypothesis in enumerate(hypothesis_set[:2]):  # Show first 2 hypothesis
                hyp_id = hypothesis.get('hypothesis_id', f'H{j+1}')
                final_answer = hypothesis.get('THEN', {}).get('final_answer', 'N/A')
                print(f"      • {hyp_id}: → {final_answer}")

        if i < len(sample_details) - 1:
            print()

    # Show totals
    avg_evidence = total_evidence / len(sample_details) if sample_details else 0
    avg_hypothesis = total_hypothesis / len(sample_details) if sample_details else 0

    print(f"\n  📈 Averages:")
    print(f"    - Evidence per sample: {avg_evidence:.1f}")
    print(f"    - Hypothesis per sample: {avg_hypothesis:.1f}")

if __name__ == "__main__":
    quick_run()