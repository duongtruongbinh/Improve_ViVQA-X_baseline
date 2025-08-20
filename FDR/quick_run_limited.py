#!/usr/bin/env python3
# FDR/quick_run_limited.py - Quick run with limited samples for testing
import os
import sys
import json
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

def quick_run_limited():
    """Quick run with only 5 samples for testing"""
    print("🚀 Quick Run - Limited Samples (Testing)")
    print("📊 Processing first 5 samples only")
    print("📁 Output: results_visualization_test")
    print("🔍 GroundingDINO: Enabled (if available)")
    print("-" * 50)
    
    try:
        from quick_run_from_results import (
            load_results, match_samples, categorize_samples, 
            create_output_directories, save_category_samples, create_summary
        )
        
        # Load results
        fdr_results, rere_results = load_results()
        if not fdr_results or not rere_results:
            print("❌ Failed to load results")
            return None
        
        # Limit to first 5 samples for testing
        fdr_results = fdr_results[:5]
        rere_results = rere_results[:5]
        
        print(f"📊 Testing with {len(fdr_results)} FDR and {len(rere_results)} ReRe samples")
        
        # Match samples
        matched_samples = match_samples(fdr_results, rere_results)
        if not matched_samples:
            print("❌ No matching samples found")
            return None
        
        print(f"🔗 Matched {len(matched_samples)} samples")
        
        # Categorize samples
        categories = categorize_samples(matched_samples)
        
        print(f"📋 Categories:")
        for category, samples in categories.items():
            print(f"  {category}: {len(samples)} samples")
        
        # Create output directories
        base_dir = "results_visualization_test"
        correct_dir = os.path.join(base_dir, "correct")
        wrong_dir = os.path.join(base_dir, "wrong")
        fdr_better_dir = os.path.join(base_dir, "fdr_better")
        rere_better_dir = os.path.join(base_dir, "rere_better")
        
        for dir_path in [correct_dir, wrong_dir, fdr_better_dir, rere_better_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"\n📁 Created test output directories in {base_dir}")
        
        # Save samples and create visualizations (max 2 per category for testing)
        total_saved = 0
        for category_name, samples in categories.items():
            if not samples:
                continue
                
            if category_name == 'both_correct':
                target_dir = correct_dir
                max_samples = 2
            elif category_name == 'both_wrong':
                target_dir = wrong_dir
                max_samples = 2
            elif category_name == 'fdr_better':
                target_dir = fdr_better_dir
                max_samples = 2
            elif category_name == 'rere_better':
                target_dir = rere_better_dir
                max_samples = 2
            
            print(f"\n🎨 Processing {category_name} samples...")
            saved_count = save_category_samples(samples[:max_samples], target_dir, category_name)
            total_saved += saved_count
            print(f"💾 Saved {saved_count}/{len(samples)} {category_name} samples")
        
        # Create summary
        summary = create_summary(categories, len(matched_samples))
        
        # Update summary file path for test
        summary_file = os.path.join(base_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Limited test completed successfully!")
        print(f"📊 Results:")
        print(f"  - Total processed: {summary['total_processed']}")
        print(f"  - Both correct: {summary['both_correct']}")
        print(f"  - Both wrong: {summary['both_wrong']}")
        print(f"  - FDR better: {summary['fdr_better']}")
        print(f"  - ReRe better: {summary['rere_better']}")
        print(f"📁 Test output directory: {base_dir}")
        
        # Show next steps
        print(f"\n📋 Next steps:")
        print(f"  - View results: python view_results_comparison.py {base_dir}")
        print(f"  - List samples: python view_results_comparison.py {base_dir} --list")
        print(f"  - Show sample: python view_results_comparison.py {base_dir} --show correct:1")
        print(f"  - Show images: python view_results_comparison.py {base_dir} --images fdr_better:1")
        
        return summary
        
    except Exception as e:
        print(f"❌ Limited test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    quick_run_limited()
