#!/usr/bin/env python3
"""
Complete FDR vs ReRe Analysis Script
Runs comprehensive comparison and creates all visualizations
"""

import os
import sys
from pathlib import Path
from compare_fdr_rere import load_fdr_results, load_rere_results, compare_results, print_comparison_summary, create_comparison_visualization
from create_summary_visualization import create_summary_visualization

def main():
    print("🚀 COMPLETE FDR vs ReRe ANALYSIS")
    print("="*60)
    print("This script will:")
    print("  1. Load FDR results from output/fdr_results.json")
    print("  2. Load ReRe results from inference_results_100_samples.json")
    print("  3. Compare and categorize results")
    print("  4. Create summary visualization")
    print("  5. Create individual comparison visualizations")
    print("="*60)

    # Step 1: Load results
    print("\n📂 Step 1: Loading Results...")
    fdr_results = load_fdr_results("/home/khuonghuynh/Improve_ViVQA-X_baseline/output/fdr_results.json")
    rere_results = load_rere_results("inference_results_100_samples.json")
    
    if not fdr_results:
        print("❌ No FDR results found. Please run FDR inference first.")
        return False
    
    if not rere_results:
        print("❌ No ReRe results found. Please check inference_results_100_samples.json")
        return False
    
    # Step 2: Compare results
    print("\n🔍 Step 2: Comparing Results...")
    comparison = compare_results(fdr_results, rere_results)
    
    # Step 3: Print detailed summary
    print_comparison_summary(comparison)
    
    # Step 4: Create summary visualization
    print("\n🎨 Step 3: Creating Summary Visualization...")
    summary_file = create_summary_visualization(comparison, "fdr_vs_rere_summary.png")
    if summary_file:
        print(f"✅ Summary visualization saved: {summary_file}")
    
    # Step 5: Create individual visualizations
    print("\n🖼️  Step 4: Creating Individual Comparison Visualizations...")
    
    output_dir = "detailed_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Categories to visualize
    categories = {
        'both_correct': '✅✅ Both Correct',
        'both_wrong': '❌❌ Both Wrong', 
        'fdr_correct_rere_wrong': '🔥✅❌ FDR Better',
        'fdr_wrong_rere_correct': '🌟❌✅ ReRe Better'
    }
    
    total_visualizations = 0
    max_per_category = 5  # Limit to avoid too many files
    
    for category, description in categories.items():
        if category not in comparison or not comparison[category]:
            print(f"   📊 {description}: No samples found")
            continue
            
        samples = comparison[category][:max_per_category]
        print(f"   📊 {description}: Creating {len(samples)} visualizations...")
        
        category_count = 0
        for i, item in enumerate(samples):
            try:
                output_file = create_comparison_visualization(item, output_dir)
                category_count += 1
                total_visualizations += 1
                
                if i == 0:  # Show first example
                    print(f"      ✅ Example: {os.path.basename(output_file)}")
                    
            except Exception as e:
                print(f"      ❌ Error creating visualization for {item['question_id']}: {e}")
        
        print(f"      📁 Created {category_count} visualizations for this category")
    
    # Step 6: Create index file
    print(f"\n📋 Step 5: Creating Analysis Report...")
    create_analysis_report(comparison, output_dir, total_visualizations)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("📁 Generated Files:")
    print(f"   📊 Summary visualization: fdr_vs_rere_summary.png")
    print(f"   🖼️  Individual comparisons: {output_dir}/ ({total_visualizations} files)")
    print(f"   📋 Analysis report: analysis_report.txt")
    print("\n🔍 Key Findings:")
    
    # Quick insights
    total_common = sum(len(comparison[key]) for key in ['both_correct', 'both_wrong', 'fdr_correct_rere_wrong', 'fdr_wrong_rere_correct'])
    if total_common > 0:
        fdr_accuracy = (len(comparison['both_correct']) + len(comparison['fdr_correct_rere_wrong'])) / total_common
        rere_accuracy = (len(comparison['both_correct']) + len(comparison['fdr_wrong_rere_correct'])) / total_common
        
        print(f"   🎯 FDR Accuracy: {fdr_accuracy:.1%}")
        print(f"   🎯 ReRe Accuracy: {rere_accuracy:.1%}")
        
        if fdr_accuracy > rere_accuracy:
            print(f"   🏆 FDR performs better by {(fdr_accuracy-rere_accuracy)*100:.1f}%")
        elif rere_accuracy > fdr_accuracy:
            print(f"   🏆 ReRe performs better by {(rere_accuracy-fdr_accuracy)*100:.1f}%")
        else:
            print(f"   🤝 Both models perform equally")
        
        agreement_rate = (len(comparison['both_correct']) + len(comparison['both_wrong'])) / total_common
        print(f"   🤝 Agreement rate: {agreement_rate:.1%}")
    
    print("="*60)
    return True

def create_analysis_report(comparison, output_dir, total_visualizations):
    """Create a detailed text report of the analysis"""
    
    report_content = []
    report_content.append("FDR vs ReRe: Detailed Analysis Report")
    report_content.append("="*50)
    report_content.append("")
    
    # Summary statistics
    total_common = sum(len(comparison[key]) for key in ['both_correct', 'both_wrong', 'fdr_correct_rere_wrong', 'fdr_wrong_rere_correct'])
    
    report_content.append("SUMMARY STATISTICS:")
    report_content.append(f"  Total common samples: {total_common}")
    report_content.append(f"  FDR-only samples: {len(comparison['fdr_only'])}")
    report_content.append(f"  ReRe-only samples: {len(comparison['rere_only'])}")
    report_content.append("")
    
    # Detailed breakdown
    report_content.append("DETAILED BREAKDOWN:")
    report_content.append(f"  Both Correct: {len(comparison['both_correct'])} samples")
    report_content.append(f"  Both Wrong: {len(comparison['both_wrong'])} samples")
    report_content.append(f"  FDR Better: {len(comparison['fdr_correct_rere_wrong'])} samples")
    report_content.append(f"  ReRe Better: {len(comparison['fdr_wrong_rere_correct'])} samples")
    report_content.append("")
    
    # Performance metrics
    if total_common > 0:
        fdr_accuracy = (len(comparison['both_correct']) + len(comparison['fdr_correct_rere_wrong'])) / total_common
        rere_accuracy = (len(comparison['both_correct']) + len(comparison['fdr_wrong_rere_correct'])) / total_common
        agreement_rate = (len(comparison['both_correct']) + len(comparison['both_wrong'])) / total_common
        
        report_content.append("PERFORMANCE METRICS:")
        report_content.append(f"  FDR Accuracy: {fdr_accuracy:.4f} ({fdr_accuracy*100:.2f}%)")
        report_content.append(f"  ReRe Accuracy: {rere_accuracy:.4f} ({rere_accuracy*100:.2f}%)")
        report_content.append(f"  Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
        report_content.append("")
    
    # Generated files
    report_content.append("GENERATED FILES:")
    report_content.append(f"  Summary visualization: fdr_vs_rere_summary.png")
    report_content.append(f"  Individual comparisons: {output_dir}/ ({total_visualizations} files)")
    report_content.append("")
    
    # Sample details for each category
    categories = {
        'both_correct': 'Both Models Correct',
        'both_wrong': 'Both Models Wrong',
        'fdr_correct_rere_wrong': 'FDR Correct, ReRe Wrong',
        'fdr_wrong_rere_correct': 'FDR Wrong, ReRe Correct'
    }
    
    for category, description in categories.items():
        if category in comparison and comparison[category]:
            report_content.append(f"{description.upper()}:")
            for i, item in enumerate(comparison[category][:3]):  # Show first 3
                report_content.append(f"  {i+1}. Question ID: {item['question_id']}")
                report_content.append(f"     Question: {item['question'][:80]}...")
                report_content.append(f"     Ground Truth: {item['ground_truth']}")
                report_content.append(f"     FDR Answer: {item['fdr']['predicted_answer']}")
                report_content.append(f"     ReRe Answer: {item['rere']['predicted_answer']}")
                report_content.append("")
            
            if len(comparison[category]) > 3:
                report_content.append(f"  ... and {len(comparison[category]) - 3} more samples")
            report_content.append("")
    
    # Save report
    with open("analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
