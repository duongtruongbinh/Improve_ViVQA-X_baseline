#!/usr/bin/env python3
# FDR/view_results_comparison.py - View FDR vs ReRe comparison results
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def load_summary(output_dir="results_visualization"):
    """Load summary from output directory"""
    summary_file = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_sample_info(sample_dir):
    """Load sample information from directory"""
    info_file = os.path.join(sample_dir, "info.json")
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def list_samples(output_dir="results_visualization"):
    """List all samples in output directory"""
    summary = load_summary(output_dir)
    if not summary:
        print(f"❌ No summary found in {output_dir}")
        return
    
    print(f"📊 FDR vs ReRe Comparison Summary:")
    print(f"  Total processed: {summary['total_processed']}")
    print(f"  Both correct: {summary['both_correct']}")
    print(f"  Both wrong: {summary['both_wrong']}")
    print(f"  FDR better: {summary['fdr_better']}")
    print(f"  ReRe better: {summary['rere_better']}")
    print(f"  FDR accuracy: {summary['fdr_accuracy']:.2%}")
    print(f"  ReRe accuracy: {summary['rere_accuracy']:.2%}")
    print(f"  Agreement rate: {summary['agreement_rate']:.2%}")
    
    # List samples by category
    categories = ['correct', 'wrong', 'fdr_better', 'rere_better']
    category_names = ['Both Correct', 'Both Wrong', 'FDR Better', 'ReRe Better']
    
    for category, name in zip(categories, category_names):
        category_dir = os.path.join(output_dir, category)
        if os.path.exists(category_dir):
            samples = [d for d in os.listdir(category_dir) if d.startswith('sample_')]
            print(f"\n{name}: {len(samples)} samples")
            
            for sample_dir_name in sorted(samples):
                sample_dir = os.path.join(category_dir, sample_dir_name)
                info = load_sample_info(sample_dir)
                if info:
                    sample_num = sample_dir_name.replace('sample_', '')
                    print(f"  Sample {sample_num}: {info['question'][:60]}...")
                    print(f"    FDR: {info['fdr_answer']} {'✅' if info['fdr_correct'] else '❌'}")
                    print(f"    ReRe: {info['rere_answer']} {'✅' if info['rere_correct'] else '❌'}")
                    print(f"    GT: {info['ground_truth']}")

def show_sample(output_dir, sample_type, sample_id):
    """Show detailed information for a specific sample"""
    sample_dir = os.path.join(output_dir, sample_type, f"sample_{sample_id}")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample not found: {sample_dir}")
        return
    
    info = load_sample_info(sample_dir)
    if not info:
        print(f"❌ Could not load sample info from {sample_dir}")
        return
    
    print(f"📋 Sample {sample_id} ({sample_type.upper().replace('_', ' ')}):")
    print(f"  ID: {info['id']}")
    print(f"  Question: {info['question']}")
    print(f"  Ground Truth: {info['ground_truth']}")
    print(f"  Category: {info['category']}")
    
    print(f"\n🤖 FDR Results:")
    print(f"  Answer: {info['fdr_answer']} {'✅' if info['fdr_correct'] else '❌'}")
    print(f"  Explanation: {info['fdr_explanation']}")
    print(f"  Status: {info['fdr_synthesis_status']}")
    print(f"  Evidence: {info['fdr_evidence_count']}, Hypothesis: {info['fdr_hypothesis_count']}")
    
    print(f"\n🔵 ReRe Results:")
    print(f"  Answer: {info['rere_answer']} {'✅' if info['rere_correct'] else '❌'}")
    print(f"  Explanation: {info['rere_explanation']}")
    if info.get('rere_full_caption'):
        print(f"  Full Caption: {info['rere_full_caption']}")
    
    # Show available files
    input_image = os.path.join(sample_dir, "input_image.jpg")
    comparison_viz = os.path.join(sample_dir, "comparison_visualization.png")
    
    if os.path.exists(input_image):
        print(f"\n📁 Files:")
        print(f"  Input Image: {input_image}")
    if os.path.exists(comparison_viz):
        print(f"  Comparison Visualization: {comparison_viz}")

def show_images(output_dir, sample_type, sample_id):
    """Display images for a specific sample"""
    sample_dir = os.path.join(output_dir, sample_type, f"sample_{sample_id}")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample not found: {sample_dir}")
        return

    # Load sample info
    info = load_sample_info(sample_dir)
    if not info:
        print(f"❌ Could not load sample info")
        return

    # Check for images
    input_image_path = os.path.join(sample_dir, "input_image.jpg")
    comparison_viz_path = os.path.join(sample_dir, "comparison_visualization.png")

    if not os.path.exists(input_image_path) and not os.path.exists(comparison_viz_path):
        print(f"❌ No images found in {sample_dir}")
        return

    # Determine layout based on available images
    if os.path.exists(input_image_path) and os.path.exists(comparison_viz_path):
        # Show both images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Input image
        img1 = Image.open(input_image_path)
        ax1.imshow(img1)
        ax1.set_title("Input Image", fontsize=12)
        ax1.axis('off')
        
        # Comparison visualization
        img2 = Image.open(comparison_viz_path)
        ax2.imshow(img2)
        ax2.set_title("FDR vs ReRe Comparison", fontsize=12)
        ax2.axis('off')
        
    elif os.path.exists(comparison_viz_path):
        # Show only comparison visualization
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        img = Image.open(comparison_viz_path)
        ax.imshow(img)
        ax.set_title("FDR vs ReRe Comparison", fontsize=12)
        ax.axis('off')
        
    elif os.path.exists(input_image_path):
        # Show only input image
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        img = Image.open(input_image_path)
        ax.imshow(img)
        ax.set_title("Input Image", fontsize=12)
        ax.axis('off')

    plt.suptitle(f"Sample {sample_id} ({sample_type.upper().replace('_', ' ')}) - {info['question'][:50]}...", fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_results(output_dir="results_visualization"):
    """Analyze results and show detailed statistics"""
    summary = load_summary(output_dir)
    if not summary:
        print(f"❌ No summary found in {output_dir}")
        return
    
    print(f"📊 Detailed Analysis for FDR vs ReRe:")
    print(f"  Total processed: {summary['total_processed']}")
    print(f"  FDR accuracy: {summary['fdr_accuracy']:.2%}")
    print(f"  ReRe accuracy: {summary['rere_accuracy']:.2%}")
    print(f"  Agreement rate: {summary['agreement_rate']:.2%}")
    
    print(f"\n📈 Performance Breakdown:")
    print(f"  Both models correct: {summary['both_correct']} ({summary['both_correct']/summary['total_processed']:.1%})")
    print(f"  Both models wrong: {summary['both_wrong']} ({summary['both_wrong']/summary['total_processed']:.1%})")
    print(f"  FDR better than ReRe: {summary['fdr_better']} ({summary['fdr_better']/summary['total_processed']:.1%})")
    print(f"  ReRe better than FDR: {summary['rere_better']} ({summary['rere_better']/summary['total_processed']:.1%})")
    
    # Calculate additional metrics
    fdr_total_correct = summary['both_correct'] + summary['fdr_better']
    rere_total_correct = summary['both_correct'] + summary['rere_better']
    
    print(f"\n🎯 Model Performance:")
    print(f"  FDR total correct: {fdr_total_correct}/{summary['total_processed']} ({fdr_total_correct/summary['total_processed']:.1%})")
    print(f"  ReRe total correct: {rere_total_correct}/{summary['total_processed']} ({rere_total_correct/summary['total_processed']:.1%})")
    
    improvement = fdr_total_correct - rere_total_correct
    print(f"  FDR improvement: +{improvement} samples ({improvement/summary['total_processed']:.1%})")

def main():
    parser = argparse.ArgumentParser(description="View FDR vs ReRe comparison results")
    parser.add_argument("output_dir", nargs="?", default="results_visualization", help="Output directory to analyze")
    parser.add_argument("--list", action="store_true", help="List all samples")
    parser.add_argument("--show", help="Show specific sample (format: type:id, e.g., correct:1, fdr_better:2)")
    parser.add_argument("--images", help="Show images for specific sample (format: type:id, e.g., rere_better:1)")
    parser.add_argument("--analyze", action="store_true", help="Analyze results and show statistics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        return
    
    if args.list:
        list_samples(args.output_dir)
    elif args.show:
        try:
            sample_type, sample_id = args.show.split(":")
            show_sample(args.output_dir, sample_type, int(sample_id))
        except ValueError:
            print("❌ Invalid format. Use: type:id (e.g., correct:1, fdr_better:2)")
    elif args.images:
        try:
            sample_type, sample_id = args.images.split(":")
            show_images(args.output_dir, sample_type, int(sample_id))
        except ValueError:
            print("❌ Invalid format. Use: type:id (e.g., rere_better:1)")
    elif args.analyze:
        analyze_results(args.output_dir)
    else:
        # Default: show summary
        summary = load_summary(args.output_dir)
        if summary:
            print(f"📊 FDR vs ReRe Summary:")
            print(f"  Total processed: {summary['total_processed']}")
            print(f"  FDR accuracy: {summary['fdr_accuracy']:.2%}")
            print(f"  ReRe accuracy: {summary['rere_accuracy']:.2%}")
            print(f"  Both correct: {summary['both_correct']}")
            print(f"  Both wrong: {summary['both_wrong']}")
            print(f"  FDR better: {summary['fdr_better']}")
            print(f"  ReRe better: {summary['rere_better']}")
        else:
            print(f"❌ No summary found in {args.output_dir}")

if __name__ == "__main__":
    main()
