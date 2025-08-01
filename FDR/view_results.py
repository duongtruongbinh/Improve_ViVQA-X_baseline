#!/usr/bin/env python3
# FDR/view_results.py - View and analyze saved results
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def load_summary(output_dir):
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

def list_samples(output_dir):
    """List all samples in output directory"""
    summary = load_summary(output_dir)
    if not summary:
        print(f"❌ No summary found in {output_dir}")
        return
    
    print(f"📊 Summary for {output_dir}:")
    print(f"  Total processed: {summary['total_processed']}")
    print(f"  Correct samples: {summary['correct_samples']}")
    print(f"  Wrong samples: {summary['wrong_samples']}")
    print(f"  Target reached: {summary['target_reached']}")
    
    # List correct samples
    correct_dir = os.path.join(output_dir, "correct")
    if os.path.exists(correct_dir):
        print(f"\n✅ Correct samples:")
        for i in range(1, summary['correct_samples'] + 1):
            sample_dir = os.path.join(correct_dir, f"sample_{i}")
            if os.path.exists(sample_dir):
                info = load_sample_info(sample_dir)
                if info:
                    print(f"  Sample {i}: {info['question'][:50]}...")
                    print(f"    Answer: {info['answer']}")
                    print(f"    Ground Truth: {info['ground_truth']}")
    
    # List wrong samples
    wrong_dir = os.path.join(output_dir, "wrong")
    if os.path.exists(wrong_dir):
        print(f"\n❌ Wrong samples:")
        for i in range(1, summary['wrong_samples'] + 1):
            sample_dir = os.path.join(wrong_dir, f"sample_{i}")
            if os.path.exists(sample_dir):
                info = load_sample_info(sample_dir)
                if info:
                    print(f"  Sample {i}: {info['question'][:50]}...")
                    print(f"    Answer: {info['answer']}")
                    print(f"    Ground Truth: {info['ground_truth']}")

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
    
    print(f"📋 Sample {sample_id} ({sample_type.upper()}):")
    print(f"  ID: {info['id']}")
    print(f"  Question: {info['question']}")
    print(f"  Answer: {info['answer']}")
    print(f"  Ground Truth: {info['ground_truth']}")
    print(f"  Explanation: {info['explanation']}")
    print(f"  Evidence: {info['evidence']}")
    print(f"  Hypothesis: {info['hypothesis']}")
    print(f"  Status: {info['synthesis_status']}")
    
    # Show images if available
    input_image = os.path.join(sample_dir, "input_image.jpg")
    output_image = os.path.join(sample_dir, "output_visualization.png")
    
    if os.path.exists(input_image):
        print(f"  Input Image: {input_image}")
    if os.path.exists(output_image):
        print(f"  Output Visualization: {output_image}")

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
    output_image_path = os.path.join(sample_dir, "output_visualization.png")
    
    if not os.path.exists(input_image_path) or not os.path.exists(output_image_path):
        print(f"❌ Images not found in {sample_dir}")
        return
    
    # Display images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Input image
    img1 = Image.open(input_image_path)
    ax1.imshow(img1)
    ax1.set_title(f"Input Image - Sample {sample_id} ({sample_type})", fontsize=12)
    ax1.axis('off')
    
    # Output visualization
    img2 = Image.open(output_image_path)
    ax2.imshow(img2)
    ax2.set_title(f"Output Visualization - Sample {sample_id}", fontsize=12)
    ax2.axis('off')
    
    plt.suptitle(f"Sample {sample_id} ({sample_type.upper()}) - {info['question'][:50]}...", fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_results(output_dir):
    """Analyze results and show statistics"""
    summary = load_summary(output_dir)
    if not summary:
        print(f"❌ No summary found in {output_dir}")
        return
    
    print(f"📊 Analysis for {output_dir}:")
    print(f"  Total processed: {summary['total_processed']}")
    print(f"  Correct samples: {summary['correct_samples']}")
    print(f"  Wrong samples: {summary['wrong_samples']}")
    
    if summary['total_processed'] > 0:
        accuracy = summary['correct_samples'] / summary['total_processed']
        print(f"  Accuracy: {accuracy:.2%}")
    
    # Analyze correct samples
    correct_dir = os.path.join(output_dir, "correct")
    if os.path.exists(correct_dir):
        print(f"\n✅ Correct samples analysis:")
        for i in range(1, summary['correct_samples'] + 1):
            sample_dir = os.path.join(correct_dir, f"sample_{i}")
            if os.path.exists(sample_dir):
                info = load_sample_info(sample_dir)
                if info:
                    print(f"  Sample {i}: {info['synthesis_status']} (Evidence: {len(info['evidence'])}, Hypothesis: {info['hypothesis']})")
    
    # Analyze wrong samples
    wrong_dir = os.path.join(output_dir, "wrong")
    if os.path.exists(wrong_dir):
        print(f"\n❌ Wrong samples analysis:")
        for i in range(1, summary['wrong_samples'] + 1):
            sample_dir = os.path.join(wrong_dir, f"sample_{i}")
            if os.path.exists(sample_dir):
                info = load_sample_info(sample_dir)
                if info:
                    print(f"  Sample {i}: {info['synthesis_status']} (Evidence: {len(info['evidence'])}, Hypothesis: {info['hypothesis']})")

def main():
    parser = argparse.ArgumentParser(description="View and analyze Auto Save Pipeline results")
    parser.add_argument("output_dir", nargs="?", default="auto_output", help="Output directory to analyze")
    parser.add_argument("--list", action="store_true", help="List all samples")
    parser.add_argument("--show", help="Show specific sample (format: type:id, e.g., correct:1)")
    parser.add_argument("--images", help="Show images for specific sample (format: type:id, e.g., correct:1)")
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
            print("❌ Invalid format. Use: type:id (e.g., correct:1)")
    elif args.images:
        try:
            sample_type, sample_id = args.images.split(":")
            show_images(args.output_dir, sample_type, int(sample_id))
        except ValueError:
            print("❌ Invalid format. Use: type:id (e.g., correct:1)")
    elif args.analyze:
        analyze_results(args.output_dir)
    else:
        # Default: show summary
        summary = load_summary(args.output_dir)
        if summary:
            print(f"📊 Summary for {args.output_dir}:")
            print(f"  Total processed: {summary['total_processed']}")
            print(f"  Correct samples: {summary['correct_samples']}")
            print(f"  Wrong samples: {summary['wrong_samples']}")
            print(f"  Target reached: {summary['target_reached']}")
        else:
            print(f"❌ No summary found in {args.output_dir}")

if __name__ == "__main__":
    main() 