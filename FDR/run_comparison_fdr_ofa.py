
#!/usr/bin/env python3

# FDR/run_comparison_fdr_ofa.py - Run FDR vs OFA-X Comparison on 100 samples
import os
import sys
import json
import yaml
import logging
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure matplotlib to avoid font warnings
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_ofa_results():
    """Load OFA results from JSON file"""
    ofa_results_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/FDR/ofa_final_results.json"
    
    print(f"📂 Loading OFA results from: {ofa_results_path}")
    
    try:
        with open(ofa_results_path, 'r', encoding='utf-8') as f:
            ofa_results = json.load(f)
        print(f"✅ Loaded {len(ofa_results)} OFA samples")
        return ofa_results
    except Exception as e:
        print(f"❌ Error loading OFA results: {e}")
        return None

def load_fdr_results():
    """Load FDR results from fdr_100_output directory"""
    auto_output_dir = "fdr_100_output"  # Thay đổi từ "auto_output" thành "fdr_100_output"
    
    print(f"📂 Loading FDR results from: {auto_output_dir}")
    
    if not os.path.exists(auto_output_dir):
        print(f"❌ Directory {auto_output_dir} not found")
        return None
    
    fdr_results = []
    
    # Load from correct and wrong directories
    for category in ['correct', 'wrong']:
        category_dir = os.path.join(auto_output_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        for sample_dir in os.listdir(category_dir):
            sample_path = os.path.join(category_dir, sample_dir)
            if not os.path.isdir(sample_path):
                continue
                
            info_file = os.path.join(sample_path, "info.json")
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        sample_data = json.load(f)
                    
                    # Add image path
                    input_image_path = os.path.join(sample_path, "input_image.jpg")
                    if os.path.exists(input_image_path):
                        sample_data['image_path'] = input_image_path
                    
                    fdr_results.append(sample_data)
                except Exception as e:
                    print(f"⚠️ Error loading {info_file}: {e}")
                    continue
    
    print(f"✅ Loaded {len(fdr_results)} FDR samples")
    return fdr_results

def match_samples_by_question_id(fdr_results, ofa_results):
    """Match FDR and OFA samples by question_id"""
    print("🔗 Matching samples by question_id...")
    
    # Create lookup dictionary for OFA results
    ofa_lookup = {}
    for sample in ofa_results:
        question_id = str(sample.get('question_id', ''))
        ofa_lookup[question_id] = sample
    
    matched_samples = []
    for fdr_sample in fdr_results:
        fdr_question_id = str(fdr_sample.get('id', ''))  # FDR uses 'id' field
        
        if fdr_question_id in ofa_lookup:
            ofa_sample = ofa_lookup[fdr_question_id]
            
            matched_sample = {
                'question_id': fdr_question_id,
                'fdr_sample': fdr_sample,
                'ofa_sample': ofa_sample,
                'question': fdr_sample.get('question', ''),
                'ground_truth': fdr_sample.get('ground_truth', ''),
                'fdr_answer': fdr_sample.get('answer', ''),
                'ofa_answer': ofa_sample.get('prediction', {}).get('answer', ''),
                'fdr_explanation': fdr_sample.get('explanation', ''),
                'ofa_explanation': ofa_sample.get('prediction', {}).get('explanation', ''),
                'image_path': fdr_sample.get('image_path', '')
            }
            matched_samples.append(matched_sample)
    
    print(f"✅ Matched {len(matched_samples)} samples")
    return matched_samples

def categorize_samples(matched_samples):
    """Categorize samples based on FDR and OFA correctness"""
    print("📊 Categorizing samples...")
    
    categories = {
        'fdr_correct_ofa_correct': [],
        'fdr_wrong_ofa_wrong': [],
        'fdr_correct_ofa_wrong': [],
        'fdr_wrong_ofa_correct': []
    }
    
    for sample in matched_samples:
        ground_truth = sample['ground_truth'].lower().strip()
        fdr_answer = sample['fdr_answer'].lower().strip()
        ofa_answer = sample['ofa_answer'].lower().strip()
        
        fdr_correct = (fdr_answer == ground_truth)
        ofa_correct = (ofa_answer == ground_truth)
        
        if fdr_correct and ofa_correct:
            categories['fdr_correct_ofa_correct'].append(sample)
        elif not fdr_correct and not ofa_correct:
            categories['fdr_wrong_ofa_wrong'].append(sample)
        elif fdr_correct and not ofa_correct:
            categories['fdr_correct_ofa_wrong'].append(sample)
        elif not fdr_correct and ofa_correct:
            categories['fdr_wrong_ofa_correct'].append(sample)
    
    # Print category statistics
    for category, samples in categories.items():
        print(f"  {category}: {len(samples)} samples")
    
    return categories

def create_comparison_directories(base_dir="comparison_fdr_ofa_output"):
    """Create directories for comparison results"""
    fdr_correct_ofa_correct_dir = os.path.join(base_dir, "fdr_correct_ofa_correct")
    fdr_wrong_ofa_wrong_dir = os.path.join(base_dir, "fdr_wrong_ofa_wrong")
    fdr_correct_ofa_wrong_dir = os.path.join(base_dir, "fdr_correct_ofa_wrong")
    fdr_wrong_ofa_correct_dir = os.path.join(base_dir, "fdr_wrong_ofa_correct")
    
    for dir_path in [fdr_correct_ofa_correct_dir, fdr_wrong_ofa_wrong_dir, 
                     fdr_correct_ofa_wrong_dir, fdr_wrong_ofa_correct_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return (fdr_correct_ofa_correct_dir, fdr_wrong_ofa_wrong_dir, 
            fdr_correct_ofa_wrong_dir, fdr_wrong_ofa_correct_dir)

def save_comparison_sample(sample, sample_dir, category_name):
    """Save comparison sample with enhanced visualization (keeping FDR format + adding OFA info)"""
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save sample info (enhanced with OFA data)
    sample_info = {
        'id': sample['question_id'],
        'question': sample['question'],
        'ground_truth': sample['ground_truth'],
        'fdr_answer': sample['fdr_answer'],
        'fdr_explanation': sample['fdr_explanation'],
        'ofa_answer': sample['ofa_answer'],
        'ofa_explanation': sample['ofa_explanation'],
        'category': category_name,
        'image_path': sample['image_path'],
        'fdr_sample': sample['fdr_sample'],  # Keep full FDR data
        'ofa_sample': sample['ofa_sample']  # Keep full OFA data
    }
    
    # Save info.json
    info_file = os.path.join(sample_dir, "info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(sample_info, f, indent=2, ensure_ascii=False)
    
    # Copy input image if exists
    if sample['image_path'] and os.path.exists(sample['image_path']):
        input_image_dest = os.path.join(sample_dir, "input_image.jpg")
        try:
            shutil.copy2(sample['image_path'], input_image_dest)
        except Exception as e:
            print(f"⚠️ Could not copy image: {e}")
    
    # Create enhanced visualization
    create_enhanced_visualization(sample, sample_dir, category_name)

def create_enhanced_visualization(sample, sample_dir, category_name):
    """Create enhanced visualization keeping FDR format but adding OFA information"""
    try:
        # Check if GroundingDINO annotated image exists
        groundingdino_path = sample['fdr_sample'].get('groundingdino_annotated_path', '')
        has_groundingdino = groundingdino_path and os.path.exists(groundingdino_path)

        if has_groundingdino:
            # 3-panel layout: Input, GroundingDINO, Enhanced Text (FDR + OFA)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        else:
            # 2-panel layout: Input, Enhanced Text (FDR + OFA)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 8))
            ax2 = None

        # Panel 1: Input image
        ax1.axis('off')
        ax1.set_title("Input Image", fontsize=12, fontweight='bold')

        if sample['image_path'] and os.path.exists(sample['image_path']):
            try:
                input_img = Image.open(sample['image_path'])
                ax1.imshow(input_img)
            except Exception as e:
                ax1.text(0.5, 0.5, f"Error loading image:\n{e}",
                        ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, "No input image available",
                    ha='center', va='center', transform=ax1.transAxes)

        # Panel 2: GroundingDINO (if available)
        if has_groundingdino and ax2 is not None:
            ax2.axis('off')
            ax2.set_title("GroundingDINO Detection", fontsize=12, fontweight='bold')

            try:
                grounding_img = Image.open(groundingdino_path)
                ax2.imshow(grounding_img)
            except Exception as e:
                ax2.text(0.5, 0.5, f"Error loading GroundingDINO image:\n{e}",
                        ha='center', va='center', transform=ax2.transAxes)

        # Panel 3: Enhanced text with both FDR and OFA information
        ax3.axis('off')

        # Prepare enhanced text content
        question = sample['question']
        ground_truth = sample['ground_truth']
        fdr_answer = sample['fdr_answer']
        ofa_answer = sample['ofa_answer']
        fdr_explanation = sample['fdr_explanation']
        ofa_explanation = sample['ofa_explanation']

        # Determine colors based on correctness
        fdr_color = 'green' if fdr_answer.lower().strip() == ground_truth.lower().strip() else 'red'
        ofa_color = 'green' if ofa_answer.lower().strip() == ground_truth.lower().strip() else 'red'

        # Question
        ax3.text(0.05, 0.95, f"Q: {question}",
                fontsize=11, fontweight='bold', color='black', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # Ground Truth
        ax3.text(0.05, 0.90, f"Ground Truth: {ground_truth}",
                fontsize=10, fontweight='bold', color='#4a9eff', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # FDR Section Separator
        ax3.text(0.05, 0.85, "==== PHÂN TÍCH CỦA FDR ====",
                fontsize=10, fontweight='bold', color='#d32f2f', transform=ax3.transAxes,
                verticalalignment='top')

        # FDR Answer with bullet
        ax3.text(0.05, 0.80, f"● Answer: {fdr_answer}",
                fontsize=10, fontweight='bold', color=fdr_color, transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # FDR Explanation
        if fdr_explanation:
            ax3.text(0.05, 0.75, f"Giải thích: {fdr_explanation}",
                    fontsize=9, color='#2e7d32', transform=ax3.transAxes,
                    verticalalignment='top', wrap=True)

        # Evidence Details (if available from FDR sample)
        fdr_sample = sample.get('fdr_sample', {})
        evidence_set = fdr_sample.get('evidence_set', [])
        evidence_y_pos = 0.68

        if evidence_set:
            # Evidence header
            ax3.text(0.05, evidence_y_pos, "[+] Bằng chứng (Evidence):",
                    fontsize=10, fontweight='bold', color='#7b1fa2', transform=ax3.transAxes,
                    verticalalignment='top')

            # Evidence items
            evidence_text = ""
            for i, evidence in enumerate(evidence_set[:3]):  # Show top 3 evidence
                evidence_id = evidence.get('evidence_id', f'E{i+1}')
                answer = evidence.get('answer', 'N/A')
                question_ev = evidence.get('question', '')
                if question_ev:
                    evidence_text += f"  • {evidence_id}: Q: {question_ev[:50]}...\n"
                    evidence_text += f"    A: {answer[:50]}...\n"
                else:
                    evidence_text += f"  • {evidence_id}: {answer[:50]}...\n"

            ax3.text(0.05, evidence_y_pos - 0.05, evidence_text,
                    fontsize=9, color='#7b1fa2', transform=ax3.transAxes,
                    verticalalignment='top')

        # Hypothesis Details (moved closer to Evidence)
        hypothesis_set = fdr_sample.get('hypothesis_set', [])
        hypothesis_y_pos = 0.50 if evidence_set else 0.68  # Closer to evidence

        if hypothesis_set:
            # Hypothesis header
            ax3.text(0.05, hypothesis_y_pos, "[+] Giả thuyết (Hypothesis):",
                    fontsize=10, fontweight='bold', color='#5d4037', transform=ax3.transAxes,
                    verticalalignment='top')

            # Hypothesis items
            hypothesis_text = ""
            for i, hypothesis in enumerate(hypothesis_set[:3]):  # Show top 3 hypothesis
                hyp_id = hypothesis.get('hypothesis_id', f'H{i+1}')
                final_answer = hypothesis.get('THEN', {}).get('final_answer', 'N/A')
                confidence = hypothesis.get('confidence_source', 0.0)
                reasoning = hypothesis.get('reasoning_description', '')

                hypothesis_text += f"  • {hyp_id}: → {final_answer} (conf: {confidence:.2f})\n"
                if reasoning:
                    hypothesis_text += f"    Lý do: {reasoning[:50]}...\n"

            ax3.text(0.05, hypothesis_y_pos - 0.05, hypothesis_text,
                    fontsize=9, color='#5d4037', transform=ax3.transAxes,
                    verticalalignment='top')

        # OFA Section (new addition below hypothesis)
        ofa_y_start = 0.35 if (evidence_set or hypothesis_set) else 0.50

        # OFA separator
        ax3.text(0.05, ofa_y_start, "==== SO SÁNH VỚI OFA-X ====",
                fontsize=10, fontweight='bold', color='#1976d2',
                transform=ax3.transAxes, verticalalignment='top')

        # OFA Answer with bullet
        ax3.text(0.05, ofa_y_start - 0.05, f"● OFA Answer: {ofa_answer}",
                fontsize=10, fontweight='bold', color=ofa_color,
                transform=ax3.transAxes, verticalalignment='top')

        # OFA Explanation
        ax3.text(0.05, ofa_y_start - 0.10, f"Giải thích: {ofa_explanation}",
                fontsize=9, color='#2e7d32', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # Bottom separator line
        ax3.text(0.05, 0.15, "─" * 50,
                fontsize=8, color='#cccccc', transform=ax3.transAxes,
                verticalalignment='top')

        # Summary counts
        evidence_count = len(evidence_set)
        hypothesis_count = len(hypothesis_set)
        ax3.text(0.05, 0.10, f"Tổng: {evidence_count} Bằng chứng, {hypothesis_count} Giả thuyết",
                fontsize=9, color='#757575', fontweight='bold', transform=ax3.transAxes,
                verticalalignment='top')

        # Final result with text symbols
        fdr_result = 'CORRECT' if fdr_answer.lower().strip() == ground_truth.lower().strip() else 'WRONG'
        ofa_result = 'CORRECT' if ofa_answer.lower().strip() == ground_truth.lower().strip() else 'WRONG'

        # Color coding for results
        fdr_result_color = '#4caf50' if fdr_result == 'CORRECT' else '#f44336'
        ofa_result_color = '#4caf50' if ofa_result == 'CORRECT' else '#f44336'

        ax3.text(0.05, 0.05, "Result: ", fontsize=10, color='#ff9800', fontweight='bold',
                transform=ax3.transAxes, verticalalignment='top')
        ax3.text(0.15, 0.05, f"FDR ({fdr_result})", fontsize=10, color=fdr_result_color, fontweight='bold',
                transform=ax3.transAxes, verticalalignment='top')
        ax3.text(0.35, 0.05, f", OFA ({ofa_result})", fontsize=10, color=ofa_result_color, fontweight='bold',
                transform=ax3.transAxes, verticalalignment='top')

        ax3.set_title("FDR vs OFA-X Comparison", fontsize=14, fontweight='bold')

        # Save the visualization
        plt.tight_layout()
        output_path = os.path.join(sample_dir, "output_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"⚠️ Error creating visualization for {sample['question_id']}: {e}")

def main():
    """Main function to run FDR-OFA comparison"""
    print("🚀 Starting FDR vs OFA-X Comparison on 100 samples")
    print("-" * 60)

    # Step 1: Load OFA results
    ofa_results = load_ofa_results()
    if not ofa_results:
        print("❌ Cannot proceed without OFA results")
        return

    # Step 2: Load FDR results
    print("\n" + "="*60)
    fdr_results = load_fdr_results()
    if not fdr_results:
        print("❌ Cannot proceed without FDR results")
        return

    # Step 3: Match samples
    print("\n" + "="*60)
    matched_samples = match_samples_by_question_id(fdr_results, ofa_results)
    if not matched_samples:
        print("❌ No matching samples found")
        return

    # Step 4: Categorize samples
    print("\n" + "="*60)
    categories = categorize_samples(matched_samples)

    # Step 5: Create output directories
    print("\n" + "="*60)
    print("📁 Creating comparison output directories...")
    dirs = create_comparison_directories("comparison_fdr_ofa_output")

    # Step 6: Save samples to appropriate directories
    print("\n" + "="*60)
    print("💾 Saving comparison samples...")

    category_mapping = {
        'fdr_correct_ofa_correct': dirs[0],
        'fdr_correct_ofa_wrong': dirs[2],
        'fdr_wrong_ofa_correct': dirs[3],
        'fdr_wrong_ofa_wrong': dirs[1]
    }

    total_saved = 0
    for category_name, samples in categories.items():
        if not samples:
            continue

        target_dir = category_mapping[category_name]
        print(f"📂 Processing {category_name}: {len(samples)} samples")

        for i, sample in enumerate(samples, 1):
            sample_dir = os.path.join(target_dir, f"sample_{i}")
            save_comparison_sample(sample, sample_dir, category_name)
            total_saved += 1

            if i % 10 == 0:
                print(f"  ✅ Saved {i}/{len(samples)} samples")

    # Step 7: Create summary
    print("\n" + "="*60)
    summary = {
        'total_processed': len(matched_samples),
        'fdr_correct_ofa_correct': len(categories['fdr_correct_ofa_correct']),
        'fdr_wrong_ofa_wrong': len(categories['fdr_wrong_ofa_wrong']),
        'fdr_correct_ofa_wrong': len(categories['fdr_correct_ofa_wrong']),
        'fdr_wrong_ofa_correct': len(categories['fdr_wrong_ofa_correct']),
        'total_saved': total_saved
    }

    # Save summary
    summary_file = "comparison_fdr_ofa_output/comparison_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final results
    print("🎉 FDR vs OFA-X Comparison completed!")
    print(f"📊 Results:")
    print(f"  - Total processed: {summary['total_processed']}")
    print(f"  - FDR✅ OFA✅: {summary['fdr_correct_ofa_correct']}")
    print(f"  - FDR❌ OFA❌: {summary['fdr_wrong_ofa_wrong']}")
    print(f"  - FDR✅ OFA❌: {summary['fdr_correct_ofa_wrong']}")
    print(f"  - FDR❌ OFA✅: {summary['fdr_wrong_ofa_correct']}")
    print(f"📁 Output directory: comparison_fdr_ofa_output/")
    print(f"📋 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
