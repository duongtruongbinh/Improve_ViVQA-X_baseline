#!/usr/bin/env python3
# FDR/run_comparison_100.py - Run FDR on 100 samples and compare with OFA-X
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

# Configure matplotlib to avoid font warnings
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

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

def run_fdr_pipeline():
    """Run FDR pipeline on 100 samples"""
    print("🚀 Running FDR pipeline on 100 samples...")
    
    try:
        from src.auto_save_pipeline import run_auto_save_pipeline
        
        # Run FDR pipeline with 100 samples (no max_correct/max_wrong limits)
        summary = run_auto_save_pipeline(
            use_vllm=True,
            max_correct=100,  # Set high to get all samples
            max_wrong=100,    # Set high to get all samples
            output_base_dir="fdr_100_output"
        )
        
        return summary
        
    except Exception as e:
        print(f"❌ FDR pipeline failed: {e}")
        return None

def load_fdr_results():
    """Load FDR results from the output directory"""
    fdr_results = []
    
    # Check both correct and wrong directories
    for category in ['correct', 'wrong']:
        category_dir = f"fdr_100_output/{category}"
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
                        sample_info = json.load(f)
                    fdr_results.append(sample_info)
                except Exception as e:
                    print(f"⚠️ Error loading {info_file}: {e}")
    
    print(f"✅ Loaded {len(fdr_results)} FDR results")
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
    """Categorize samples based on correctness comparison"""
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
    
    print(f"📊 Categorization results:")
    for category, samples in categories.items():
        print(f"  - {category}: {len(samples)} samples")
    
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
    
    # Copy input image from original FDR output first
    try:
        # Find the original FDR sample directory
        fdr_sample = sample['fdr_sample']
        original_sample_dir = None

        # Look for the original sample in FDR output
        for category in ['correct', 'wrong']:
            category_dir = f"fdr_100_output/{category}"
            if os.path.exists(category_dir):
                for sample_folder in os.listdir(category_dir):
                    sample_path = os.path.join(category_dir, sample_folder)
                    info_file = os.path.join(sample_path, "info.json")
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        if str(info.get('id', '')) == str(sample['question_id']):
                            original_sample_dir = sample_path
                            break
                if original_sample_dir:
                    break

        # Copy input image from original FDR output
        if original_sample_dir:
            original_input_path = os.path.join(original_sample_dir, "input_image.jpg")
            if os.path.exists(original_input_path):
                shutil.copy2(original_input_path, os.path.join(sample_dir, "input_image.jpg"))
                # Update the image_path in sample_info to point to the copied image
                sample_info['image_path'] = os.path.join(sample_dir, "input_image.jpg")
            else:
                logging.warning(f"Input image not found in original FDR output: {original_input_path}")
        else:
            logging.warning(f"Original FDR sample directory not found for question_id: {sample['question_id']}")

    except Exception as e:
        logging.warning(f"Could not copy input image: {e}")

    # Save sample info after updating image_path
    with open(os.path.join(sample_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_info, f, indent=2, ensure_ascii=False)

    # Copy GroundingDINO image if exists (from FDR output)
    copy_groundingdino_image(sample, sample_dir)
    
    # Create enhanced visualization (FDR format + ReRe info)
    create_enhanced_visualization(sample, sample_dir, category_name)

def copy_groundingdino_image(sample, sample_dir):
    """Copy GroundingDINO annotated image from FDR output if available"""
    try:
        # Look for GroundingDINO image in original FDR output
        fdr_sample = sample['fdr_sample']
        original_sample_dir = None
        
        # Find the original FDR sample directory
        for category in ['correct', 'wrong']:
            category_dir = f"fdr_100_output/{category}"
            if os.path.exists(category_dir):
                for sample_folder in os.listdir(category_dir):
                    sample_path = os.path.join(category_dir, sample_folder)
                    info_file = os.path.join(sample_path, "info.json")
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        if str(info.get('id', '')) == str(sample['question_id']):
                            original_sample_dir = sample_path
                            break
                if original_sample_dir:
                    break
        
        if original_sample_dir:
            # Copy GroundingDINO image if exists
            groundingdino_path = os.path.join(original_sample_dir, "groundingdino_annotated.jpg")
            if os.path.exists(groundingdino_path):
                import shutil
                shutil.copy2(groundingdino_path, os.path.join(sample_dir, "groundingdino_annotated.jpg"))
                
    except Exception as e:
        logging.warning(f"Could not copy GroundingDINO image: {e}")

def create_enhanced_visualization(sample, sample_dir, category_name):
    """Create enhanced visualization keeping FDR format but adding OFA information"""
    try:
        # Check if GroundingDINO image exists
        groundingdino_path = os.path.join(sample_dir, "groundingdino_annotated.jpg")
        has_groundingdino = os.path.exists(groundingdino_path)
        
        if has_groundingdino:
            # 3-panel layout: Input, GroundingDINO, Enhanced Text (FDR + OFA)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        else:
            # 2-panel layout: Input, Enhanced Text (FDR + OFA)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 8))
            ax2 = None
        
        # Panel 1: Input image
        # Try to load from the copied input image first
        input_image_path = os.path.join(sample_dir, "input_image.jpg")
        image_loaded = False

        if os.path.exists(input_image_path):
            try:
                img = Image.open(input_image_path)
                ax1.imshow(img)
                ax1.set_title("Input Image", fontsize=14, fontweight='bold')
                ax1.axis('off')
                image_loaded = True
            except Exception as e:
                logging.warning(f"Could not load input image from {input_image_path}: {e}")

        # Fallback to original image_path if available
        if not image_loaded:
            image_path = sample.get('image_path', '')
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    ax1.imshow(img)
                    ax1.set_title("Input Image", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    image_loaded = True
                except Exception as e:
                    logging.warning(f"Could not load image from {image_path}: {e}")

        # Show "Image not available" if no image could be loaded
        if not image_loaded:
            ax1.text(0.5, 0.5, "Image not available", ha='center', va='center', fontsize=12)
            ax1.set_title("Input Image", fontsize=14, fontweight='bold')
            ax1.axis('off')
        
        # Panel 2: GroundingDINO (if available)
        if ax2 is not None:
            if has_groundingdino:
                grounding_img = Image.open(groundingdino_path)
                ax2.imshow(grounding_img)
                ax2.set_title("GroundingDINO Detection", fontsize=14, fontweight='bold')
                ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, "GroundingDINO not available", ha='center', va='center', fontsize=12)
                ax2.set_title("GroundingDINO Detection", fontsize=14, fontweight='bold')
                ax2.axis('off')
        
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
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(sample_dir, "output_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Created enhanced visualization: {output_path}")
        
    except Exception as e:
        logging.error(f"❌ Error creating enhanced visualization: {e}")

def main():
    """Main function to run FDR-OFA comparison"""
    print("🚀 Starting FDR vs OFA-X Comparison on 100 samples")
    print("-" * 60)

    # Step 1: Load OFA results
    ofa_results = load_ofa_results()
    if not ofa_results:
        print("❌ Cannot proceed without OFA results")
        return
    
    # Step 2: Run FDR pipeline
    print("\n" + "="*60)
    fdr_summary = run_fdr_pipeline()
    if not fdr_summary:
        print("❌ FDR pipeline failed")
        return
    
    # Step 3: Load FDR results
    print("\n" + "="*60)
    fdr_results = load_fdr_results()
    if not fdr_results:
        print("❌ No FDR results found")
        return
    
    # Step 4: Match samples
    print("\n" + "="*60)
    matched_samples = match_samples_by_question_id(fdr_results, ofa_results)
    if not matched_samples:
        print("❌ No matching samples found")
        return
    
    # Step 5: Categorize samples
    categories = categorize_samples(matched_samples)
    
    # Step 6: Create output directories
    print("\n" + "="*60)
    print("📁 Creating comparison output directories...")
    dirs = create_comparison_directories("comparison_fdr_ofa_output")
    
    # Step 7: Save samples to appropriate directories
    print("💾 Saving comparison samples...")
    total_saved = 0
    
    category_mapping = {
        'fdr_correct_ofa_correct': dirs[0],
        'fdr_wrong_ofa_wrong': dirs[1],
        'fdr_correct_ofa_wrong': dirs[2],
        'fdr_wrong_ofa_correct': dirs[3]
    }
    
    for category_name, samples in categories.items():
        if not samples:
            continue
            
        target_dir = category_mapping[category_name]
        print(f"\n🎨 Processing {category_name} samples...")
        
        for i, sample in enumerate(samples):
            sample_id = i + 1
            sample_dir = os.path.join(target_dir, f"sample_{sample_id}")
            save_comparison_sample(sample, sample_dir, category_name)
            total_saved += 1
        
        print(f"💾 Saved {len(samples)} {category_name} samples")
    
    # Step 8: Create summary
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
