#!/usr/bin/env python3
# FDR/quick_run_from_results.py - Quick visualization from existing results
import os
import sys
import json
import re
from pathlib import Path
import numpy as np

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

def load_results():
    """Load FDR and ReRe results from JSON files"""
    # File paths
    fdr_results_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/output/fdr_results.json"
    rere_results_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/FDR/inference_results_100_samples.json"
    
    print("📂 Loading results from files...")
    print(f"  FDR Results: {fdr_results_path}")
    print(f"  ReRe Results: {rere_results_path}")
    
    # Load FDR results
    try:
        with open(fdr_results_path, 'r', encoding='utf-8') as f:
            fdr_results = json.load(f)
        print(f"✅ Loaded {len(fdr_results)} FDR samples")
    except Exception as e:
        print(f"❌ Error loading FDR results: {e}")
        return None, None
    
    # Load ReRe results
    try:
        with open(rere_results_path, 'r', encoding='utf-8') as f:
            rere_results = json.load(f)
        print(f"✅ Loaded {len(rere_results)} ReRe samples")
    except Exception as e:
        print(f"❌ Error loading ReRe results: {e}")
        return fdr_results, None
    
    return fdr_results, rere_results

def match_samples(fdr_results, rere_results):
    """Match FDR and ReRe samples by question_id"""
    print("🔗 Matching samples by question_id...")
    
    # Create lookup dictionaries
    fdr_lookup = {}
    for sample in fdr_results:
        question_id = str(sample.get('question_id', ''))
        fdr_lookup[question_id] = sample
    
    rere_lookup = {}
    for sample in rere_results:
        question_id = str(sample.get('question_id', ''))
        rere_lookup[question_id] = sample
    
    # Find common samples
    common_ids = set(fdr_lookup.keys()) & set(rere_lookup.keys())
    print(f"📊 Found {len(common_ids)} common samples")
    
    matched_samples = []
    for question_id in sorted(common_ids):
        fdr_sample = fdr_lookup[question_id]
        rere_sample = rere_lookup[question_id]
        
        matched_samples.append({
            'question_id': question_id,
            'fdr': fdr_sample,
            'rere': rere_sample
        })
    
    return matched_samples

def categorize_samples(matched_samples):
    """Categorize samples into correct/wrong for both models"""
    print("📋 Categorizing samples...")
    
    both_correct = []
    both_wrong = []
    fdr_better = []
    rere_better = []
    
    for sample in matched_samples:
        fdr_sample = sample['fdr']
        rere_sample = sample['rere']
        
        # Normalize answers for comparison
        fdr_answer = str(fdr_sample.get('final_answer', '')).lower().strip()
        rere_answer = str(rere_sample.get('predicted_answer', '')).lower().strip()
        
        # Get ground truth (prefer from FDR as it might be more processed)
        ground_truth = str(fdr_sample.get('ground_truth', '')).lower().strip()
        if not ground_truth:
            ground_truth = str(rere_sample.get('ground_truth_answer', '')).lower().strip()
        
        # Check correctness
        fdr_correct = _is_answer_correct(fdr_answer, ground_truth)
        rere_correct = _is_answer_correct(rere_answer, ground_truth)
        
        # Categorize
        if fdr_correct and rere_correct:
            both_correct.append(sample)
        elif not fdr_correct and not rere_correct:
            both_wrong.append(sample)
        elif fdr_correct and not rere_correct:
            fdr_better.append(sample)
        elif not fdr_correct and rere_correct:
            rere_better.append(sample)
    
    print(f"✅ Both Correct: {len(both_correct)}")
    print(f"❌ Both Wrong: {len(both_wrong)}")
    print(f"🟢 FDR Better: {len(fdr_better)}")
    print(f"🔵 ReRe Better: {len(rere_better)}")
    
    return {
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'fdr_better': fdr_better,
        'rere_better': rere_better
    }

def _is_answer_correct(predicted, ground_truth):
    """Check if predicted answer matches ground truth"""
    if not predicted or not ground_truth:
        return False
    
    # Simple string matching with normalization
    predicted = predicted.lower().strip()
    ground_truth = ground_truth.lower().strip()
    
    return (predicted == ground_truth or 
            ground_truth in predicted or 
            predicted in ground_truth)

def create_output_directories(base_dir="results_visualization"):
    """Create directories for visualization output"""
    correct_dir = os.path.join(base_dir, "correct")
    wrong_dir = os.path.join(base_dir, "wrong")
    fdr_better_dir = os.path.join(base_dir, "fdr_better")
    rere_better_dir = os.path.join(base_dir, "rere_better")
    
    for dir_path in [correct_dir, wrong_dir, fdr_better_dir, rere_better_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return correct_dir, wrong_dir, fdr_better_dir, rere_better_dir

def quick_run_from_results():
    """Main function to run visualization from existing results"""
    print("🚀 Quick Run - Visualization from Results")
    print("📊 Loading FDR vs ReRe comparison results")
    print("📁 Output: results_visualization")
    print("-" * 50)
    
    try:
        # Load results
        fdr_results, rere_results = load_results()
        if not fdr_results or not rere_results:
            print("❌ Failed to load results")
            return None
        
        # Match samples
        matched_samples = match_samples(fdr_results, rere_results)
        if not matched_samples:
            print("❌ No matching samples found")
            return None
        
        # Categorize samples
        categories = categorize_samples(matched_samples)
        
        # Create output directories
        correct_dir, wrong_dir, fdr_better_dir, rere_better_dir = create_output_directories()
        
        print(f"\n📁 Created output directories:")
        print(f"  - Both correct: {correct_dir}")
        print(f"  - Both wrong: {wrong_dir}")
        print(f"  - FDR better: {fdr_better_dir}")
        print(f"  - ReRe better: {rere_better_dir}")
        
        # Save samples and create visualizations
        total_saved = 0
        for category_name, samples in categories.items():
            if category_name == 'both_correct':
                target_dir = correct_dir
                max_samples = 3
            elif category_name == 'both_wrong':
                target_dir = wrong_dir
                max_samples = 3
            elif category_name == 'fdr_better':
                target_dir = fdr_better_dir
                max_samples = 3
            elif category_name == 'rere_better':
                target_dir = rere_better_dir
                max_samples = 3
            
            saved_count = save_category_samples(samples[:max_samples], target_dir, category_name)
            total_saved += saved_count
            print(f"💾 Saved {saved_count}/{len(samples)} {category_name} samples")
        
        # Create summary
        summary = create_summary(categories, len(matched_samples))

        print(f"\n🎉 Visualization completed successfully!")
        print(f"📊 Results:")
        print(f"  - Total processed: {summary['total_processed']}")
        print(f"  - Both correct: {summary['both_correct']}")
        print(f"  - Both wrong: {summary['both_wrong']}")
        print(f"  - FDR better: {summary['fdr_better']}")
        print(f"  - ReRe better: {summary['rere_better']}")
        print(f"📁 Output directory: results_visualization")

        # Show evidence and hypothesis details (following quick_run.py format)
        _show_comparison_summary(summary, categories)

        # Show next steps
        print(f"\n📋 Next steps:")
        print(f"  - View results: python view_results_comparison.py")
        print(f"  - List samples: python view_results_comparison.py --list")
        print(f"  - Show sample: python view_results_comparison.py --show correct:1")
        print(f"  - Show images: python view_results_comparison.py --images fdr_better:1")
        print(f"  - Analyze: python view_results_comparison.py --analyze")

        return summary
        
    except Exception as e:
        print(f"❌ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_category_samples(samples, target_dir, category_name):
    """Save samples in a category with visualizations"""
    saved_count = 0

    for i, sample in enumerate(samples):
        sample_id = i + 1
        sample_dir = os.path.join(target_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        try:
            # Create combined sample info
            combined_info = create_combined_sample_info(sample, category_name)

            # Save info.json
            info_file = os.path.join(sample_dir, "info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(combined_info, f, indent=2, ensure_ascii=False)

            # Copy input image if available
            copy_input_image(sample, sample_dir)

            # Create output visualization (following quick_run.py naming)
            create_comparison_visualization(sample, sample_dir, category_name)

            saved_count += 1

        except Exception as e:
            print(f"⚠️ Error saving sample {sample_id} in {category_name}: {e}")
            continue

    return saved_count

def create_combined_sample_info(sample, category_name):
    """Create combined sample info from FDR and ReRe results (following quick_run.py format)"""
    fdr_sample = sample['fdr']
    rere_sample = sample['rere']

    # Extract question (handle both string and list formats)
    question = fdr_sample.get('question', '')
    if isinstance(question, list):
        question = question[0] if question else ''

    rere_question = rere_sample.get('question', '')
    if isinstance(rere_question, list):
        rere_question = rere_question[0] if rere_question else ''

    # Use the more complete question
    final_question = question if len(question) > len(rere_question) else rere_question

    # Ground truth
    ground_truth = fdr_sample.get('ground_truth', '') or rere_sample.get('ground_truth_answer', '')

    # Create info in the same format as quick_run.py auto_save_pipeline
    combined_info = {
        'id': sample['question_id'],
        'question': final_question,
        'answer': fdr_sample.get('final_answer', ''),  # Primary answer (FDR)
        'ground_truth': ground_truth,
        'explanation': fdr_sample.get('explanation', ''),  # Primary explanation (FDR)
        'evidence': fdr_sample.get('causal_trace', []),  # FDR evidence
        'hypothesis': f"FDR: {fdr_sample.get('synthesis_status', '')}",  # FDR hypothesis status
        'synthesis_status': fdr_sample.get('synthesis_status', ''),

        # Additional comparison data
        'category': category_name,
        'fdr_answer': fdr_sample.get('final_answer', ''),
        'fdr_explanation': fdr_sample.get('explanation', ''),
        'fdr_synthesis_status': fdr_sample.get('synthesis_status', ''),
        'fdr_evidence_count': fdr_sample.get('evidence_count', 0),
        'fdr_hypothesis_count': fdr_sample.get('hypothesis_count', 0),
        'fdr_causal_trace': fdr_sample.get('causal_trace', []),

        # ReRe results (NEW - as requested)
        'rere_answer': rere_sample.get('predicted_answer', ''),
        'rere_explanation': rere_sample.get('predicted_explanation', ''),
        'rere_full_caption': rere_sample.get('full_caption', ''),

        # Ground truth explanations
        'ground_truth_explanations': (fdr_sample.get('ground_truth_explanations', []) or
                                    rere_sample.get('ground_truth_explanations', [])),

        # Image info
        'image_path': fdr_sample.get('image_path', ''),

        # Correctness
        'fdr_correct': _is_answer_correct(
            str(fdr_sample.get('final_answer', '')).lower().strip(),
            str(ground_truth).lower().strip()
        ),
        'rere_correct': _is_answer_correct(
            str(rere_sample.get('predicted_answer', '')).lower().strip(),
            str(ground_truth).lower().strip()
        )
    }

    return combined_info

def copy_input_image(sample, sample_dir):
    """Copy input image to sample directory"""
    try:
        from PIL import Image

        image_path = sample['fdr'].get('image_path', '')
        if image_path and os.path.exists(image_path):
            input_img = Image.open(image_path)
            input_img.save(os.path.join(sample_dir, "input_image.jpg"))
        else:
            print(f"⚠️ Image not found: {image_path}")
    except Exception as e:
        print(f"⚠️ Could not copy image: {e}")

def create_comparison_visualization(sample, sample_dir, category_name):
    """Create a visualization in the style of quick_run.py with FDR and ReRe comparison"""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        fdr_sample = sample['fdr']
        rere_sample = sample['rere']

        # Check for GroundingDINO image in multiple possible locations
        image_path = fdr_sample.get('image_path', '')
        groundingdino_path = None

        if image_path:
            input_dir = os.path.dirname(image_path)
            input_filename = os.path.basename(image_path)

            # Try multiple naming patterns for GroundingDINO files
            possible_groundingdino_names = [
                f"groundingdino_{input_filename}",
                f"grounding_dino_{input_filename}",
                f"{os.path.splitext(input_filename)[0]}_grounding.jpg",
                f"{os.path.splitext(input_filename)[0]}_groundingdino.jpg",
                f"annotated_{input_filename}",
                f"detection_{input_filename}"
            ]

            # Check in the same directory as input image
            for gd_name in possible_groundingdino_names:
                potential_path = os.path.join(input_dir, gd_name)
                if os.path.exists(potential_path):
                    groundingdino_path = potential_path
                    break

            # If not found, check in common GroundingDINO output directories
            if not groundingdino_path:
                common_gd_dirs = [
                    os.path.join(input_dir, "groundingdino_output"),
                    os.path.join(input_dir, "grounding_dino"),
                    os.path.join(input_dir, "annotations"),
                    os.path.join(os.path.dirname(input_dir), "groundingdino_output"),
                    "/home/khuonghuynh/Improve_ViVQA-X_baseline/GroundingDINO/outputs"
                ]

                for gd_dir in common_gd_dirs:
                    if os.path.exists(gd_dir):
                        for gd_name in possible_groundingdino_names:
                            potential_path = os.path.join(gd_dir, gd_name)
                            if os.path.exists(potential_path):
                                groundingdino_path = potential_path
                                break
                        if groundingdino_path:
                            break

            # Copy GroundingDINO image to sample directory if found
            if groundingdino_path:
                try:
                    groundingdino_img = Image.open(groundingdino_path)
                    groundingdino_img.save(os.path.join(sample_dir, "groundingdino_annotated.jpg"))
                    print(f"✅ Found and copied GroundingDINO image: {groundingdino_path}")
                except Exception as e:
                    print(f"⚠️ Error copying GroundingDINO image: {e}")

        # Check if we have GroundingDINO in sample directory
        sample_groundingdino_path = os.path.join(sample_dir, "groundingdino_annotated.jpg")
        has_groundingdino = os.path.exists(sample_groundingdino_path)

        # If no GroundingDINO found, try to generate one using VerifierAgent (like quick_run.py)
        if not has_groundingdino and image_path and os.path.exists(image_path):
            has_groundingdino = generate_groundingdino_with_verifier(image_path, sample_dir, fdr_sample, rere_sample)

        if has_groundingdino:
            # 3-panel layout: Input, GroundingDINO, Text (like original quick_run.py)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        else:
            # 2-panel layout: Input, Text
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 8))
            ax2 = None

        # Left panel: Input image
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                ax1.imshow(img)
                ax1.set_title("Input Image", fontsize=14, fontweight='bold')
                ax1.axis('off')
            except Exception as e:
                ax1.text(0.5, 0.5, f"Image not available\n{str(e)}",
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title("Input Image (Error)", fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "Image not found",
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Input Image (Not Found)", fontsize=14, fontweight='bold')

        # Middle panel: GroundingDINO annotated image (if available)
        if has_groundingdino and ax2 is not None:
            try:
                groundingdino_img = Image.open(sample_groundingdino_path)
                ax2.imshow(groundingdino_img)
                ax2.set_title("GroundingDINO Detection", fontsize=14, fontweight='bold')
                ax2.axis('off')
            except Exception as e:
                ax2.text(0.5, 0.5, f"GroundingDINO image error\n{str(e)}",
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("GroundingDINO (Error)", fontsize=14, fontweight='bold')

        # Right panel: Text information (following quick_run.py format)
        ax3.axis('off')

        # Question
        question = fdr_sample.get('question', '')
        if isinstance(question, list):
            question = question[0] if question else ''
        ax3.text(0.05, 0.98, f"Question: {question}",
                fontsize=10, fontweight='bold', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # Ground Truth
        ground_truth = fdr_sample.get('ground_truth', '') or rere_sample.get('ground_truth_answer', '')
        ax3.text(0.05, 0.93, f"Ground Truth: {ground_truth}",
                fontsize=10, fontweight='bold', color='blue', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # FDR Answer
        fdr_answer = fdr_sample.get('final_answer', '')
        ax3.text(0.05, 0.88, f"FDR Answer: {fdr_answer}",
                fontsize=10, fontweight='bold', color='green', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # ReRe Answer (NEW - added as requested)
        rere_answer = rere_sample.get('predicted_answer', '')
        ax3.text(0.05, 0.83, f"ReRe Answer: {rere_answer}",
                fontsize=10, fontweight='bold', color='purple', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # Status
        status = fdr_sample.get('synthesis_status', '')
        ax3.text(0.05, 0.78, f"FDR Status: {status}",
                fontsize=10, color='orange', transform=ax3.transAxes,
                verticalalignment='top')

        # Category
        category_colors = {
            'both_correct': 'green',
            'both_wrong': 'red',
            'fdr_better': 'darkgreen',
            'rere_better': 'purple'
        }
        ax3.text(0.05, 0.73, f"Category: {category_name.replace('_', ' ').title()}",
                fontsize=10, color=category_colors.get(category_name, 'black'),
                transform=ax3.transAxes, verticalalignment='top')

        # FDR Explanation (following quick_run.py format)
        fdr_explanation = fdr_sample.get('explanation', '')
        ax3.text(0.05, 0.68, f"FDR Explanation: {fdr_explanation}",
                fontsize=9, color='darkgreen', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # ReRe Explanation (NEW - added as requested)
        rere_explanation = rere_sample.get('predicted_explanation', '')
        ax3.text(0.05, 0.58, f"ReRe Explanation: {rere_explanation}",
                fontsize=9, color='purple', transform=ax3.transAxes,
                verticalalignment='top', wrap=True)

        # Evidence Details (following quick_run.py format)
        # Note: ReRe doesn't have evidence_set, so we show FDR's evidence
        # This maintains the original quick_run.py structure
        evidence_count = fdr_sample.get('evidence_count', 0)
        hypothesis_count = fdr_sample.get('hypothesis_count', 0)

        if evidence_count > 0:
            evidence_text = "FDR Evidence Details:\n"
            # Try to get evidence from causal_trace or other fields
            causal_trace = fdr_sample.get('causal_trace', [])
            if causal_trace:
                for i, trace in enumerate(causal_trace[:3]):  # Show first 3
                    evidence_id = trace.get('hypothesis_id', f'E{i+1}')
                    reasoning = trace.get('reasoning_description', 'N/A')
                    evidence_text += f"  • {evidence_id}: {reasoning}\n"
            else:
                evidence_text += f"  • Evidence count: {evidence_count}\n"

            ax3.text(0.05, 0.48, evidence_text,
                    fontsize=9, color='brown', transform=ax3.transAxes,
                    verticalalignment='top')

        # Hypothesis Details (following quick_run.py format)
        if hypothesis_count > 0:
            hypothesis_text = "FDR Hypothesis Details:\n"
            if causal_trace:
                for i, trace in enumerate(causal_trace[:2]):  # Show first 2
                    hyp_id = trace.get('hypothesis_id', f'H{i+1}')
                    confidence = trace.get('confidence_source', 0)
                    hypothesis_text += f"  • {hyp_id}: Confidence {confidence:.2f}\n"
            else:
                hypothesis_text += f"  • Hypothesis count: {hypothesis_count}\n"

            # Position hypothesis below evidence (dynamic positioning)
            y_pos = 0.35 if evidence_count > 0 else 0.48
            ax3.text(0.05, y_pos, hypothesis_text,
                    fontsize=9, color='brown', transform=ax3.transAxes,
                    verticalalignment='top')

        # Summary counts at bottom (following quick_run.py format)
        summary_y_pos = 0.25 if (evidence_count > 0 or hypothesis_count > 0) else 0.48
        ax3.text(0.05, summary_y_pos, f"FDR Total: {evidence_count} Evidence, {hypothesis_count} Hypothesis",
                fontsize=9, color='gray', fontweight='bold', transform=ax3.transAxes,
                verticalalignment='top')

        # Correctness indicators (without emojis to avoid font issues)
        fdr_correct = _is_answer_correct(
            str(fdr_sample.get('final_answer', '')).lower().strip(),
            str(ground_truth).lower().strip()
        )
        rere_correct = _is_answer_correct(
            str(rere_sample.get('predicted_answer', '')).lower().strip(),
            str(ground_truth).lower().strip()
        )

        correctness_y = summary_y_pos - 0.05
        ax3.text(0.05, correctness_y, f"FDR: {'CORRECT' if fdr_correct else 'WRONG'} | ReRe: {'CORRECT' if rere_correct else 'WRONG'}",
                fontsize=10, fontweight='bold',
                color='green' if (fdr_correct and rere_correct) else 'red' if (not fdr_correct and not rere_correct) else 'orange',
                transform=ax3.transAxes, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "output_visualization.png"),  # Keep original filename
                    dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not create comparison visualization: {e}")

def generate_groundingdino_with_verifier(image_path, sample_dir, fdr_sample, rere_sample):
    """Generate GroundingDINO annotation using VerifierAgent (like quick_run.py)"""
    try:
        # Import VerifierAgent and config
        from src.agents.verifier import VerifierAgent

        # Extract question for VerifierAgent
        question = fdr_sample.get('question', '')
        if isinstance(question, list):
            question = question[0] if question else ''

        print(f"🔍 Using VerifierAgent to generate GroundingDINO for: {question[:50]}...")

        # Load config to get VerifierAgent settings (like in auto_save_pipeline.py)
        try:
            import yaml
            # Config is in FDR directory
            fdr_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(fdr_dir, 'config.yaml')
            print(f"🔧 Loading config from: {config_path}")

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            agents_config = config.get('agents_config', {})
            verifier_config = agents_config.get('verifier', {})
            print(f"🔧 Verifier config: {verifier_config}")
        except Exception as e:
            print(f"⚠️ Could not load config, using defaults: {e}")
            verifier_config = {}

        # Initialize VerifierAgent with GroundingDINO enabled (same as auto_save_pipeline.py)
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=True,
            enable_dam=verifier_config.get('enable_dam', True),
            groundingdino_docker=verifier_config.get('groundingdino_docker', False),
            model_preference="vlm"
        )

        # Manually enable GroundingDINO if config says so (override hardcoded disable)
        if verifier_config.get('enable_groundingdino', False):
            print(f"🔧 Manually enabling GroundingDINO from config...")
            verifier.groundingdino_enabled = True
            # Try to initialize GroundingDINO
            try:
                verifier._initialize_groundingdino()
                print(f"✅ GroundingDINO initialized successfully")
            except Exception as e:
                print(f"⚠️ GroundingDINO initialization failed: {e}")
                verifier.groundingdino_enabled = False

        # Check if GroundingDINO is actually enabled
        if not verifier.groundingdino_enabled:
            print(f"⚠️ GroundingDINO is disabled in VerifierAgent")
            return False

        # Generate initial response (this will create GroundingDINO if enabled)
        initial_response = verifier.generate_initial_response(question, image_path)

        # Check if GroundingDINO was created (same logic as auto_save_pipeline.py)
        input_dir = os.path.dirname(image_path)
        input_filename = os.path.basename(image_path)
        potential_groundingdino_path = os.path.join(input_dir, f"groundingdino_{input_filename}")

        if os.path.exists(potential_groundingdino_path):
            # Copy GroundingDINO image to sample directory (same as auto_save_pipeline.py)
            try:
                from PIL import Image
                groundingdino_img = Image.open(potential_groundingdino_path)
                groundingdino_img.save(os.path.join(sample_dir, "groundingdino_annotated.jpg"))
                print(f"✅ VerifierAgent generated GroundingDINO: {potential_groundingdino_path}")
                return True
            except Exception as e:
                print(f"⚠️ Error copying VerifierAgent GroundingDINO: {e}")
                return False
        else:
            print(f"⚠️ VerifierAgent did not generate GroundingDINO (may be disabled)")
            return False

    except Exception as e:
        print(f"⚠️ VerifierAgent GroundingDINO generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# GroundingDINO functions removed - now using VerifierAgent like quick_run.py

def create_summary(categories, total_samples):
    """Create summary of the visualization results"""
    summary = {
        'total_processed': total_samples,
        'both_correct': len(categories['both_correct']),
        'both_wrong': len(categories['both_wrong']),
        'fdr_better': len(categories['fdr_better']),
        'rere_better': len(categories['rere_better']),
        'fdr_accuracy': (len(categories['both_correct']) + len(categories['fdr_better'])) / total_samples if total_samples > 0 else 0,
        'rere_accuracy': (len(categories['both_correct']) + len(categories['rere_better'])) / total_samples if total_samples > 0 else 0,
        'agreement_rate': len(categories['both_correct']) / total_samples if total_samples > 0 else 0
    }

    # Save summary
    summary_file = os.path.join("results_visualization", "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary

def _show_comparison_summary(summary, categories):
    """Show comparison summary details (following quick_run.py format)"""
    print(f"\n🔍 FDR vs ReRe Comparison Summary:")

    # Show sample details for each category
    category_names = {
        'both_correct': 'Both Correct',
        'both_wrong': 'Both Wrong',
        'fdr_better': 'FDR Better',
        'rere_better': 'ReRe Better'
    }

    for category_key, samples in categories.items():
        if not samples:
            continue

        category_name = category_names.get(category_key, category_key)
        print(f"\n  {category_name}: {len(samples)} samples")

        # Show first 3 samples in each category
        for i, sample in enumerate(samples[:3]):
            fdr_sample = sample['fdr']
            rere_sample = sample['rere']

            print(f"    Sample {i+1}:")

            # Question
            question = fdr_sample.get('question', '')
            if isinstance(question, list):
                question = question[0] if question else ''
            print(f"      Question: {question[:60]}...")

            # Answers
            fdr_answer = fdr_sample.get('final_answer', '')
            rere_answer = rere_sample.get('predicted_answer', '')
            ground_truth = fdr_sample.get('ground_truth', '') or rere_sample.get('ground_truth_answer', '')

            print(f"      FDR Answer: {fdr_answer}")
            print(f"      ReRe Answer: {rere_answer}")
            print(f"      Ground Truth: {ground_truth}")

            # Evidence and hypothesis counts (FDR only)
            evidence_count = fdr_sample.get('evidence_count', 0)
            hypothesis_count = fdr_sample.get('hypothesis_count', 0)
            print(f"      FDR Evidence: {evidence_count}, Hypothesis: {hypothesis_count}")

            if i < len(samples) - 1 and i < 2:  # Don't add extra line after last sample
                print()

    # Show overall statistics
    total_samples = summary['total_processed']
    if total_samples > 0:
        print(f"\n  📈 Performance Metrics:")
        print(f"    - FDR Accuracy: {summary['fdr_accuracy']:.1%}")
        print(f"    - ReRe Accuracy: {summary['rere_accuracy']:.1%}")
        print(f"    - Agreement Rate: {summary['agreement_rate']:.1%}")

        # Calculate improvement
        fdr_correct = summary['both_correct'] + summary['fdr_better']
        rere_correct = summary['both_correct'] + summary['rere_better']
        improvement = fdr_correct - rere_correct

        print(f"    - FDR vs ReRe: +{improvement} samples ({improvement/total_samples:.1%} improvement)")

if __name__ == "__main__":
    quick_run_from_results()
