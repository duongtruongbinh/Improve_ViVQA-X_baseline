#!/usr/bin/env python3
"""
CLEVR-X Multi-Reference Explanation Evaluation Script
Evaluates predicted explanations against each ground truth explanation separately,
then averages the scores as the final score for each question.
"""

import json
import logging
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def setup_logging():
    """Setup logging configuration"""
    log_file = "output/clevr_x_multi_ref_evaluation.log"
    os.makedirs("output", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_data():
    """Load both FDR results and ground truth data"""
    results_file = "/home/huytd/multi-agent/multi-agent/FDR/output/clevr_x_fdr_results.json"
    gt_file = "/mnt/VLAI_data/CLEVR-X/CLEVR_val_explanations_v0.7.10.json"
    
    logging.info("📥 Loading FDR results...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    logging.info("📥 Loading ground truth data...")
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    results = results_data.get('results', [])
    gt_questions = gt_data.get('questions', [])
    
    logging.info(f"✅ Loaded {len(results)} results and {len(gt_questions)} GT questions")
    return results, gt_questions

def import_evaluation_modules():
    """Import evaluation modules with proper error handling"""
    try:
        # Import NLTK modules
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        
        # Download required NLTK data with comprehensive coverage
        nltk_downloads = [
            'punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 
            'averaged_perceptron_tagger', 'stopwords'
        ]
        
        for resource in nltk_downloads:
            try:
                # Try to find the resource first
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                elif resource == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif resource == 'omw-1.4':
                    nltk.data.find('corpora/omw-1.4')
                elif resource == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
            except LookupError:
                logging.info(f"📥 Downloading NLTK resource: {resource}")
                try:
                    nltk.download(resource, quiet=True)
                    logging.info(f"✅ Successfully downloaded: {resource}")
                except Exception as download_error:
                    logging.warning(f"Failed to download {resource}: {download_error}")
        
        logging.info("✅ NLTK modules loaded successfully")
        nltk_available = True
    except ImportError as e:
        logging.warning(f"NLTK not available: {e}")
        nltk_available = False
    
    try:
        from rouge_score import rouge_scorer
        logging.info("✅ ROUGE-score loaded successfully")
        rouge_available = True
    except ImportError as e:
        logging.warning(f"ROUGE-score not available: {e}")
        logging.info("📥 Attempting to install rouge-score...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
            from rouge_score import rouge_scorer
            logging.info("✅ ROUGE-score installed and loaded successfully")
            rouge_available = True
        except Exception as install_error:
            logging.warning(f"Failed to install rouge-score: {install_error}")
            rouge_available = False
    
    try:
        from bert_score import score as bert_scorer
        logging.info("✅ BERTScore loaded successfully")
        bertscore_available = True
    except ImportError as e:
        logging.warning(f"BERTScore not available: {e}")
        logging.info("📥 Attempting to install bert-score...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score"])
            from bert_score import score as bert_scorer
            logging.info("✅ BERTScore installed and loaded successfully")
            bertscore_available = True
        except Exception as install_error:
            logging.warning(f"Failed to install bert-score: {install_error}")
            bertscore_available = False
    
    # Additional metric libraries
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        logging.info("✅ COCO evaluation metrics (CIDEr, SPICE) loaded successfully")
        coco_metrics_available = True
    except ImportError as e:
        logging.warning(f"COCO evaluation metrics not available: {e}")
        logging.info("📥 Attempting to install pycocoevalcap...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocoevalcap"])
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.spice.spice import Spice
            logging.info("✅ COCO evaluation metrics installed and loaded successfully")
            coco_metrics_available = True
        except Exception as install_error:
            logging.warning(f"Failed to install pycocoevalcap: {install_error}")
            coco_metrics_available = False
    
    return {
        'nltk_available': nltk_available,
        'rouge_available': rouge_available, 
        'bertscore_available': bertscore_available,
        'coco_metrics_available': coco_metrics_available
    }

def compute_bleu_scores(generated: str, reference: str) -> Dict[str, float]:
    """Compute BLEU scores for generated vs reference"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        generated_tokens = word_tokenize(generated.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        smoothing = SmoothingFunction().method1
        
        bleu_1 = sentence_bleu([reference_tokens], generated_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
    except Exception as e:
        logging.warning(f"BLEU computation failed: {e}")
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

def compute_rouge_scores(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L scores"""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        rouge_l = scores['rougeL']
        
        return {
            "rouge_l_precision": rouge_l.precision,
            "rouge_l_recall": rouge_l.recall,
            "rouge_l_f1": rouge_l.fmeasure
        }
    except Exception as e:
        logging.warning(f"ROUGE computation failed: {e}")
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

def compute_meteor_score(generated: str, reference: str) -> float:
    """Compute METEOR score"""
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        
        generated_tokens = word_tokenize(generated.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        meteor = meteor_score([reference_tokens], generated_tokens)
        return meteor
    except Exception as e:
        logging.warning(f"METEOR computation failed: {e}")
        return 0.0

def compute_bert_score(generated: str, reference: str, device: str = "cuda") -> Dict[str, float]:
    """Compute BERTScore"""
    try:
        from bert_score import score as bert_scorer
        
        # Set offline mode to avoid downloading models during evaluation
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online for model download if needed
        
        P, R, F1 = bert_scorer([generated], [reference], lang="en", 
                             model_type="bert-base-uncased", device=device, verbose=False)
        return {
            "bert_precision": P.item(),
            "bert_recall": R.item(), 
            "bert_f1": F1.item()
        }
    except Exception as e:
        logging.warning(f"BERTScore computation failed: {e}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

def compute_cider_score(generated: str, reference: str) -> float:
    """Compute CIDEr score"""
    try:
        from pycocoevalcap.cider.cider import Cider
        
        cider_scorer = Cider()
        
        # Format data for CIDEr (expects dict format)
        gts = {"0": [reference]}  # ground truth
        res = {"0": [generated]}  # results
        
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score)
    except Exception as e:
        logging.warning(f"CIDEr computation failed: {e}")
        return 0.0

def compute_spice_score(generated: str, reference: str) -> float:
    """Compute SPICE score"""
    try:
        from pycocoevalcap.spice.spice import Spice
        import sys
        import os
        from io import StringIO
        
        # Redirect stdout/stderr to suppress Java output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            spice_scorer = Spice()
            gts = {"0": [reference]}
            res = {"0": [generated]}
            score, _ = spice_scorer.compute_score(gts, res)
            return float(score)
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    except Exception as e:
        logging.warning(f"SPICE computation failed: {e}")
        return 0.0

def evaluate_single_pair(generated: str, reference: str, modules_available: Dict, device: str = "cuda") -> Dict[str, float]:
    """Evaluate a single generated-reference pair with all available metrics"""
    scores = {}
    
    # BLEU scores
    if modules_available['nltk_available']:
        bleu_scores = compute_bleu_scores(generated, reference)
        scores.update(bleu_scores)
    
    # ROUGE scores  
    if modules_available['rouge_available']:
        rouge_scores = compute_rouge_scores(generated, reference)
        scores.update(rouge_scores)
    
    # METEOR score
    if modules_available['nltk_available']:
        meteor = compute_meteor_score(generated, reference)
        scores['meteor'] = meteor
    
    # BERTScore
    if modules_available['bertscore_available']:
        bert_scores = compute_bert_score(generated, reference, device)
        scores.update(bert_scores)
    
    # CIDEr score
    if modules_available.get('coco_metrics_available', False):
        cider = compute_cider_score(generated, reference)
        scores['cider'] = cider
    
    # SPICE score
    if modules_available.get('coco_metrics_available', False):
        spice = compute_spice_score(generated, reference)
        scores['spice'] = spice
    
    return scores

def evaluate_multi_reference(generated: str, references: List[str], modules_available: Dict, device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate generated explanation against multiple references.
    Computes score with each reference separately, then averages.
    """
    if not generated or not references:
        return {}
    
    all_scores = []
    
    # Evaluate against each reference
    for ref in references:
        if ref.strip():  # Skip empty references
            scores = evaluate_single_pair(generated, ref, modules_available, device)
            all_scores.append(scores)
    
    if not all_scores:
        return {}
    
    # Average all scores
    averaged_scores = {}
    metric_names = all_scores[0].keys()
    
    for metric in metric_names:
        values = [score_dict[metric] for score_dict in all_scores if metric in score_dict]
        if values:
            averaged_scores[metric] = np.mean(values)
            averaged_scores[f"{metric}_std"] = np.std(values)
            averaged_scores[f"{metric}_max"] = np.max(values)
            averaged_scores[f"{metric}_min"] = np.min(values)
    
    averaged_scores['num_references'] = len(all_scores)
    
    return averaged_scores

def run_evaluation(results: List[Dict], gt_questions: List[Dict], num_samples: int = 5000):
    """Run multi-reference evaluation on CLEVR-X data"""
    
    logging.info("🔬 Starting multi-reference explanation evaluation...")
    
    # Check available modules
    modules_available = import_evaluation_modules()
    
    # Determine device
    device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    logging.info(f"🔧 Using device: {device}")
    
    # Evaluation results
    sample_results = []
    all_averaged_scores = defaultdict(list)
    
    # Process each sample
    num_samples = min(num_samples, len(results), len(gt_questions))
    logging.info(f"📊 Evaluating {num_samples} samples...")
    
    for i in range(num_samples):
        if i % 500 == 0:
            logging.info(f"Progress: {i}/{num_samples}")
        
        result = results[i]
        gt_question = gt_questions[i]
        
        # Get generated and reference explanations
        generated_explanation = result.get('predicted_explanation', '').strip()
        reference_explanations = gt_question.get('factual_explanation', [])
        
        if not generated_explanation or not reference_explanations:
            continue
        
        # Evaluate with multi-reference approach
        averaged_scores = evaluate_multi_reference(
            generated_explanation, reference_explanations, modules_available, device
        )
        
        if averaged_scores:
            # Store sample result
            sample_result = {
                'sample_index': i,
                'question': result.get('question', ''),
                'generated_explanation': generated_explanation,
                'reference_explanations': reference_explanations,
                'num_references': len(reference_explanations),
                'scores': averaged_scores
            }
            sample_results.append(sample_result)
            
            # Collect scores for overall statistics
            for metric, score in averaged_scores.items():
                if not metric.endswith('_std') and not metric.endswith('_max') and not metric.endswith('_min') and metric != 'num_references':
                    all_averaged_scores[metric].append(score)
    
    # Compute overall statistics
    overall_metrics = {}
    for metric, scores in all_averaged_scores.items():
        if scores:
            overall_metrics[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
    
    logging.info(f"✅ Evaluated {len(sample_results)} samples successfully")
    
    return {
        'overall_metrics': overall_metrics,
        'sample_results': sample_results,
        'evaluation_info': {
            'total_samples': num_samples,
            'successful_evaluations': len(sample_results),
            'modules_available': modules_available,
            'device_used': device,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }

def save_results(evaluation_results: Dict, output_file: str):
    """Save evaluation results to JSON file"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare compact results (only first 100 detailed samples)
    compact_results = {
        'evaluation_info': evaluation_results['evaluation_info'],
        'overall_metrics': evaluation_results['overall_metrics'],
        'detailed_samples': evaluation_results['sample_results'][:100]  # First 100 samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Results saved to: {output_file}")

def print_summary(evaluation_results: Dict):
    """Print evaluation summary"""
    
    overall_metrics = evaluation_results['overall_metrics']
    eval_info = evaluation_results['evaluation_info']
    
    logging.info("=" * 70)
    logging.info("🎯 CLEVR-X MULTI-REFERENCE EVALUATION RESULTS")
    logging.info("=" * 70)
    
    logging.info(f"📊 Evaluation Info:")
    logging.info(f"  Total samples: {eval_info['total_samples']}")
    logging.info(f"  Successful evaluations: {eval_info['successful_evaluations']}")
    logging.info(f"  Device used: {eval_info['device_used']}")
    
    logging.info(f"\n📈 Overall Metrics (averaged across multiple references):")
    
    # BLEU scores
    if 'bleu_4' in overall_metrics:
        logging.info("  BLEU Scores:")
        for i in range(1, 5):
            metric = f'bleu_{i}'
            if metric in overall_metrics:
                score = overall_metrics[metric]['mean']
                logging.info(f"    BLEU-{i}: {score:.4f}")
    
    # ROUGE scores
    if 'rouge_l_f1' in overall_metrics:
        logging.info("  ROUGE-L Scores:")
        for metric in ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1']:
            if metric in overall_metrics:
                score = overall_metrics[metric]['mean']
                name = metric.replace('rouge_l_', '').title()
                logging.info(f"    {name}: {score:.4f}")
    
    # METEOR score
    if 'meteor' in overall_metrics:
        score = overall_metrics['meteor']['mean']
        logging.info(f"  METEOR: {score:.4f}")
    
    # BERTScore
    if 'bert_f1' in overall_metrics:
        logging.info("  BERTScore:")
        for metric in ['bert_precision', 'bert_recall', 'bert_f1']:
            if metric in overall_metrics:
                score = overall_metrics[metric]['mean']
                name = metric.replace('bert_', '').title()
                logging.info(f"    {name}: {score:.4f}")
    
    # CIDEr score
    if 'cider' in overall_metrics:
        score = overall_metrics['cider']['mean']
        logging.info(f"  CIDEr: {score:.4f}")
    
    # SPICE score
    if 'spice' in overall_metrics:
        score = overall_metrics['spice']['mean']
        logging.info(f"  SPICE: {score:.4f}")
    
    logging.info("=" * 70)

def main():
    """Main function"""
    setup_logging()
    
    logging.info("🚀 CLEVR-X Multi-Reference Explanation Evaluation")
    logging.info("⚠️ Make sure to activate 'ma_vqa' environment before running!")
    logging.info("=" * 70)
    
    try:
        # Load data
        results, gt_questions = load_data()
        
        # Run evaluation
        evaluation_results = run_evaluation(results, gt_questions, num_samples=5000)
        
        # Save results
        output_file = "output/clevr_x_multi_ref_evaluation_results.json"
        save_results(evaluation_results, output_file)
        
        # Print summary
        print_summary(evaluation_results)
        
        logging.info("🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
