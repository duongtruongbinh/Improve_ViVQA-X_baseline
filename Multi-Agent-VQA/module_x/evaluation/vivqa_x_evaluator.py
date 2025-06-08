#!/usr/bin/env python3
"""
Evaluation metrics for ViVQA-X task
Computes BLEU-1,2,3,4, METEOR, ROUGE-L, CIDEr, SPICE, BERTScore
"""

import json
import re
from typing import Dict, List, Any
import os
import string

# Try to import evaluation libraries
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    PYCOCO_AVAILABLE = True
    print("✓ pycocoevalcap libraries loaded successfully")
except ImportError as e:
    print(f"Warning: pycocoevalcap not available: {e}")
    print("Please install: pip install pycocoevalcap")
    PYCOCO_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
    print("✓ BERTScore loaded successfully")
except ImportError as e:
    print(f"Warning: bert-score not available: {e}")
    print("Please install: pip install bert-score")
    BERT_SCORE_AVAILABLE = False

class ViVQAXEvaluator:
    """
    Evaluator for ViVQA-X task (Vietnamese VQA with Explanations)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def clean_text(self, text) -> str:
        """Clean text for evaluation"""
        # Handle case where text might be a list
        if isinstance(text, list):
            # If it's a list, take the first element or return empty string
            text = text[0] if text else ""
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text for more robust comparison."""
        # Handle case where text might be a list
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str):
            text = str(text)

        # Lowercase, remove punctuation, strip whitespace
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text

    def _is_answer_correct(self, pred: str, target: str) -> bool:
        """
        Checks if a predicted answer is correct, with more flexible matching.
        """
        pred_norm = self._normalize_answer(pred)
        target_norm = self._normalize_answer(target)

        # 1. Direct match
        if pred_norm == target_norm:
            return True

        # 2. Synonym matching for common answers (yes/no)
        yes_synonyms = {"có", "đúng", "vâng", "phải", "yes"}
        no_synonyms = {"không", "sai", "no"}

        if target_norm in yes_synonyms and pred_norm in yes_synonyms:
            return True
        
        if target_norm in no_synonyms and pred_norm in no_synonyms:
            return True
            
        # 3. Handle Vietnamese classifiers
        classifiers = ["con", "cái", "chiếc", "quả", "bông", "hoa", "người", "xe"]
        
        pred_words = pred_norm.split()
        target_words = target_norm.split()

        # Handle classifier at the beginning of the target answer
        if len(target_words) > 1 and target_words[0] in classifiers:
            if " ".join(target_words[1:]) == pred_norm:
                return True

        # Handle classifier at the beginning of the predicted answer
        if len(pred_words) > 1 and pred_words[0] in classifiers:
            if " ".join(pred_words[1:]) == target_norm:
                return True
                
        # 4. Handle word order for short answers
        if sorted(pred_words) == sorted(target_words):
            return True
            
        # 5. Handle cases like target="không" and pred="không yên tĩnh" (not calm)
        if target_norm == "không" and pred_norm.startswith("không "):
            return True
            
        # 6. Basic English to Vietnamese mapping for common VQA terms
        translation_map = {
            "baseball": "bóng chày",
            "tennis": "quần vợt",
        }
        if translation_map.get(pred_norm, None) == target_norm:
            return True

        return False
    
    def get_nlg_scores(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute NLG scores
        
        Args:
            references: List of lists of ground truth explanations
            hypotheses: List of predicted explanations
            
        Returns:
            Dictionary of scores
        """
        
        if not PYCOCO_AVAILABLE or not BERT_SCORE_AVAILABLE:
            print("Warning: Some evaluation libraries not available")
            return self._basic_scores(references, hypotheses)
        
        # Convert to format expected by pycocoevalcap
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        scores = {}
        
        try:
            # Initialize scorers
            scorers = [
                (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                (Spice(), "SPICE")
            ]
            
            # Compute scores
            for scorer, method in scorers:
                try:
                    score, _ = scorer.compute_score(gts, res)
                    if isinstance(method, list):
                        for m, s in zip(method, score):
                            scores[m] = s
                    else:
                        scores[method] = score
                except Exception as e:
                    print(f"Warning: Failed to compute {method}: {str(e)}")
                    if isinstance(method, list):
                        for m in method:
                            scores[m] = 0.0
                    else:
                        scores[method] = 0.0
            
            # Compute BERTScore
            try:
                # Use first reference for each hypothesis
                ref_texts = [refs[0] if refs else "" for refs in references]
                P, R, F1 = bert_score(hypotheses, ref_texts, lang='vi', device=self.device)
                scores['BERTScore_F1'] = F1.mean().item()
                scores['BERTScore_P'] = P.mean().item()
                scores['BERTScore_R'] = R.mean().item()
            except Exception as e:
                print(f"Warning: Failed to compute BERTScore: {str(e)}")
                scores['BERTScore_F1'] = 0.0
                scores['BERTScore_P'] = 0.0
                scores['BERTScore_R'] = 0.0
                
        except Exception as e:
            print(f"Error computing NLG scores: {str(e)}")
            return self._basic_scores(references, hypotheses)
        
        return scores
    
    def _basic_scores(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """Basic scores when advanced metrics are not available"""
        # Simple exact match
        exact_matches = 0
        for refs, hyp in zip(references, hypotheses):
            if any(ref.strip().lower() == hyp.strip().lower() for ref in refs):
                exact_matches += 1
        
        exact_match_score = exact_matches / len(hypotheses) if hypotheses else 0.0
        
        return {
            'BLEU-1': 0.0,
            'BLEU-2': 0.0,
            'BLEU-3': 0.0,
            'BLEU-4': 0.0,
            'METEOR': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0,
            'SPICE': 0.0,
            'BERTScore_F1': 0.0,
            'ExactMatch': exact_match_score
        }
    
    def evaluate_answers(self, predicted_answers: List[str], target_answers: List[str]) -> Dict[str, float]:
        """Evaluate answer accuracy"""
        if len(predicted_answers) != len(target_answers):
            raise ValueError("Number of predictions and targets must match")
        
        correct = 0
        total = len(predicted_answers)
        
        for pred, target in zip(predicted_answers, target_answers):
            if self._is_answer_correct(pred, target):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_full_results(self, results_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Evaluate full results from JSON file
        
        Args:
            results_file: Path to results JSON file
            output_file: Path to save evaluation scores
            
        Returns:
            Dictionary with all evaluation scores
        """
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract data
        predicted_answers = []
        target_answers = []
        predicted_explanations = []
        target_explanations = []
        
        for item in data:
            predicted_answers.append(item['pred_ans'])
            target_answers.append(item['gt_ans'])
            predicted_explanations.append(item['pred_explain'])
            
            # Handle both single string and list format for ground truth
            if isinstance(item['gt_explain'], list):
                target_explanations.append(item['gt_explain'])
            else:
                target_explanations.append([item['gt_explain']])
        
        # Evaluate answers
        answer_scores = self.evaluate_answers(predicted_answers, target_answers)
        
        # Get correct predictions for filtered evaluation
        correct_indices = []
        for i, (pred, target) in enumerate(zip(predicted_answers, target_answers)):
            if self._is_answer_correct(pred, target):
                correct_indices.append(i)
        
        # Evaluate explanations (unfiltered)
        print("Computing NLG scores for all explanations...")
        unfiltered_scores = self.get_nlg_scores(target_explanations, predicted_explanations)
        
        # Evaluate explanations (filtered - only correct answers)
        if correct_indices:
            filtered_target = [target_explanations[i] for i in correct_indices]
            filtered_predicted = [predicted_explanations[i] for i in correct_indices]
            print(f"Computing NLG scores for {len(correct_indices)} correct answers...")
            filtered_scores = self.get_nlg_scores(filtered_target, filtered_predicted)
        else:
            print("No correct answers found for filtered evaluation")
            filtered_scores = {key: 0.0 for key in unfiltered_scores.keys()}
        
        # Compute scaled scores (unfiltered * accuracy)
        scaled_scores = {key: value * answer_scores['accuracy'] 
                        for key, value in unfiltered_scores.items()}
        
        # Compile results
        results = {
            'accuracy': answer_scores['accuracy'],
            'task_score': answer_scores['accuracy'],  # Same as accuracy for VQA-X
            'correct_answers': answer_scores['correct'],
            'total_samples': answer_scores['total'],
            'unfiltered_scores': unfiltered_scores,
            'filtered_scores': filtered_scores,
            'scaled_scores': scaled_scores
        }
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Evaluation scores saved to {output_file}")
        
        return results
    
    def evaluate_results_from_data(self, results_data: List[Dict], output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate results directly from data (similar to evaluate_full_results but without file I/O)
        
        Args:
            results_data: List of result dictionaries
            output_dir: Directory to save evaluation scores
            
        Returns:
            Dictionary with all evaluation scores
        """
        
        # Extract data
        predicted_answers = [item['pred_ans'] for item in results_data]
        target_answers = [item['gt_ans'] for item in results_data]
        predicted_explanations = [item['pred_explain'] for item in results_data]
        target_explanations = []

        for item in results_data:
            # Handle both single string and list format for ground truth
            if isinstance(item['gt_explain'], list):
                target_explanations.append(item['gt_explain'])
            else:
                target_explanations.append([item['gt_explain']])

        # Evaluate answers to get accuracy
        answer_scores = self.evaluate_answers(predicted_answers, target_answers)
        accuracy = answer_scores['accuracy']
        correct_count = answer_scores['correct']
        total_examples = answer_scores['total']
        task_score = accuracy  # Same as accuracy for VQA-X
        
        # Get indices of correct predictions for filtered evaluation
        correct_indices = []
        for i, (pred, target) in enumerate(zip(predicted_answers, target_answers)):
            if self._is_answer_correct(pred, target):
                correct_indices.append(i)
        
        # Evaluate explanations (unfiltered - all samples)
        print("Computing NLG scores for all explanations...")
        unfiltered_scores = self.get_nlg_scores(target_explanations, predicted_explanations)
        
        # Evaluate explanations (filtered - only correct answers)
        if correct_indices:
            filtered_target = [target_explanations[i] for i in correct_indices]
            filtered_predicted = [predicted_explanations[i] for i in correct_indices]
            print(f"Computing NLG scores for {len(correct_indices)} correct answers...")
            filtered_scores = self.get_nlg_scores(filtered_target, filtered_predicted)
        else:
            print("No correct answers found for filtered evaluation")
            filtered_scores = {key: 0.0 for key in unfiltered_scores.keys()}
        
        # Compute scaled scores (unfiltered * accuracy)
        scaled_scores = {key: value * accuracy for key, value in unfiltered_scores.items()}
        
        # Compile results (following reference code format)
        results = {
            'accuracy': accuracy,
            'task_score': task_score,
            'correct_answers': correct_count,
            'total_samples': total_examples,
            'unfiltered_scores': unfiltered_scores,
            'filtered_scores': filtered_scores,
            'scaled_scores': scaled_scores
        }
        
        # Save results if output directory specified
        if output_dir:
            output_file = os.path.join(output_dir, "vivqa_x_evaluation_scores.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Evaluation scores saved to {output_file}")
        
        # Print evaluation summary
        self.print_evaluation_summary(results)
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct answers: {results['correct_answers']}")
        print(f"Answer accuracy: {results['accuracy']:.4f}")
        print(f"Task score: {results['task_score']:.4f}")
        
        print("\nNLG Scores (All samples - Unfiltered):")
        for metric, score in results['unfiltered_scores'].items():
            print(f"  {metric:15s}: {score:.4f}")
        
        print("\nNLG Scores (Correct answers only - Filtered):")
        for metric, score in results['filtered_scores'].items():
            print(f"  {metric:15s}: {score:.4f}")
        
        print("\nScaled NLG Scores (Unfiltered × Accuracy):")
        for metric, score in results['scaled_scores'].items():
            print(f"  {metric:15s}: {score:.4f}")
        
        print("="*60)

def evaluate_vivqa_x_results(results_file: str, output_file: str = None):
    """
    Convenience function to evaluate ViVQA-X results
    
    Args:
        results_file: Path to results JSON file
        output_file: Optional path to save evaluation scores
    """
    
    evaluator = ViVQAXEvaluator()
    results = evaluator.evaluate_full_results(results_file, output_file)
    evaluator.print_evaluation_summary(results)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ViVQA-X results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--output", help="Path to save evaluation scores")
    
    args = parser.parse_args()
    
    evaluate_vivqa_x_results(args.results, args.output)
