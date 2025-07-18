"""
EvalModule for MVKB-X Pipeline
Comprehensive evaluation metrics for VQA accuracy and explanation quality.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import torch

# Optional imports for explanation evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. BLEU scores will not be computed.")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Semantic similarity scores will not be computed.")


class EvalModule:
    """
    Comprehensive evaluation module for MVKB-X pipeline.
    Evaluates both VQA accuracy and explanation quality using multiple metrics.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        global TRANSFORMERS_AVAILABLE
        self.device = device
        self.semantic_model = None
        self.semantic_tokenizer = None
        
        # Initialize semantic similarity model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.semantic_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.semantic_model = AutoModel.from_pretrained(model_name).to(device)
                logging.info(f"✅ Semantic similarity model loaded: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load semantic model: {e}")
                TRANSFORMERS_AVAILABLE = False
        
        logging.info(f"🔬 EvalModule initialized on {device}")

    def evaluate_results(self, results: List[Dict[str, Any]], 
                        ground_truth_explanations: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of MVKB-X pipeline results.
        
        Args:
            results: List of pipeline results with final_answer and generated_explanation
            ground_truth_explanations: Optional dict mapping question_id to list of reference explanations
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"🔬 Evaluating {len(results)} results...")
        
        # 🔧 Auto-extract ground_truth_explanations if not provided
        if ground_truth_explanations is None:
            ground_truth_explanations = {}
            for result in results:
                question_id = result.get('question_id')
                gt_explanations = result.get('ground_truth_explanations', [])
                if question_id and gt_explanations:
                    ground_truth_explanations[question_id] = gt_explanations
            
            if ground_truth_explanations:
                logging.info(f"📚 Auto-extracted ground truth explanations for {len(ground_truth_explanations)} questions")
            else:
                logging.warning("⚠️ No ground truth explanations found in results")
        
        # VQA Accuracy Metrics
        vqa_metrics = self._evaluate_vqa_accuracy(results)
        
        # Explanation Quality Metrics (if ground truth explanations provided)
        explanation_metrics = {}
        if ground_truth_explanations:
            explanation_metrics = self._evaluate_explanation_quality(results, ground_truth_explanations)
            logging.info(f"📝 Evaluated explanations for {explanation_metrics.get('evaluated_explanations', 0)} questions")
        else:
            logging.info("📝 Skipping explanation evaluation (no ground truth explanations)")
        
        # Consistency Metrics (answer-explanation alignment)
        consistency_metrics = self._evaluate_consistency(results)
        
        # Confidence Metrics
        confidence_metrics = self._evaluate_confidence(results)
        
        # Combine all metrics
        evaluation_results = {
            "summary": {
                "total_questions": len(results),
                "evaluated_questions": vqa_metrics.get("evaluated_count", 0),
                "vqa_accuracy": vqa_metrics.get("accuracy", 0.0),
                "explanation_available": len(explanation_metrics) > 0
            },
            "vqa_metrics": vqa_metrics,
            "explanation_metrics": explanation_metrics,
            "consistency_metrics": consistency_metrics,
            "confidence_metrics": confidence_metrics
        }
        
        # Log summary
        self._log_evaluation_summary(evaluation_results)
        
        return evaluation_results

    def _evaluate_vqa_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate VQA answer accuracy"""
        correct_count = 0
        evaluated_count = 0
        
        answer_distribution = Counter()
        error_analysis = defaultdict(list)
        
        for result in results:
            ground_truth = result.get("ground_truth")
            final_answer = result.get("final_answer", "")
            
            if ground_truth is not None:
                evaluated_count += 1
                
                # Normalize answers for comparison
                gt_normalized = str(ground_truth).lower().strip()
                pred_normalized = str(final_answer).lower().strip()
                
                is_correct = gt_normalized == pred_normalized
                if is_correct:
                    correct_count += 1
                else:
                    # Error analysis
                    error_analysis["incorrect_predictions"].append({
                        "question_id": result.get("question_id"),
                        "question": result.get("question", ""),
                        "predicted": final_answer,
                        "ground_truth": ground_truth
                    })
                
                answer_distribution[final_answer] += 1
        
        accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "evaluated_count": evaluated_count,
            "answer_distribution": dict(answer_distribution.most_common(10)),
            "error_analysis": dict(error_analysis)
        }

    def _evaluate_explanation_quality(self, results: List[Dict[str, Any]], 
                                    ground_truth_explanations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate explanation quality using multiple metrics"""
        global TRANSFORMERS_AVAILABLE, NLTK_AVAILABLE
        
        bleu_scores = []
        semantic_scores = []
        length_stats = []
        
        evaluated_explanations = 0
        
        for result in results:
            question_id = result.get("question_id")
            generated_explanation = result.get("explanation", "")
            
            if question_id in ground_truth_explanations and generated_explanation:
                reference_explanations = ground_truth_explanations[question_id]
                evaluated_explanations += 1
                
                # BLEU Score
                if NLTK_AVAILABLE:
                    bleu_score = self._compute_bleu_score(generated_explanation, reference_explanations)
                    bleu_scores.append(bleu_score)
                
                # Semantic Similarity
                if TRANSFORMERS_AVAILABLE and self.semantic_model:
                    semantic_score = self._compute_semantic_similarity(generated_explanation, reference_explanations)
                    semantic_scores.append(semantic_score)
                
                # Length statistics
                length_stats.append(len(generated_explanation.split()))
        
        explanation_metrics = {
            "evaluated_explanations": evaluated_explanations,
            "avg_length": np.mean(length_stats) if length_stats else 0.0,
            "std_length": np.std(length_stats) if length_stats else 0.0
        }
        
        if bleu_scores:
            explanation_metrics.update({
                "bleu_1": np.mean([score["bleu_1"] for score in bleu_scores]),
                "bleu_2": np.mean([score["bleu_2"] for score in bleu_scores]),
                "bleu_3": np.mean([score["bleu_3"] for score in bleu_scores]),
                "bleu_4": np.mean([score["bleu_4"] for score in bleu_scores])
            })
        
        if semantic_scores:
            explanation_metrics.update({
                "semantic_similarity": np.mean(semantic_scores),
                "semantic_std": np.std(semantic_scores)
            })
        
        return explanation_metrics

    def _evaluate_consistency(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate consistency between answers and explanations"""
        consistency_scores = []
        
        for result in results:
            final_answer = result.get("final_answer", "")
            explanation = result.get("explanation", "")
            
            if final_answer and explanation:
                # Simple consistency check: does explanation mention the answer?
                consistency_score = self._compute_answer_explanation_consistency(final_answer, explanation)
                consistency_scores.append(consistency_score)
        
        return {
            "avg_consistency": np.mean(consistency_scores) if consistency_scores else 0.0,
            "consistency_distribution": self._compute_score_distribution(consistency_scores)
        }

    def _evaluate_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate confidence calibration"""
        confidence_scores = []
        accuracy_by_confidence = defaultdict(list)
        
        for result in results:
            mvkb_trace = result.get("mvkb_trace", {})
            synthesis_result = mvkb_trace.get("synthesis_result", {})
            confidence_breakdown = synthesis_result.get("confidence_breakdown", {})
            
            winning_score = confidence_breakdown.get("winning_score", 0.0)
            total_votes = confidence_breakdown.get("total_votes", 1.0)
            
            # Normalize confidence
            normalized_confidence = winning_score / total_votes if total_votes > 0 else 0.0
            confidence_scores.append(normalized_confidence)
            
            # Accuracy by confidence level
            is_correct = result.get("is_correct", False)
            confidence_bin = self._get_confidence_bin(normalized_confidence)
            accuracy_by_confidence[confidence_bin].append(is_correct)
        
        # Calculate calibration metrics
        calibration_metrics = {}
        for bin_name, accuracies in accuracy_by_confidence.items():
            if accuracies:
                calibration_metrics[f"accuracy_{bin_name}"] = np.mean(accuracies)
                calibration_metrics[f"count_{bin_name}"] = len(accuracies)
        
        return {
            "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0.0,
            "calibration": calibration_metrics
        }

    def _compute_bleu_score(self, generated: str, references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores"""
        global NLTK_AVAILABLE
        
        if not NLTK_AVAILABLE:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        try:
            generated_tokens = word_tokenize(generated.lower())
            reference_tokens_list = [word_tokenize(ref.lower()) for ref in references]
            
            smoothing = SmoothingFunction().method1
            
            bleu_1 = sentence_bleu(reference_tokens_list, generated_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(reference_tokens_list, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu(reference_tokens_list, generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(reference_tokens_list, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
        except Exception as e:
            logging.warning(f"BLEU computation failed: {e}")
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    def _compute_semantic_similarity(self, generated: str, references: List[str]) -> float:
        """Compute semantic similarity using sentence embeddings"""
        global TRANSFORMERS_AVAILABLE
        
        if not TRANSFORMERS_AVAILABLE or not self.semantic_model:
            return 0.0
        
        try:
            # Encode generated explanation
            generated_embedding = self._encode_text(generated)
            
            # Encode all references and take maximum similarity
            max_similarity = 0.0
            for reference in references:
                ref_embedding = self._encode_text(reference)
                similarity = F.cosine_similarity(generated_embedding, ref_embedding, dim=0).item()
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
        except Exception as e:
            logging.warning(f"Semantic similarity computation failed: {e}")
            return 0.0

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding using semantic model"""
        inputs = self.semantic_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.semantic_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.squeeze()

    def _compute_answer_explanation_consistency(self, answer: str, explanation: str) -> float:
        """Compute consistency between answer and explanation"""
        answer_lower = answer.lower().strip()
        explanation_lower = explanation.lower()
        
        # Simple heuristic: does explanation mention the answer?
        if answer_lower in explanation_lower:
            return 1.0
        
        # Check for semantic consistency (simplified)
        answer_words = set(answer_lower.split())
        explanation_words = set(explanation_lower.split())
        
        if answer_words.intersection(explanation_words):
            return 0.5
        
        return 0.0

    def _get_confidence_bin(self, confidence: float) -> str:
        """Get confidence bin label"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def _compute_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Compute distribution of scores"""
        if not scores:
            return {}
        
        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        
        for score in scores:
            if score <= 0.2:
                bins["0.0-0.2"] += 1
            elif score <= 0.4:
                bins["0.2-0.4"] += 1
            elif score <= 0.6:
                bins["0.4-0.6"] += 1
            elif score <= 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        
        return bins

    def _log_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Log evaluation summary"""
        summary = evaluation_results["summary"]
        vqa_metrics = evaluation_results["vqa_metrics"]
        
        logging.info("="*60)
        logging.info("📊 MVKB-X EVALUATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Total Questions: {summary['total_questions']}")
        logging.info(f"Evaluated Questions: {summary['evaluated_questions']}")
        logging.info(f"VQA Accuracy: {summary['vqa_accuracy']:.3f}")
        
        if evaluation_results["explanation_metrics"]:
            exp_metrics = evaluation_results["explanation_metrics"]
            logging.info(f"Explanation Evaluations: {exp_metrics['evaluated_explanations']}")
            if "bleu_4" in exp_metrics:
                logging.info(f"BLEU-4 Score: {exp_metrics['bleu_4']:.3f}")
            if "semantic_similarity" in exp_metrics:
                logging.info(f"Semantic Similarity: {exp_metrics['semantic_similarity']:.3f}")
        
        consistency_metrics = evaluation_results["consistency_metrics"]
        logging.info(f"Answer-Explanation Consistency: {consistency_metrics.get('avg_consistency', 0.0):.3f}")
        
        confidence_metrics = evaluation_results["confidence_metrics"]
        logging.info(f"Average Confidence: {confidence_metrics.get('avg_confidence', 0.0):.3f}")
        logging.info("="*60)

    def save_evaluation_report(self, evaluation_results: Dict[str, Any], output_path: str):
        """Save detailed evaluation report to JSON file"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"📄 Evaluation report saved to: {output_path}") 