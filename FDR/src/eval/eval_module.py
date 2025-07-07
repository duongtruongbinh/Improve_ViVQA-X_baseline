"""
EvalModule for FDR Pipeline
Comprehensive evaluation metrics for VQA accuracy and explanation quality.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import torch
import sys
import os

# Optional imports for explanation evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. BLEU and METEOR scores will not be computed.")

try:
    from bert_score import score as bert_scorer
    BERTSCORE_AVAILABLE = True
    logging.info("✅ bert-score library available.")
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score not installed. BERTScore will not be computed. Run `pip install bert-score`")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
    logging.info("✅ rouge-score library available.")
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not installed. ROUGE scores will not be computed. Run `pip install rouge-score`")

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    COCO_METRICS_AVAILABLE = True
    logging.info("✅ pycocoevalcap library available.")
except ImportError:
    COCO_METRICS_AVAILABLE = False
    logging.warning("pycocoevalcap not installed. CIDEr and SPICE scores will not be computed. Run `pip install pycocoevalcap`")

# Define TRANSFORMERS_AVAILABLE at module level
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers not available. Semantic similarity scores will not be computed.")


class EvalModule:
    """
    Comprehensive evaluation module for FDR pipeline.
    Evaluates both VQA accuracy and explanation quality using multiple metrics.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
                # Don't modify global TRANSFORMERS_AVAILABLE here, just note the failure
        
        logging.info(f"🔬 EvalModule initialized on {device}")

    def evaluate_results(self, results: List[Dict[str, Any]], 
                        ground_truth_explanations: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of FDR pipeline results.
        
        Args:
            results: List of pipeline results with final_answer and generated_explanation
            ground_truth_explanations: Optional dict mapping question_id to list of reference explanations
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"🔬 Evaluating {len(results)} results...")
        
        # VQA Accuracy Metrics
        vqa_metrics = self._evaluate_vqa_accuracy(results)
        
        # Explanation Quality Metrics (if ground truth explanations provided)
        explanation_metrics = {}
        if ground_truth_explanations:
            explanation_metrics = self._evaluate_explanation_quality(results, ground_truth_explanations)
        
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
        bert_scores_answers = []
        
        for result in results:
            ground_truth = result.get("ground_truth_answer") or result.get("ground_truth")
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
                
                # BERTScore for answer comparison
                if BERTSCORE_AVAILABLE and final_answer and ground_truth:
                    p, r, f1 = self._compute_bert_score_single(str(final_answer), str(ground_truth))
                    bert_scores_answers.append({"p": p, "r": r, "f1": f1})
                
                answer_distribution[final_answer] += 1
        
        accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
        
        vqa_metrics = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "evaluated_count": evaluated_count,
            "answer_distribution": dict(answer_distribution.most_common(10)),
            "error_analysis": dict(error_analysis)
        }
        
        # Add BERTScore metrics for answers
        if bert_scores_answers:
            vqa_metrics.update({
                "answer_bert_precision": np.mean([score["p"] for score in bert_scores_answers]),
                "answer_bert_recall": np.mean([score["r"] for score in bert_scores_answers]),
                "answer_bert_f1": np.mean([score["f1"] for score in bert_scores_answers])
            })
        
        return vqa_metrics

    def _compute_bert_score_single(self, generated: str, reference: str) -> tuple[float, float, float]:
        """Compute BERTScore for single reference"""
        if not BERTSCORE_AVAILABLE:
            return 0.0, 0.0, 0.0
        
        try:
            P, R, F1 = bert_scorer([generated], [reference], lang="vi", 
                                 model_type="bert-base-multilingual-cased", device=self.device)
            return P.item(), R.item(), F1.item()
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return 0.0, 0.0, 0.0

    def _evaluate_explanation_quality(self, results: List[Dict[str, Any]], 
                                    ground_truth_explanations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate explanation quality using multiple metrics"""
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        cider_scores = []
        spice_scores = []
        semantic_scores = []
        bert_scores = []
        length_stats = []
        
        evaluated_explanations = 0
        
        for result in results:
            question_id = result.get("question_id")
            generated_explanation = result.get("generated_explanation", "") or result.get("explanation", "")
            
            if question_id in ground_truth_explanations and generated_explanation:
                reference_explanations = ground_truth_explanations[question_id]
                evaluated_explanations += 1
                
                # BLEU Score (1-4)
                if NLTK_AVAILABLE:
                    bleu_score = self._compute_bleu_score(generated_explanation, reference_explanations)
                    bleu_scores.append(bleu_score)
                
                # ROUGE-L Score
                if ROUGE_AVAILABLE:
                    rouge_score = self._compute_rouge_score(generated_explanation, reference_explanations)
                    rouge_scores.append(rouge_score)
                
                # METEOR Score
                if NLTK_AVAILABLE:
                    meteor_score = self._compute_meteor_score(generated_explanation, reference_explanations)
                    meteor_scores.append(meteor_score)
                
                # CIDEr Score
                if COCO_METRICS_AVAILABLE:
                    cider_score = self._compute_cider_score(generated_explanation, reference_explanations)
                    cider_scores.append(cider_score)
                
                # SPICE Score
                if COCO_METRICS_AVAILABLE:
                    spice_score = self._compute_spice_score(generated_explanation, reference_explanations)
                    spice_scores.append(spice_score)
                
                # BERTScore
                if BERTSCORE_AVAILABLE:
                    p, r, f1 = self._compute_bert_score(generated_explanation, reference_explanations)
                    bert_scores.append({"p": p, "r": r, "f1": f1})
                
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
        
        # BLEU metrics
        if bleu_scores:
            explanation_metrics.update({
                "bleu_1": np.mean([score["bleu_1"] for score in bleu_scores]),
                "bleu_2": np.mean([score["bleu_2"] for score in bleu_scores]),
                "bleu_3": np.mean([score["bleu_3"] for score in bleu_scores]),
                "bleu_4": np.mean([score["bleu_4"] for score in bleu_scores])
            })
        
        # ROUGE metrics
        if rouge_scores:
            explanation_metrics.update({
                "rouge_l_precision": np.mean([score["rouge_l_precision"] for score in rouge_scores]),
                "rouge_l_recall": np.mean([score["rouge_l_recall"] for score in rouge_scores]),
                "rouge_l_f1": np.mean([score["rouge_l_f1"] for score in rouge_scores])
            })
        
        # METEOR metrics
        if meteor_scores:
            explanation_metrics.update({
                "meteor": np.mean(meteor_scores),
                "meteor_std": np.std(meteor_scores)
            })
        
        # CIDEr metrics
        if cider_scores:
            explanation_metrics.update({
                "cider": np.mean(cider_scores),
                "cider_std": np.std(cider_scores)
            })
        
        # SPICE metrics
        if spice_scores:
            explanation_metrics.update({
                "spice": np.mean(spice_scores),
                "spice_std": np.std(spice_scores)
            })
        
        # BERTScore metrics
        if bert_scores:
            explanation_metrics.update({
                "bert_precision": np.mean([score["p"] for score in bert_scores]),
                "bert_recall": np.mean([score["r"] for score in bert_scores]),
                "bert_f1": np.mean([score["f1"] for score in bert_scores])
            })
        
        # Semantic similarity metrics
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
            explanation = result.get("generated_explanation", "") or result.get("explanation", "")
            
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

    def _compute_rouge_score(self, generated: str, references: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L scores"""
        if not ROUGE_AVAILABLE:
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}
        
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            # Compute ROUGE-L against all references and take the maximum
            max_precision = 0.0
            max_recall = 0.0
            max_f1 = 0.0
            
            for reference in references:
                scores = scorer.score(reference, generated)
                rouge_l = scores['rougeL']
                
                max_precision = max(max_precision, rouge_l.precision)
                max_recall = max(max_recall, rouge_l.recall)
                max_f1 = max(max_f1, rouge_l.fmeasure)
            
            return {
                "rouge_l_precision": max_precision,
                "rouge_l_recall": max_recall,
                "rouge_l_f1": max_f1
            }
        except Exception as e:
            logging.warning(f"ROUGE computation failed: {e}")
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    def _compute_meteor_score(self, generated: str, references: List[str]) -> float:
        """Compute METEOR score"""
        if not NLTK_AVAILABLE:
            return 0.0
        
        try:
            generated_tokens = word_tokenize(generated.lower())
            reference_tokens_list = [word_tokenize(ref.lower()) for ref in references]
            
            # METEOR computes score against multiple references automatically
            meteor = meteor_score(reference_tokens_list, generated_tokens)
            return meteor
        except Exception as e:
            logging.warning(f"METEOR computation failed: {e}")
            return 0.0

    def _compute_cider_score(self, generated: str, references: List[str]) -> float:
        """Compute CIDEr score"""
        if not COCO_METRICS_AVAILABLE:
            return 0.0
        
        try:
            # CIDEr expects specific format: dict with image_id -> [generated] and [references]
            cider_scorer = Cider()
            
            # Format data for CIDEr
            gts = {"0": references}  # ground truth
            res = {"0": [generated]}  # results
            
            score, _ = cider_scorer.compute_score(gts, res)
            return float(score)
        except Exception as e:
            logging.warning(f"CIDEr computation failed: {e}")
            return 0.0

    def _compute_spice_score(self, generated: str, references: List[str]) -> float:
        """
        Compute SPICE score, redirecting stdout/stderr to suppress noisy Java logs.
        This is a more robust method to silence the underlying Java processes.
        """
        if not COCO_METRICS_AVAILABLE:
            return 0.0

        # Backup original stdout and stderr file descriptors
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        
        # Duplicate original file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # Open a file to /dev/null
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            # Redirect stdout and stderr to /dev/null
            os.dup2(devnull_fd, original_stdout_fd)
            os.dup2(devnull_fd, original_stderr_fd)

            spice_scorer = Spice()
            gts = {"0": references}
            res = {"0": [generated]}
            score, _ = spice_scorer.compute_score(gts, res)
            return float(score)
        except Exception as e:
            # Restore stdout/stderr before logging the warning
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)
            logging.warning(f"SPICE computation failed: {e}")
            return 0.0
        finally:
            # Always restore stdout and stderr
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)
            
            # Close the duplicated file descriptors and /dev/null
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)

    def _compute_semantic_similarity(self, generated: str, references: List[str]) -> float:
        """Compute semantic similarity using sentence embeddings"""
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
        logging.info("📊 FDR COMPREHENSIVE EVALUATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Total Questions: {summary['total_questions']}")
        logging.info(f"Evaluated Questions: {summary['evaluated_questions']}")
        logging.info(f"VQA Accuracy: {summary['vqa_accuracy']:.3f}")
        
        # VQA Answer BERTScore
        if "answer_bert_f1" in vqa_metrics:
            logging.info("🎯 Answer BERTScore:")
            logging.info(f"  Precision: {vqa_metrics['answer_bert_precision']:.3f}")
            logging.info(f"  Recall: {vqa_metrics['answer_bert_recall']:.3f}")
            logging.info(f"  F1: {vqa_metrics['answer_bert_f1']:.3f}")
        
        if evaluation_results["explanation_metrics"]:
            exp_metrics = evaluation_results["explanation_metrics"]
            logging.info(f"Explanation Evaluations: {exp_metrics['evaluated_explanations']}")
            
            # BLEU metrics
            if "bleu_4" in exp_metrics:
                logging.info("📝 BLEU Scores:")
                logging.info(f"  BLEU-1: {exp_metrics['bleu_1']:.3f}")
                logging.info(f"  BLEU-2: {exp_metrics['bleu_2']:.3f}")
                logging.info(f"  BLEU-3: {exp_metrics['bleu_3']:.3f}")
                logging.info(f"  BLEU-4: {exp_metrics['bleu_4']:.3f}")
            
            # ROUGE metrics
            if "rouge_l_f1" in exp_metrics:
                logging.info("📄 ROUGE-L Scores:")
                logging.info(f"  Precision: {exp_metrics['rouge_l_precision']:.3f}")
                logging.info(f"  Recall: {exp_metrics['rouge_l_recall']:.3f}")
                logging.info(f"  F1: {exp_metrics['rouge_l_f1']:.3f}")
            
            # METEOR score
            if "meteor" in exp_metrics:
                logging.info(f"☄️ METEOR Score: {exp_metrics['meteor']:.3f}")
            
            # CIDEr score
            if "cider" in exp_metrics:
                logging.info(f"🎯 CIDEr Score: {exp_metrics['cider']:.3f}")
            
            # SPICE score
            if "spice" in exp_metrics:
                logging.info(f"🌶️ SPICE Score: {exp_metrics['spice']:.3f}")
            
            # BERTScore for explanations
            if "bert_f1" in exp_metrics:
                logging.info("🤖 Explanation BERTScore:")
                logging.info(f"  Precision: {exp_metrics['bert_precision']:.3f}")
                logging.info(f"  Recall: {exp_metrics['bert_recall']:.3f}")
                logging.info(f"  F1: {exp_metrics['bert_f1']:.3f}")
            
            # Semantic similarity
            if "semantic_similarity" in exp_metrics:
                logging.info(f"🧠 Semantic Similarity: {exp_metrics['semantic_similarity']:.3f}")
        
        consistency_metrics = evaluation_results["consistency_metrics"]
        logging.info(f"🔗 Answer-Explanation Consistency: {consistency_metrics.get('avg_consistency', 0.0):.3f}")
        
        confidence_metrics = evaluation_results["confidence_metrics"]
        logging.info(f"📊 Average Confidence: {confidence_metrics.get('avg_confidence', 0.0):.3f}")
        logging.info("="*60)

    def save_evaluation_report(self, evaluation_results: Dict[str, Any], output_path: str):
        """Save detailed evaluation report to JSON file"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"📄 Evaluation report saved to: {output_path}")

    def _compute_bert_score(self, generated: str, references: List[str]) -> tuple[float, float, float]:
        """Compute BERTScore"""
        if not BERTSCORE_AVAILABLE:
            return 0.0, 0.0, 0.0
        
        try:
            # lang="vi" for Vietnamese, but multi-lingual models work well
            P, R, F1 = bert_scorer([generated], [references], lang="vi", model_type="bert-base-multilingual-cased", device=self.device)
            # The result is a tensor, we take the first (and only) item
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return 0.0, 0.0, 0.0

    def evaluate_comprehensive(self, results: List[Dict[str, Any]], 
                             ground_truth_explanations: Optional[Dict[str, List[str]]] = None,
                             include_g_eval: bool = False,
                             g_evaluator = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation including all metrics:
        VQA Accuracy, BLEU (1-4), ROUGE-L, METEOR, CIDEr, BERTScore, SPICE, G-Eval
        
        Args:
            results: List of pipeline results with final_answer and generated_explanation
            ground_truth_explanations: Optional dict mapping question_id to list of reference explanations
            include_g_eval: Whether to include G-Eval metrics
            g_evaluator: GEvaluator instance for G-Eval
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"🔬 Running comprehensive evaluation on {len(results)} results...")
        
        # Run standard evaluation
        evaluation_results = self.evaluate_results(results, ground_truth_explanations)
        
        # Add G-Eval if requested
        if include_g_eval and g_evaluator and ground_truth_explanations:
            logging.info("🎭 Running G-Eval assessment...")
            g_eval_metrics = g_evaluator.evaluate_explanation_batch(results, ground_truth_explanations)
            evaluation_results["g_eval_metrics"] = g_eval_metrics
        
        # Update summary with all available metrics
        summary = evaluation_results["summary"]
        summary["metrics_included"] = {
            "vqa_accuracy": True,
            "bleu_scores": NLTK_AVAILABLE,
            "rouge_scores": ROUGE_AVAILABLE, 
            "meteor_score": NLTK_AVAILABLE,
            "cider_score": COCO_METRICS_AVAILABLE,
            "spice_score": COCO_METRICS_AVAILABLE,
            "bert_score": BERTSCORE_AVAILABLE,
            "semantic_similarity": TRANSFORMERS_AVAILABLE,
            "g_eval": include_g_eval and g_evaluator is not None
        }
        
        return evaluation_results 