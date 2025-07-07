"""
G-Evaluator for VQA Response Quality Assessment
Uses GPT-4o-mini to evaluate VQA answers based on multiple criteria
"""

import logging
import json
import numpy as np
from openai import OpenAI
from typing import Dict, Any, Optional, List
from collections import defaultdict


class GEvaluator:
    """
    G-Eval style evaluator for VQA responses and explanations using GPT-4o-mini.
    """
    
    def __init__(self, client: OpenAI, model_name: str = "gpt-4o-mini"):
        """
        Initialize the G-Evaluator.
        
        Args:
            client: An initialized OpenAI client.
            model_name: The model to use for evaluation.
        """
        self.client = client
        self.model_name = model_name
        if self.client:
            logging.info(f"🔬 G-Evaluator initialized with {model_name}")
        else:
            logging.warning("G-Evaluator initialized without an OpenAI client. Explanation evaluation will be skipped.")
    
    def evaluate_explanation_batch(self, results: List[Dict[str, Any]], ground_truth_explanations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Run G-Eval on a sample of results for deeper, qualitative metrics on explanations.
        """
        if not self.client:
            return {}

        logging.info("Running G-Eval for qualitative assessment of explanations...")
        all_scores = defaultdict(list)
        
        sample_size = min(len(results), 10)
        # Using list directly since np.random.choice requires a 1-D array-like
        indices = np.random.choice(len(results), sample_size, replace=False)
        sampled_results = [results[i] for i in indices]

        for result in sampled_results:
            question_id = result.get("question_id")
            if not question_id or question_id not in ground_truth_explanations:
                continue

            generated_explanation = result.get("explanation", "") # Changed from generated_explanation to explanation
            if not generated_explanation:
                continue
            
            eval_pack = {
                "question": result.get("question", ""),
                "generated_explanation": generated_explanation,
                "reference_explanations": ground_truth_explanations[question_id]
            }

            scores = self._compute_explanation_metrics(eval_pack)
            if scores:
                for key, value in scores.items():
                    all_scores[key].append(value)

        if not all_scores:
            return {"evaluated_samples": 0}

        avg_scores = {f"avg_{key}": np.mean(values) for key, values in all_scores.items()}
        avg_scores["evaluated_samples"] = len(all_scores["relevance"])
        
        logging.info(f"G-Eval for explanations completed on {avg_scores['evaluated_samples']} samples.")
        return avg_scores

    def _compute_explanation_metrics(self, eval_pack: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Calls GPT-4o-mini to get scores for relevance, coherence, and faithfulness for an explanation.
        """
        ref_explanations_formatted = "\n".join([f"- {ref}" for ref in eval_pack['reference_explanations']])
        
        prompt = f"""
You are an expert evaluator for a Visual Question Answering (VQA) system. Your task is to evaluate the generated explanation for a given question based on a set of reference explanations. Provide scores on a scale of 1 to 5 for the following criteria:

1.  **Relevance**: Does the explanation directly address the question and the visual content (even though you can't see the image)? Is it on-topic? (1=irrelevant, 5=highly relevant)
2.  **Coherence**: Is the explanation easy to understand, logical, and well-structured? (1=incoherent, 5=highly coherent)
3.  **Faithfulness**: How well does the generated explanation align with the information provided in the reference explanations? Does it contradict the references? (1=contradictory/unfaithful, 5=highly faithful)

**Input Data:**
- **Question**: "{eval_pack['question']}"
- **Generated Explanation**: "{eval_pack['generated_explanation']}"
- **Reference Explanations**: 
{ref_explanations_formatted}

**Instructions**:
Return your evaluation as a JSON object with the keys "relevance", "coherence", and "faithfulness". Do not include any other text or markdown.

**JSON Output Format:**
{{
  "relevance": <score_1_to_5>,
  "coherence": <score_1_to_5>,
  "faithfulness": <score_1_to_5>
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI evaluator. Your response must be a single, clean JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            scores = json.loads(content)

            for key in ["relevance", "coherence", "faithfulness"]:
                if not (isinstance(scores.get(key), (int, float)) and 1 <= scores.get(key, 0) <= 5):
                    logging.warning(f"G-Eval returned an invalid or out-of-range score for {key}: {scores.get(key)}. Skipping.")
                    return None
            
            return scores

        except Exception as e:
            logging.error(f"An unexpected error occurred during G-Eval for explanation: {e}", exc_info=True)
            return None
    
    def evaluate_single(self, 
                       question: str, 
                       answer: str, 
                       image_description: Optional[str] = None,
                       criteria: str = "overall") -> Dict[str, Any]:
        """
        Evaluate a single VQA response
        
        Args:
            question: The VQA question
            answer: The model's answer
            image_description: Optional description of the image
            criteria: Evaluation criteria type
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Construct evaluation prompt
            eval_prompt = self._construct_eval_prompt(
                question, answer, image_description, criteria
            )
            
            # Get evaluation from GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert VQA evaluator. Provide objective, detailed assessments."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            eval_text = response.choices[0].message.content
            return self._parse_evaluation(eval_text)
            
        except Exception as e:
            logging.error(f"❌ G-Eval error: {e}")
            return {
                "error": str(e),
                "overall_score": "N/A",
                "analysis": "Evaluation failed"
            }
    
    def _construct_eval_prompt(self, 
                              question: str, 
                              answer: str, 
                              image_description: Optional[str],
                              criteria: str) -> str:
        """Construct the evaluation prompt"""
        
        base_prompt = f"""
Please evaluate the following VQA (Visual Question Answering) response:

**Question:** {question}
**Answer:** {answer}
"""
        
        if image_description:
            base_prompt += f"**Image Context:** {image_description}\n"
        
        base_prompt += f"""
**Evaluation Task:** Rate this answer on a scale of 1-5 based on the following criteria:

1. **Accuracy** (1-5): How factually correct is the answer?
2. **Relevance** (1-5): How well does the answer address the question?
3. **Completeness** (1-5): Does the answer provide sufficient detail?
4. **Clarity** (1-5): How clear and understandable is the answer?

**Instructions:**
- Provide scores for each criterion (1-5 scale)
- Calculate an overall score (average of all criteria)
- Provide brief analysis explaining the scores
- Be objective and consistent in your evaluation

**Format your response as:**
Accuracy: [score]/5
Relevance: [score]/5  
Completeness: [score]/5
Clarity: [score]/5
Overall: [score]/5

Analysis: [Your detailed explanation]
"""
        
        return base_prompt
    
    def _parse_evaluation(self, eval_text: str) -> Dict[str, Any]:
        """Parse the evaluation response from GPT-4o-mini"""
        
        result = {
            "accuracy_score": "N/A",
            "relevance_score": "N/A", 
            "completeness_score": "N/A",
            "clarity_score": "N/A",
            "overall_score": "N/A",
            "analysis": "",
            "raw_evaluation": eval_text
        }
        
        try:
            lines = eval_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('Accuracy:'):
                    result["accuracy_score"] = self._extract_score(line)
                elif line.startswith('Relevance:'):
                    result["relevance_score"] = self._extract_score(line)
                elif line.startswith('Completeness:'):
                    result["completeness_score"] = self._extract_score(line)
                elif line.startswith('Clarity:'):
                    result["clarity_score"] = self._extract_score(line)
                elif line.startswith('Overall:'):
                    result["overall_score"] = self._extract_score(line)
                elif line.startswith('Analysis:'):
                    # Get everything after "Analysis:"
                    analysis_start = eval_text.find('Analysis:')
                    if analysis_start != -1:
                        result["analysis"] = eval_text[analysis_start + 9:].strip()
                    break
            
        except Exception as e:
            logging.warning(f"⚠️ Error parsing evaluation: {e}")
            result["analysis"] = f"Parsing error: {e}"
        
        return result
    
    def _extract_score(self, line: str) -> str:
        """Extract numerical score from evaluation line"""
        try:
            # Look for pattern like "4/5" or just "4"
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', line)
            if score_match:
                return score_match.group(1)
        except:
            pass
        return "N/A"
    
    def evaluate_batch(self, 
                      evaluation_data: list,
                      criteria: str = "overall") -> Dict[str, Any]:
        """
        Evaluate multiple VQA responses in batch
        
        Args:
            evaluation_data: List of dicts with 'question', 'answer', 'image_description'
            criteria: Evaluation criteria type
            
        Returns:
            Dictionary with batch evaluation results
        """
        results = []
        total_scores = []
        
        for i, data in enumerate(evaluation_data):
            logging.info(f"🔬 Evaluating response {i+1}/{len(evaluation_data)}")
            
            eval_result = self.evaluate_single(
                question=data['question'],
                answer=data['answer'], 
                image_description=data.get('image_description'),
                criteria=criteria
            )
            
            results.append(eval_result)
            
            # Track overall scores for statistics
            overall_score = eval_result.get('overall_score')
            if overall_score != "N/A":
                try:
                    total_scores.append(float(overall_score))
                except:
                    pass
        
        # Calculate batch statistics
        stats = {}
        if total_scores:
            stats = {
                "mean_score": sum(total_scores) / len(total_scores),
                "max_score": max(total_scores),
                "min_score": min(total_scores),
                "total_evaluated": len(total_scores)
            }
        
        return {
            "individual_results": results,
            "batch_statistics": stats,
            "total_items": len(evaluation_data)
        } 