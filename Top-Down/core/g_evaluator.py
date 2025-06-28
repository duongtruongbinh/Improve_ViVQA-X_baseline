"""
G-Evaluator for VQA Response Quality Assessment
Uses GPT-4o-mini to evaluate VQA answers based on multiple criteria
"""

import logging
from openai import OpenAI
from typing import Dict, Any, Optional


class GEvaluator:
    """
    G-Eval style evaluator for VQA responses using GPT-4o-mini
    Evaluates answers based on accuracy, relevance, and completeness
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key_file: str = "openai_key.txt"):
        """
        Initialize the G-Evaluator
        
        Args:
            model_name: The model to use for evaluation (default: gpt-4o-mini)
            api_key_file: Path to the OpenAI API key file
        """
        self.model_name = model_name
        
        # Setup OpenAI client
        try:
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
            logging.info(f"🔬 G-Evaluator initialized with {model_name}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize G-Evaluator: {e}")
            raise
    
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