"""
FDR Evaluation Module
Comprehensive evaluation metrics for VQA accuracy and explanation quality.

Supported Metrics:
- VQA Accuracy
- BLEU (BLEU-1 to BLEU-4)
- ROUGE-L
- METEOR
- CIDEr
- BERTScore (for both answers and explanations)
- SPICE
- Semantic Similarity
- G-Eval (GPT-4o-mini based evaluation)

Example Usage:
```python
from src.eval import EvalModule, GEvaluator
from openai import OpenAI

# Initialize evaluators
eval_module = EvalModule(device="cuda")
g_evaluator = GEvaluator(OpenAI(api_key="your-key"))

# Prepare data
results = [
    {
        "question_id": "1",
        "question": "What color is the car?",
        "final_answer": "red",
        "generated_explanation": "The car appears to be red in color.",
        "ground_truth_answer": "red"
    }
]

ground_truth_explanations = {
    "1": [
        "The car is red colored.",
        "The vehicle has a red paint job.",
        "The automobile appears red."
    ]
}

# Run comprehensive evaluation
results = eval_module.evaluate_comprehensive(
    results=results,
    ground_truth_explanations=ground_truth_explanations,
    include_g_eval=True,
    g_evaluator=g_evaluator
)

# Save report
eval_module.save_evaluation_report(results, "evaluation_report.json")
```
"""

from .eval_module import EvalModule
from .g_evaluator import GEvaluator

__all__ = ["EvalModule", "GEvaluator"] 