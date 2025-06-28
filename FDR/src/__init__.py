"""
MVKB-X (Multi-View Knowledge Base with eXplanation) Pipeline
Core module for VQA with explainable reasoning

Components:
- VerifierAgent: VLM + GroundingDINO + DAM for visual verification
- StrategistAgent: LLM for MVKB construction and hypothesis generation  
- SynthesizerAgent: Algorithm 2 weighted voting mechanism
- ExplanationAgent: Natural language explanation generation
- EvalModule: Comprehensive evaluation metrics for VQA + explanations
"""

from .agents import VerifierAgent, StrategistAgent, SynthesizerAgent, ExplanationAgent
from .eval import EvalModule
from .g_evaluator import GEvaluator
from .pipeline import run_mvkb_x_pipeline, run_fdr_pipeline

__all__ = [
    # Core Agents
    'VerifierAgent',
    'StrategistAgent', 
    'SynthesizerAgent',
    'ExplanationAgent',
    
    # Evaluation
    'EvalModule',
    'GEvaluator',
    
    # Pipeline Functions
    'run_mvkb_x_pipeline',
    'run_fdr_pipeline'  # Legacy compatibility
]

__version__ = "1.0.0"
__author__ = "MVKB-X Research Team" 