"""
FDR (Faithful Decomposed Reasoning) Core Module
Provides the main components for visual question answering with explanations.

Core agents:
- VerifierAgent: Visual verification and object detection
- StrategistAgent: MVKB construction and explanation generation  
- SynthesizerAgent: Logical reasoning and answer synthesis
"""

from .agents import VerifierAgent, StrategistAgent, SynthesizerAgent
from .pipeline import run_fdr_pipeline

__version__ = "2.0.0"

__all__ = [
    # Core agents
    'VerifierAgent',
    'StrategistAgent', 
    'SynthesizerAgent',
    
    # Pipeline
    'run_fdr_pipeline',
    
    # Version
    '__version__'
] 