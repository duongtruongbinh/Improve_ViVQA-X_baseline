"""
Module X - ViVQA-X evaluation module for Multi-Agent VQA framework

This module provides specialized functionality for evaluating Multi-Agent VQA systems
on the ViVQA-X dataset (Vietnamese Visual Question Answering with Explanations).
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent VQA Team"

# Import main components
from .dataloader import ViVQAXDataset, ViVQAXDataLoader
from .inference import ViVQAXInference  
from .evaluation import ViVQAXEvaluator
from .utils import load_config, save_json, load_json, get_timestamp

__all__ = [
    'ViVQAXDataset', 'ViVQAXDataLoader', 'ViVQAXInference', 
    'ViVQAXEvaluator', 'load_config', 'save_json', 'load_json', 
    'get_timestamp'
]
