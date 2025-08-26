"""
GQA-REX Pipeline Components for FDR Framework
Specialized components for handling GQA and GQA-REX datasets
"""

from .gqa_rex_loader import GQAREXLoader
from .gqa_rex_config import get_gqa_rex_config

__all__ = [
    'GQAREXLoader',
    'get_gqa_rex_config'
]
