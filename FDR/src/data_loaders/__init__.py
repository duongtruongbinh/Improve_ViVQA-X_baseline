"""
Data Loaders for FDR Pipeline
Modular data loading system supporting multiple VQA dataset formats.
"""

from .clevr_x_loader import CLEVRXLoader

__all__ = [
    'CLEVRXLoader',
]
