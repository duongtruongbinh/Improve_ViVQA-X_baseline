"""
GQA-REX Configuration for FDR Pipeline
Configuration settings and utilities for GQA-REX dataset integration
"""

import logging
from typing import Dict, Any


def get_gqa_rex_config() -> Dict[str, Any]:
    """
    Get standard GQA-REX configuration for FDR pipeline.
    
    Returns:
        Configuration dictionary for GQA-REX dataset
    """
    return {
        "gqa_rex_train": {
            "name": "GQA-REX Training Set",
            "data_path": "/mnt/VLAI_data/GQA/train_balanced_questions.json",
            "rex_path": "/mnt/VLAI_data/GQA-REX/converted_explanation_train.json",
            "scene_graph_path": "/mnt/VLAI_data/GQA/train_sceneGraphs.json",
            "image_dir": "/mnt/VLAI_data/GQA/images",
            "format": "gqa_rex",
            "language": "english",
            "split": "train",
            "use_explanations": True,
            "use_scene_graphs": True,
            "description": "GQA training set with GQA-REX reasoning explanations"
        },
        
        "gqa_rex_val": {
            "name": "GQA-REX Validation Set",
            "data_path": "/mnt/VLAI_data/GQA/val_balanced_questions.json",
            "rex_path": "/mnt/VLAI_data/GQA-REX/converted_explanation_val.json",
            "scene_graph_path": "/mnt/VLAI_data/GQA/val_sceneGraphs.json",
            "image_dir": "/mnt/VLAI_data/GQA/images",
            "format": "gqa_rex",
            "language": "english", 
            "split": "validation",
            "use_explanations": True,
            "use_scene_graphs": True,
            "description": "GQA validation set with GQA-REX reasoning explanations"
        }
    }


def create_gqa_rex_config_yaml() -> str:
    """
    Create a YAML configuration section for GQA-REX datasets.
    
    Returns:
        YAML configuration string that can be added to config.yaml
    """
    yaml_config = '''
# GQA-REX Dataset Configurations
datasets:
  gqa_rex_train:
    name: "GQA-REX Training Set"
    data_path: "/mnt/VLAI_data/GQA/train_balanced_questions.json"
    rex_path: "/mnt/VLAI_data/GQA-REX/converted_explanation_train.json"
    scene_graph_path: "/mnt/VLAI_data/GQA/train_sceneGraphs.json"
    image_dir: "/mnt/VLAI_data/GQA/images"
    format: "gqa_rex"
    language: "english"
    split: "train"
    use_explanations: true
    use_scene_graphs: true
    description: "GQA training set with GQA-REX reasoning explanations"
    
  gqa_rex_val:
    name: "GQA-REX Validation Set"
    data_path: "/mnt/VLAI_data/GQA/val_balanced_questions.json"
    rex_path: "/mnt/VLAI_data/GQA-REX/converted_explanation_val.json"
    scene_graph_path: "/mnt/VLAI_data/GQA/val_sceneGraphs.json"
    image_dir: "/mnt/VLAI_data/GQA/images"
    format: "gqa_rex"
    language: "english"
    split: "validation"
    use_explanations: true
    use_scene_graphs: true
    description: "GQA validation set with GQA-REX reasoning explanations"

# Processing Configuration for GQA-REX
processing_config:
  # Number of samples to process (0 for all)
  num_samples: 0
  
  # Enable detailed logging
  detailed_logging: true
  
  # Batch processing settings
  batch_size: 1
  
  # Scene graph integration
  use_scene_graphs: true
  
  # Explanation processing
  max_explanation_length: 500
  preprocess_explanations: true

# GQA-REX specific evaluation settings
evaluation_config:
  # Enable GQA-REX specific metrics
  enable_gqa_rex_metrics: true
  
  # Evaluate explanation quality
  evaluate_explanations: true
  
  # Compare with ground truth explanations
  compare_explanations: true
  
  # Scene graph reasoning evaluation
  evaluate_scene_graph_reasoning: true

# Backend configuration optimized for GQA-REX
backend_config:
  # Model settings
  model_name: "qwen2.5-vl-7b-instruct"
  temperature: 0.1
  max_tokens: 2048
  
  # Vision model settings for scene understanding
  vision_model_preference: "vl"  # Use VL model for complex visual reasoning
  
  # Language model settings for explanation generation
  language_model_preference: "llm"  # Use LLM for text processing
'''
    
    return yaml_config.strip()


def validate_gqa_rex_paths(config: Dict[str, Any]) -> bool:
    """
    Validate that all required GQA-REX paths exist.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        True if all paths are valid, False otherwise
    """
    import os
    
    required_paths = [
        'data_path',      # GQA questions
        'rex_path',       # GQA-REX explanations  
        'image_dir',      # Image directory
    ]
    
    optional_paths = [
        'scene_graph_path'  # Scene graphs (optional)
    ]
    
    # Check required paths
    for path_key in required_paths:
        path = config.get(path_key)
        if not path:
            logging.error(f"❌ Missing required path: {path_key}")
            return False
        
        if not os.path.exists(path):
            logging.error(f"❌ Path does not exist: {path_key} = {path}")
            return False
        
        logging.info(f"✅ Validated path: {path_key} = {path}")
    
    # Check optional paths
    for path_key in optional_paths:
        path = config.get(path_key)
        if path:
            if os.path.exists(path):
                logging.info(f"✅ Optional path found: {path_key} = {path}")
            else:
                logging.warning(f"⚠️ Optional path missing: {path_key} = {path}")
        else:
            logging.info(f"ℹ️ Optional path not configured: {path_key}")
    
    return True


def get_sample_gqa_rex_questions() -> Dict[str, Any]:
    """
    Get some sample question IDs for testing GQA-REX integration.
    
    Returns:
        Dictionary with sample questions for different reasoning types
    """
    return {
        "spatial_reasoning": [
            "07196281",  # Is the green lawn large or small?
            "09987143",  # Is the white house to the left or to the right of the person that wears a hat?
        ],
        "object_recognition": [
            "00235513",  # What is the brown animal?
            "05515938",  # What animal is shown? (example)
        ],
        "attribute_questions": [
            "12470721",  # Which color is the cap?
            "08171823",  # What color is the bat?
        ],
        "counting": [
            "077772",    # Are there airplanes in the sky?
            "08578425",  # Do you see both windows and cars?
        ],
        "complex_reasoning": [
            "16765501",  # Do you see any guys to the right of the boy?
            "18116487",  # Is the person to the right of the man wearing a cap?
        ]
    }
