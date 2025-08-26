"""
GQA-REX Dataset Loader for FDR Pipeline
Handles loading and preprocessing GQA + GQA-REX format data for multi-agent VQA processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter


class GQAREXLoader:
    """
    GQA-REX dataset loader combining GQA questions/answers with GQA-REX explanations.
    Supports both train and validation splits.
    """
    
    def __init__(self, 
                 gqa_data_path: str, 
                 rex_data_path: str, 
                 image_dir: str,
                 scene_graph_path: Optional[str] = None):
        """
        Initialize GQA-REX loader.
        
        Args:
            gqa_data_path: Path to GQA questions JSON file (e.g., train_balanced_questions.json)
            rex_data_path: Path to GQA-REX explanations JSON file (e.g., converted_explanation_train.json)
            image_dir: Path to GQA images directory (e.g., /mnt/VLAI_data/GQA/images)
            scene_graph_path: Optional path to scene graphs JSON file
        """
        self.gqa_data_path = gqa_data_path
        self.rex_data_path = rex_data_path
        self.image_dir = Path(image_dir)
        self.scene_graph_path = scene_graph_path
        
        self.gqa_data = None
        self.rex_data = None
        self.scene_graphs = None
        self.linked_samples = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load GQA and GQA-REX data and find linked samples"""
        try:
            # Load GQA questions
            logging.info(f"Loading GQA data from: {self.gqa_data_path}")
            with open(self.gqa_data_path, 'r') as f:
                self.gqa_data = json.load(f)
            logging.info(f"✅ Loaded {len(self.gqa_data)} GQA questions")
            
            # Load GQA-REX explanations
            logging.info(f"Loading GQA-REX data from: {self.rex_data_path}")
            with open(self.rex_data_path, 'r') as f:
                self.rex_data = json.load(f)
            logging.info(f"✅ Loaded {len(self.rex_data)} GQA-REX explanations")
            
            # Load scene graphs if provided
            if self.scene_graph_path and os.path.exists(self.scene_graph_path):
                logging.info(f"Loading scene graphs from: {self.scene_graph_path}")
                with open(self.scene_graph_path, 'r') as f:
                    self.scene_graphs = json.load(f)
                logging.info(f"✅ Loaded {len(self.scene_graphs)} scene graphs")
            
            # Validate image directory
            if not self.image_dir.exists():
                logging.error(f"❌ Image directory not found: {self.image_dir}")
                raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            
            # Count available images
            image_files = list(self.image_dir.glob("*.jpg"))
            logging.info(f"📸 Found {len(image_files)} images in {self.image_dir}")
            
            # Find linked samples (questions that exist in both datasets)
            self._find_linked_samples()
            
        except Exception as e:
            logging.error(f"Failed to load GQA-REX data: {e}")
            raise
    
    def _find_linked_samples(self):
        """Find samples that exist in both GQA and GQA-REX datasets"""
        gqa_keys = set(self.gqa_data.keys())
        rex_keys = set(self.rex_data.keys())
        
        # Find intersection
        common_keys = gqa_keys.intersection(rex_keys)
        logging.info(f"🔗 Found {len(common_keys)} linked samples between GQA and GQA-REX")
        
        # Create linked samples list
        valid_samples = []
        missing_images = 0
        
        for question_id in common_keys:
            gqa_sample = self.gqa_data[question_id]
            rex_explanation = self.rex_data[question_id]
            
            # Get image info
            image_id = gqa_sample.get('imageId')
            if not image_id:
                continue
                
            image_path = self.image_dir / f"{image_id}.jpg"
            
            # Skip if image doesn't exist
            if not image_path.exists():
                missing_images += 1
                continue
            
            # Get scene graph if available
            scene_graph = None
            if self.scene_graphs and image_id in self.scene_graphs:
                scene_graph = self.scene_graphs[image_id]
            
            valid_samples.append({
                'question_id': question_id,
                'gqa_data': gqa_sample,
                'rex_explanation': rex_explanation,
                'image_path': str(image_path),
                'image_id': image_id,
                'scene_graph': scene_graph
            })
        
        self.linked_samples = valid_samples
        logging.info(f"✅ Created {len(self.linked_samples)} valid linked samples")
        if missing_images > 0:
            logging.warning(f"⚠️ Skipped {missing_images} samples due to missing images")
    
    def get_sample_count(self) -> int:
        """Get total number of linked samples"""
        return len(self.linked_samples)
    
    def get_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get samples in FDR pipeline format.
        
        Args:
            limit: Maximum number of samples to return (None for all)
            
        Returns:
            List of samples formatted for FDR pipeline
        """
        samples_to_process = self.linked_samples[:limit] if limit else self.linked_samples
        formatted_samples = []
        
        for sample in samples_to_process:
            try:
                gqa = sample['gqa_data']
                rex_explanation = sample['rex_explanation']
                
                # Extract GQA information
                question = gqa.get('question', '')
                answer = gqa.get('answer', '')
                full_answer = gqa.get('fullAnswer', '')
                image_id = gqa.get('imageId', '')
                
                # Get question types and metadata
                question_types = gqa.get('types', {})
                semantic_type = question_types.get('semantic', 'unknown')
                structural_type = question_types.get('structural', 'unknown')
                
                # Get semantic operations if available
                semantic_operations = gqa.get('semantic', [])
                
                # Format for FDR pipeline
                formatted_sample = {
                    'question_id': sample['question_id'],
                    'question': question,
                    'answer': answer,
                    'image_path': sample['image_path'],
                    'image_filename': f"{image_id}.jpg",
                    'image_id': image_id,
                    
                    # GQA-REX explanation
                    'explanation': rex_explanation,
                    
                    # Additional GQA metadata
                    'full_answer': full_answer,
                    'question_types': question_types,
                    'semantic_type': semantic_type,
                    'structural_type': structural_type,
                    'semantic_operations': semantic_operations,
                    
                    # Scene graph (if available)
                    'scene_graph': sample.get('scene_graph'),
                    
                    # Dataset info
                    'dataset': 'gqa-rex',
                    'original_gqa': gqa,  # Keep original for reference
                }
                
                formatted_samples.append(formatted_sample)
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to format sample {sample.get('question_id', 'unknown')}: {e}")
                continue
        
        logging.info(f"📋 Formatted {len(formatted_samples)} samples for FDR pipeline")
        return formatted_samples
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        if not self.linked_samples:
            return {}
        
        # Analyze question types
        semantic_types = []
        structural_types = []
        
        for sample in self.linked_samples:
            gqa = sample['gqa_data']
            types = gqa.get('types', {})
            semantic_types.append(types.get('semantic', 'unknown'))
            structural_types.append(types.get('structural', 'unknown'))
        
        semantic_counts = Counter(semantic_types)
        structural_counts = Counter(structural_types)
        
        return {
            'total_samples': len(self.linked_samples),
            'gqa_questions': len(self.gqa_data) if self.gqa_data else 0,
            'rex_explanations': len(self.rex_data) if self.rex_data else 0,
            'scene_graphs': len(self.scene_graphs) if self.scene_graphs else 0,
            'semantic_types': dict(semantic_counts),
            'structural_types': dict(structural_counts),
            'top_semantic_types': semantic_counts.most_common(5),
            'top_structural_types': structural_counts.most_common(5),
        }


# Utility function for easy instantiation
def create_gqa_rex_loader(split: str = "train", 
                         base_gqa_path: str = "/mnt/VLAI_data/GQA",
                         base_rex_path: str = "/mnt/VLAI_data/GQA-REX") -> GQAREXLoader:
    """
    Create GQA-REX loader with standard paths.
    
    Args:
        split: Dataset split ("train" or "val")
        base_gqa_path: Base path to GQA dataset
        base_rex_path: Base path to GQA-REX dataset
        
    Returns:
        GQAREXLoader instance
    """
    # Define file paths based on split
    if split == "train":
        gqa_questions = f"{base_gqa_path}/train_balanced_questions.json"
        rex_explanations = f"{base_rex_path}/converted_explanation_train.json"
        scene_graphs = f"{base_gqa_path}/train_sceneGraphs.json"
    elif split == "val":
        gqa_questions = f"{base_gqa_path}/val_balanced_questions.json"
        rex_explanations = f"{base_rex_path}/converted_explanation_val.json"
        scene_graphs = f"{base_gqa_path}/val_sceneGraphs.json"
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
    
    image_dir = f"{base_gqa_path}/images"
    
    return GQAREXLoader(
        gqa_data_path=gqa_questions,
        rex_data_path=rex_explanations,
        image_dir=image_dir,
        scene_graph_path=scene_graphs
    )
