"""
CLEVR-X Dataset Loader for FDR Pipeline
Handles loading and preprocessing CLEVR-X format data for multi-agent VQA processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class CLEVRXLoader:
    """
    CLEVR-X dataset loader with support for explanations and program-based reasoning.
    """
    
    def __init__(self, data_path: str, image_dir: str):
        """
        Initialize CLEVR-X loader.
        
        Args:
            data_path: Path to CLEVR-X JSON file (e.g., CLEVR_val_explanations_v0.7.10.json)
            image_dir: Path to CLEVR images directory (e.g., /path/to/CLEVR/images/val)
        """
        self.data_path = data_path
        self.image_dir = Path(image_dir)
        self.data = None
        self.questions = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load CLEVR-X JSON data"""
        try:
            logging.info(f"Loading CLEVR-X data from: {self.data_path}")
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            
            # Extract questions
            self.questions = self.data.get('questions', [])
            
            logging.info(f"✅ Loaded {len(self.questions)} CLEVR-X questions")
            logging.info(f"Dataset info: {self.data.get('info', {})}")
            
            # Validate image directory
            if not self.image_dir.exists():
                logging.error(f"❌ Image directory not found: {self.image_dir}")
                raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            
            # Count available images
            image_files = list(self.image_dir.glob("*.png"))
            logging.info(f"📸 Found {len(image_files)} images in {self.image_dir}")
            
        except Exception as e:
            logging.error(f"Failed to load CLEVR-X data: {e}")
            raise
            
    def get_sample_count(self) -> int:
        """Get total number of samples"""
        return len(self.questions)
    
    def get_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get samples in FDR pipeline format.
        
        Args:
            limit: Maximum number of samples to return (None for all)
            
        Returns:
            List of samples formatted for FDR pipeline
        """
        questions = self.questions[:limit] if limit else self.questions
        samples = []
        
        for i, q in enumerate(questions):
            try:
                # Construct full image path
                image_filename = q.get('image_filename', '')
                image_path = str(self.image_dir / image_filename)
                
                # Verify image exists
                if not os.path.exists(image_path):
                    logging.warning(f"⚠️ Image not found: {image_path}")
                    continue
                
                # Extract primary explanation (first factual explanation)
                explanations = q.get('factual_explanation', [])
                primary_explanation = explanations[0] if explanations else ""
                
                # Format sample for FDR pipeline
                sample = {
                    'question': q.get('question', ''),
                    'answer': q.get('answer', ''),
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'explanation': primary_explanation,
                    'all_explanations': explanations,
                    'counter_factual_explanation': q.get('counter_factual_explanation', []),
                    'program': q.get('program', []),
                    'image_index': q.get('image_index', i),
                    'question_index': q.get('question_index', i),
                    'question_family_index': q.get('question_family_index', -1),
                    'split': q.get('split', 'unknown'),
                    'dataset_type': 'clevr-x'
                }
                
                samples.append(sample)
                
            except Exception as e:
                logging.warning(f"⚠️ Skipping sample {i}: {e}")
                continue
        
        logging.info(f"✅ Prepared {len(samples)} CLEVR-X samples")
        return samples
    
    def get_sample_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a single sample by index"""
        if 0 <= index < len(self.questions):
            samples = self.get_samples()
            return samples[index] if index < len(samples) else None
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': 'CLEVR-X',
            'total_questions': len(self.questions),
            'data_path': self.data_path,
            'image_dir': str(self.image_dir),
            'dataset_info': self.data.get('info', {}) if self.data else {},
            'format': 'clevr-x',
            'has_explanations': True,
            'has_programs': True,
            'explanation_types': ['factual_explanation', 'counter_factual_explanation']
        }
    
    def validate_samples(self, limit: int = 10) -> Dict[str, Any]:
        """
        Validate first few samples to check data quality.
        
        Args:
            limit: Number of samples to validate
            
        Returns:
            Validation report
        """
        samples = self.get_samples(limit=limit)
        
        validation_report = {
            'total_samples_checked': len(samples),
            'valid_samples': 0,
            'missing_images': 0,
            'missing_questions': 0,
            'missing_answers': 0,
            'missing_explanations': 0,
            'issues': []
        }
        
        for i, sample in enumerate(samples):
            is_valid = True
            
            # Check image file exists
            if not os.path.exists(sample['image_path']):
                validation_report['missing_images'] += 1
                validation_report['issues'].append(f"Sample {i}: Image not found - {sample['image_path']}")
                is_valid = False
            
            # Check question
            if not sample.get('question', '').strip():
                validation_report['missing_questions'] += 1
                validation_report['issues'].append(f"Sample {i}: Empty question")
                is_valid = False
            
            # Check answer
            if not sample.get('answer', '').strip():
                validation_report['missing_answers'] += 1
                validation_report['issues'].append(f"Sample {i}: Empty answer")
                is_valid = False
            
            # Check explanation
            if not sample.get('explanation', '').strip():
                validation_report['missing_explanations'] += 1
                validation_report['issues'].append(f"Sample {i}: Empty explanation")
                is_valid = False
            
            if is_valid:
                validation_report['valid_samples'] += 1
        
        validation_report['validation_success_rate'] = (
            validation_report['valid_samples'] / len(samples) * 100 if samples else 0
        )
        
        return validation_report


def test_clevr_x_loader():
    """Test function for CLEVR-X loader"""
    print("=== Testing CLEVR-X Loader ===")
    
    # Initialize loader
    data_path = "/mnt/VLAI_data/CLEVR-X/CLEVR_val_explanations_v0.7.10.json"
    image_dir = "/mnt/VLAI_data/CLEVR/CLEVR_v1.0/images/val"
    
    try:
        loader = CLEVRXLoader(data_path, image_dir)
        
        # Get basic info
        info = loader.get_info()
        print(f"Dataset Info: {info}")
        
        # Get first few samples
        samples = loader.get_samples(limit=3)
        print(f"\n=== First 3 Samples ===")
        
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['answer']}")
            print(f"  Image: {sample['image_filename']}")
            print(f"  Explanation: {sample['explanation'][:100]}...")
            print(f"  Program steps: {len(sample['program'])}")
            print(f"  Image exists: {os.path.exists(sample['image_path'])}")
        
        # Validation test
        validation = loader.validate_samples(limit=10)
        print(f"\n=== Validation Report ===")
        print(f"Success rate: {validation['validation_success_rate']:.1f}%")
        print(f"Valid samples: {validation['valid_samples']}/{validation['total_samples_checked']}")
        if validation['issues']:
            print("Issues found:")
            for issue in validation['issues'][:3]:
                print(f"  - {issue}")
        
        print("✅ CLEVR-X Loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CLEVR-X Loader test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when executed directly
    test_clevr_x_loader()
