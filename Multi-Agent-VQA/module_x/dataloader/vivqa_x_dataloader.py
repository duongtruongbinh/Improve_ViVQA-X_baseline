import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any

class ViVQAXDataset(Dataset):
    """
    Dataset class for ViVQA-X (Vietnamese Visual Question Answering with Explanations)
    """
    def __init__(self, args: Dict[str, Any], transform=None):
        self.args = args
        self.dataset_split = args['datasets']['vivqa_x_dataset_split']
        self.transform = transform
        
        # Set paths based on split
        if self.dataset_split == 'train':
            self.data_file = self.args['datasets']['vivqa_x_train_file']
        elif self.dataset_split == 'val':
            self.data_file = self.args['datasets']['vivqa_x_val_file']
        elif self.dataset_split == 'test':
            self.data_file = self.args['datasets']['vivqa_x_test_file']
        else:
            raise ValueError(f"Unknown dataset split: {self.dataset_split}")
            
        self.images_dir = self.args['datasets']['vivqa_x_images_dir']
        
        # Load data
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Apply data limiting if specified
        if self.args['datasets'].get('use_num_test_data', False):
            num_samples = self.args['datasets'].get('num_test_data', 10)
            self.data = self.data[:num_samples]
            print(f"Limited dataset to {num_samples} samples for testing")
            
        print(f"Loaded {len(self.data)} samples from {self.dataset_split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract fields from ViVQA-X format
        question_id = item.get('question_id', f"{self.dataset_split}_{idx}")
        image_name = item['image_name']
        image_path = os.path.join(self.images_dir, image_name)
        question = item['question']
        answer = item['answer']
        explanation = item.get('explanation', '')
        
        # Handle image_id extraction from image_name (e.g., "COCO_val2014_000000123456.jpg")
        if 'COCO' in image_name:
            image_id = image_name.split('_')[-1].split('.')[0]
            image_id = int(image_id)
        else:
            image_id = idx
        
        if self.args['inference']['verbose']:
            curr_data = f'image_path: {image_path} question: {question} answer: {answer}'
            print(f'\033[95m{curr_data}\033[0m')  # Colors.HEADER equivalent
            
        return {
            'image_id': image_id,
            'image_path': image_path,
            'question': question,
            'question_id': question_id,
            'answer': answer,
            'explanation': explanation,
            'image_name': image_name
        }


def load_vivqa_x_dataset(args):
    """
    Factory function to create ViVQA-X dataset
    """
    return ViVQAXDataset(args)


def get_dataloader(args):
    """
    Create dataloader for ViVQA-X dataset
    """
    dataset = load_vivqa_x_dataset(args)
    
    # Single item processing for multi-agent framework
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Multi-agent framework processes one at a time
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0]  # Return single item instead of batch
    )
    
    return dataloader, len(dataset)
