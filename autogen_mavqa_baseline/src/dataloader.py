import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class GQADataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset_split = args['datasets']['gqa_dataset_split']

        if self.dataset_split == 'val':
            self.questions_file = self.args['datasets']['gqa_val_questions_file']
        elif self.dataset_split == 'val-subset':
            self.questions_file = self.args['datasets']['gqa_val_subset_questions_file']
        else:
            self.questions_file = self.args['datasets']['gqa_test_questions_file']

        self.images_dir = self.args['datasets']['gqa_images_dir']
        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if self.dataset_split == 'val-subset':
            annot = self.questions[idx]
            image_id = annot['image']
            image_path = os.path.join(self.images_dir, image_id)
        else:
            annot = self.questions[list(self.questions.keys())[idx]]
            image_id = annot['imageId']
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        question = annot['question']
        answer = annot['answer']

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'question_id': -1, 'answer': answer}


class VQAv2Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.transform = transform
        self.dataset_split = args['datasets']['vqa_v2_dataset_split']

        self.questions_file = None
        self.images_dir = None
        self.questions = []
        self.answers_by_qid = {}
        
        is_minival_self_contained = False

        if self.dataset_split == 'val':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_val_questions_file']
            
            if "minival.json" in self.questions_file:
                is_minival_self_contained = True
            
            if is_minival_self_contained:
                with open(self.questions_file, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict) and 'questions' in loaded_data:
                        self.questions = loaded_data['questions']
                    elif isinstance(loaded_data, list):
                        self.questions = loaded_data
                    else:
                        raise ValueError(f"Unsupported structure in VQA questions_file: {self.questions_file}")
            else:
                answers_file_path = self.args['datasets']['vqa_v2_val_annotations_file']
                with open(self.questions_file, 'r') as f:
                    self.questions = json.load(f).get('questions', [])
                with open(answers_file_path, 'r') as f:
                    annotations = json.load(f).get('annotations', [])
                if annotations:
                    self.answers_by_qid = {ans['question_id']: ans for ans in annotations}

        elif self.dataset_split == 'rest-val':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_rest_val_questions_file']
            answers_file_path = self.args['datasets']['vqa_v2_rest_val_annotations_file']
            with open(self.questions_file, 'r') as f:
                 self.questions = json.load(f).get('questions', [])
            with open(answers_file_path, 'r') as f:
                self.answers_by_qid = json.load(f)

        elif self.dataset_split == 'val1000':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_val1000_questions_file']
            answers_file_path = self.args['datasets']['vqa_v2_val1000_annotations_file']
            with open(self.questions_file, 'r') as f:
                self.questions = json.load(f).get('questions', [])
            with open(answers_file_path, 'r') as f:
                annotations = json.load(f).get('annotations', [])
            if annotations:
                 self.answers_by_qid = {ans['question_id']: ans for ans in annotations}
        
        elif self.dataset_split == 'test' or self.dataset_split == 'test-dev':
            self.images_dir = self.args['datasets']['vqa_v2_test_images_dir']
            if self.dataset_split == 'test':
                self.questions_file = self.args['datasets']['vqa_v2_test_questions_file']
            else: 
                self.questions_file = self.args['datasets']['vqa_v2_test_dev_questions_file']
            with open(self.questions_file, 'r') as f:
                self.questions = json.load(f).get('questions', [])
        else:
            raise ValueError(f"Unsupported VQA dataset_split: {self.dataset_split}")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if not self.questions:
             raise IndexError("VQAv2Dataset: Questions not loaded or empty.")
        
        annot = self.questions[idx]
        image_id = annot['image_id']
        question = annot['question']
        question_id = annot['question_id']
        
        image_filename = ""
        if self.dataset_split in ['val', 'val1000', 'rest-val']:
            image_filename = f"COCO_val2014_{image_id:012}.jpg"
        elif self.dataset_split in ['test', 'test-dev']:
            image_filename = f"COCO_test2015_{image_id:012}.jpg"
        else:
             raise ValueError(f"Cannot determine image filename for VQA dataset_split: {self.dataset_split}")

        image_path = os.path.join(self.images_dir, image_filename)
        answer = ""

        is_minival_self_contained = False
        if self.dataset_split == 'val' and "minival.json" in self.questions_file:
             is_minival_self_contained = True

        if is_minival_self_contained:
            if 'multiple_choice_answer' in annot:
                answer = annot['multiple_choice_answer']
            elif 'answers' in annot and annot['answers']:
                answer = annot['answers'][0]['answer']
        elif self.dataset_split == 'rest-val':
            ans_data = self.answers_by_qid.get(str(question_id))
            if ans_data and 'multiple_choice_answer' in ans_data:
                answer = ans_data['multiple_choice_answer']
        elif self.answers_by_qid: 
            ans_data = self.answers_by_qid.get(question_id)
            if ans_data and 'multiple_choice_answer' in ans_data:
                answer = ans_data['multiple_choice_answer']
        
        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'question_id': question_id, 'answer': answer}