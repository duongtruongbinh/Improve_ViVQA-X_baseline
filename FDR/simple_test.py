#!/usr/bin/env python3
# Simple test to check if files are accessible
import json
import os

def test_files():
    print("🧪 Testing file access...")
    
    # Test FDR results
    fdr_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/output/fdr_results.json"
    print(f"FDR file exists: {os.path.exists(fdr_path)}")
    
    if os.path.exists(fdr_path):
        try:
            with open(fdr_path, 'r') as f:
                fdr_data = json.load(f)
            print(f"FDR samples: {len(fdr_data)}")
            if fdr_data:
                sample = fdr_data[0]
                print(f"First FDR sample keys: {list(sample.keys())}")
                print(f"Question: {sample.get('question', '')}")
                print(f"Image path: {sample.get('image_path', '')}")
                print(f"Image exists: {os.path.exists(sample.get('image_path', ''))}")
        except Exception as e:
            print(f"Error reading FDR: {e}")
    
    # Test ReRe results
    rere_path = "/home/khuonghuynh/Improve_ViVQA-X_baseline/FDR/inference_results_100_samples.json"
    print(f"\nReRe file exists: {os.path.exists(rere_path)}")
    
    if os.path.exists(rere_path):
        try:
            with open(rere_path, 'r') as f:
                rere_data = json.load(f)
            print(f"ReRe samples: {len(rere_data)}")
            if rere_data:
                sample = rere_data[0]
                print(f"First ReRe sample keys: {list(sample.keys())}")
                print(f"Question: {sample.get('question', '')}")
                print(f"Predicted answer: {sample.get('predicted_answer', '')}")
        except Exception as e:
            print(f"Error reading ReRe: {e}")

if __name__ == "__main__":
    test_files()
