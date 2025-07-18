#!/usr/bin/env python3
"""
Debug script to check ground truth values in results.json
"""

import json
import sys

def debug_results(results_file):
    """Debug ground truth values in results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total results: {len(results)}")
    print("="*50)
    
    for i, result in enumerate(results):
        question_id = result.get('question_id', 'unknown')
        final_answer = result.get('final_answer', 'N/A')
        ground_truth = result.get('ground_truth', 'NOT_FOUND')
        
        print(f"Sample {i+1}:")
        print(f"  Question ID: {question_id}")
        print(f"  Final Answer: '{final_answer}'")
        print(f"  Ground Truth: '{ground_truth}'")
        print(f"  Ground Truth Type: {type(ground_truth)}")
        
        if ground_truth != 'NOT_FOUND' and ground_truth is not None:
            # Test accuracy calculation
            final_lower = str(final_answer).lower().strip()
            ground_lower = str(ground_truth).lower().strip()
            is_correct = final_lower == ground_lower
            print(f"  Match: {is_correct}")
        else:
            print(f"  Match: NO_GROUND_TRUTH")
        print("-" * 30)

if __name__ == "__main__":
    results_file = "/home/huytd/multi-agent/multi-agent/FDR/output/results.json"
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    debug_results(results_file)
