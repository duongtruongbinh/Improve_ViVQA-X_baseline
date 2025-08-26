#!/usr/bin/env python3
"""
Test script for GQA-REX Loader
Quick validation of GQA-REX dataset integration
"""

import sys
import logging
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

from gqa_rex_pipeline.gqa_rex_loader import create_gqa_rex_loader

def test_gqa_rex_loader():
    """Test GQA-REX loader functionality"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing GQA-REX Loader")
    print("=" * 50)
    
    try:
        # Test validation split
        print("📚 Loading GQA-REX validation split...")
        val_loader = create_gqa_rex_loader(split="val")
        
        # Get statistics
        stats = val_loader.get_dataset_stats()
        print(f"✅ Validation split loaded successfully!")
        print(f"   • Linked samples: {stats['total_samples']}")
        print(f"   • GQA questions: {stats['gqa_questions']}")
        print(f"   • REX explanations: {stats['rex_explanations']}")
        
        if stats.get('scene_graphs'):
            print(f"   • Scene graphs: {stats['scene_graphs']}")
        
        # Show sample question types
        if stats.get('top_semantic_types'):
            print("   • Top question types:")
            for q_type, count in stats['top_semantic_types'][:3]:
                print(f"     - {q_type}: {count} samples")
        
        # Get first few samples
        print(f"\n📋 Getting first 3 samples...")
        samples = val_loader.get_samples(limit=3)
        
        for i, sample in enumerate(samples):
            print(f"\n🔍 Sample {i+1}:")
            print(f"   • Question ID: {sample['question_id']}")
            print(f"   • Question: {sample['question']}")
            print(f"   • Answer: {sample['answer']}")
            print(f"   • REX Explanation: {sample['explanation']}")
            print(f"   • Image: {sample['image_filename']}")
            print(f"   • Question Type: {sample.get('semantic_type', 'unknown')}")
        
        print(f"\n✅ GQA-REX Loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gqa_rex_loader()
    sys.exit(0 if success else 1)
