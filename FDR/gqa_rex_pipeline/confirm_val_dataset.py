#!/usr/bin/env python3
"""
Simple test to confirm GQA-REX validation dataset loading
"""

import sys
import logging
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

from gqa_rex_pipeline.gqa_rex_loader import create_gqa_rex_loader

def test_val_dataset():
    """Simple test for validation dataset"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Confirming GQA-REX Validation Dataset")
    print("=" * 40)
    
    try:
        # Load validation set
        print("📚 Loading GQA-REX validation dataset...")
        val_loader = create_gqa_rex_loader(split="val")
        
        # Get statistics
        stats = val_loader.get_dataset_stats()
        
        print("✅ Validation dataset loaded successfully!")
        print(f"   • Total samples: {stats['total_samples']:,}")
        print(f"   • GQA questions: {stats['gqa_questions']:,}")  
        print(f"   • REX explanations: {stats['rex_explanations']:,}")
        print(f"   • Scene graphs: {stats.get('scene_graphs', 0):,}")
        
        # Test sample loading
        print("\n📋 Testing sample loading...")
        samples = val_loader.get_samples(limit=3)
        
        for i, sample in enumerate(samples, 1):
            print(f"   Sample {i}: Q{sample['question_id']} - {sample['answer']}")
            
        print(f"\n🎯 CONFIRMATION: Pipeline will use GQA-REX VALIDATION set")
        print(f"   • Available samples: {stats['total_samples']:,}")
        print(f"   • Dataset split: validation")
        print(f"   • Ready for evaluation: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_val_dataset()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Validation dataset test")
    sys.exit(0 if success else 1)
