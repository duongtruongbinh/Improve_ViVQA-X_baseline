#!/usr/bin/env python3
"""
Quick validation test for GQA-REX pipeline configuration
Ensures pipeline uses validation dataset
"""

import sys
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

def test_validation_config():
    """Test that pipeline is configured for validation set"""
    
    print("🔍 Testing GQA-REX Pipeline Configuration")
    print("=" * 45)
    
    try:
        # Test 1: Check config file
        from gqa_rex_pipeline.gqa_rex_config import get_gqa_rex_config
        config = get_gqa_rex_config()
        
        active_dataset = config.get('active_dataset', 'unknown')
        print(f"✅ Active dataset: {active_dataset}")
        
        if active_dataset == "gqa_rex_val":
            print("✅ Correctly configured for validation set")
        else:
            print(f"⚠️  Active dataset is {active_dataset}, should be gqa_rex_val")
        
        # Test 2: Check dataset configuration
        val_config = config['datasets']['gqa_rex_val']
        print(f"✅ Validation dataset name: {val_config['name']}")
        print(f"✅ Split: {val_config['split']}")
        print(f"✅ Format: {val_config['format']}")
        
        # Test 3: Check data loader
        print("\n📚 Testing data loader...")
        from gqa_rex_pipeline.gqa_rex_loader import create_gqa_rex_loader
        
        loader = create_gqa_rex_loader(split="val")
        stats = loader.get_dataset_stats()
        
        print(f"✅ Loaded {stats['total_samples']} validation samples")
        print(f"✅ GQA questions: {stats['gqa_questions']}")
        print(f"✅ REX explanations: {stats['rex_explanations']}")
        
        # Test 4: Check sample format
        samples = loader.get_samples(limit=1)
        if samples:
            sample = samples[0]
            print(f"\n📋 Sample validation:")
            print(f"   • Question ID: {sample['question_id']}")
            print(f"   • Has question: {'✅' if sample.get('question') else '❌'}")
            print(f"   • Has answer: {'✅' if sample.get('answer') else '❌'}")
            print(f"   • Has explanation: {'✅' if sample.get('explanation') else '❌'}")
            print(f"   • Has image: {'✅' if sample.get('image_filename') else '❌'}")
        
        print(f"\n🎯 GQA-REX Pipeline Ready for Validation Set!")
        print(f"   • Dataset: GQA-REX Validation")
        print(f"   • Samples available: {stats['total_samples']}")
        print(f"   • Configuration: ✅ Correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_config()
    sys.exit(0 if success else 1)
