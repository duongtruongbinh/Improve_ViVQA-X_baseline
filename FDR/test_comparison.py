#!/usr/bin/env python3
# Test script for comparison pipeline
import os
import sys
import json
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent
sys.path.insert(0, str(fdr_dir))

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    
    try:
        from quick_run import load_rere_results
        print("✅ load_rere_results imported successfully")
    except Exception as e:
        print(f"❌ load_rere_results import failed: {e}")
        return False
    
    try:
        from src.comparison_pipeline import run_comparison_pipeline
        print("✅ run_comparison_pipeline imported successfully")
    except Exception as e:
        print(f"❌ run_comparison_pipeline import failed: {e}")
        return False
    
    return True

def test_rere_loading():
    """Test ReRe results loading"""
    print("\n🧪 Testing ReRe results loading...")
    
    try:
        from quick_run import load_rere_results
        rere_results = load_rere_results()
        
        if rere_results:
            print(f"✅ Successfully loaded {len(rere_results)} ReRe results")
            print(f"First sample keys: {list(rere_results[0].keys())}")
            print(f"Sample question_id: {rere_results[0].get('question_id')}")
            return rere_results
        else:
            print("❌ Failed to load ReRe results")
            return None
    except Exception as e:
        print(f"❌ Error loading ReRe results: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_config_loading():
    """Test config loading"""
    print("\n🧪 Testing config loading...")
    
    try:
        from src.comparison_pipeline import load_config
        config = load_config()
        
        print(f"✅ Config loaded successfully")
        print(f"Active dataset: {config.get('active_dataset')}")
        print(f"Enable DAM: {config.get('agents_config', {}).get('verifier', {}).get('enable_dam')}")
        print(f"Num samples: {config.get('processing_config', {}).get('num_samples')}")
        
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_initialization():
    """Test agent initialization without running full pipeline"""
    print("\n🧪 Testing agent initialization...")
    
    try:
        from src.comparison_pipeline import load_config
        from src.agents import VerifierAgent
        
        config = load_config()
        agents_config = config.get('agents_config', {})
        verifier_config = agents_config.get('verifier', {})
        
        print(f"Verifier config: {verifier_config}")
        
        # Try to initialize VerifierAgent with DAM disabled
        verifier = VerifierAgent(
            temperature=verifier_config.get('temperature', 0.7),
            max_tokens=verifier_config.get('max_tokens', 1000),
            use_vllm=True,
            enable_dam=False,  # Force disable DAM
            enable_groundingdino=verifier_config.get('enable_groundingdino', True),
            groundingdino_docker=False,
            model_preference="vlm"
        )
        
        print("✅ VerifierAgent initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comparison pipeline tests...")
    print("-" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Stopping.")
        return
    
    # Test 2: ReRe loading
    rere_results = test_rere_loading()
    if not rere_results:
        print("\n❌ ReRe loading failed. Stopping.")
        return
    
    # Test 3: Config loading
    config = test_config_loading()
    if not config:
        print("\n❌ Config loading failed. Stopping.")
        return
    
    # Test 4: Agent initialization
    if not test_agent_initialization():
        print("\n❌ Agent initialization failed. Stopping.")
        return
    
    print("\n🎉 All tests passed! Pipeline should work.")
    print("\n📋 Next steps:")
    print("  - Run: python quick_run.py")
    print("  - Or run with limited samples by setting num_samples in config.yaml")

if __name__ == "__main__":
    main()
