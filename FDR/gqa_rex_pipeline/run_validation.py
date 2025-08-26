#!/usr/bin/env python3
"""
Quick Run GQA-REX Validation Pipeline
Simple script to run pipeline on validation set
"""

import sys
import os
from pathlib import Path

# Add FDR directory to Python path
fdr_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fdr_dir))

def run_validation_pipeline():
    """Run GQA-REX pipeline on validation set"""
    
    print("🎯 GQA-REX Validation Pipeline")
    print("=" * 35)
    
    # Default settings for validation
    print("📋 Configuration:")
    print("   • Dataset: GQA-REX Validation")
    print("   • Samples: 100 (for testing)")
    print("   • Backend: vLLM")
    print("   • Evaluation: Enabled")
    print("")
    
    # Build command
    cmd = [
        "python", 
        str(fdr_dir / "main_gqa_rex.py"),
        "--dataset", "gqa_rex_val",
        "--backend", "vllm", 
        "--evaluate",
        "--samples", "100"
    ]
    
    print("🚀 Starting pipeline...")
    print(f"   Command: {' '.join(cmd)}")
    print("")
    
    # Change to FDR directory
    os.chdir(str(fdr_dir))
    
    # Execute pipeline
    import subprocess
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed with exit code: {result.returncode}")
        
    return result.returncode == 0

if __name__ == "__main__":
    success = run_validation_pipeline()
    sys.exit(0 if success else 1)
