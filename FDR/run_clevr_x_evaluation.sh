#!/bin/bash
# Script to run CLEVR-X multi-reference explanation evaluation
# This script activates the ma_vqa environment and runs the evaluation

echo "🚀 Starting CLEVR-X Multi-Reference Explanation Evaluation"
echo "📋 Activating ma_vqa environment..."

# Change to FDR directory
cd /home/huytd/multi-agent/multi-agent/FDR

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Activating ma_vqa environment..."
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ma_vqa
    
    # Check if activation was successful
    if [[ "$CONDA_DEFAULT_ENV" == "ma_vqa" ]]; then
        echo "✅ Successfully activated ma_vqa environment"
        echo "🔧 Current environment: $CONDA_DEFAULT_ENV"
        
        # Show Python version and key packages
        echo "🐍 Python version:"
        python --version
        
        echo "📦 Checking key packages..."
        python -c "
import sys
print('Python executable:', sys.executable)

try:
    import nltk
    print('✅ NLTK available')
except ImportError:
    print('❌ NLTK not available')

try:
    import rouge_score
    print('✅ rouge-score available')
except ImportError:
    print('❌ rouge-score not available')
    
try:
    import bert_score
    print('✅ bert-score available')
except ImportError:
    print('❌ bert-score not available')

try:
    import torch
    print('✅ PyTorch available, CUDA available:', torch.cuda.is_available())
except ImportError:
    print('❌ PyTorch not available')
"
        
        echo ""
        echo "🔬 Running multi-reference explanation evaluation..."
        python evaluate_clevr_x_multi_ref.py
        
        echo ""
        echo "✅ Evaluation completed!"
        echo "📊 Check output/clevr_x_multi_ref_evaluation_results.json for results"
        
    else
        echo "❌ Failed to activate ma_vqa environment"
        echo "🔧 Current environment: $CONDA_DEFAULT_ENV"
        echo "💡 Please make sure ma_vqa environment exists and try again"
        exit 1
    fi
    
else
    echo "❌ Conda not found in PATH"
    echo "💡 Please make sure conda is installed and accessible"
    echo "🔧 Trying to run with current Python environment..."
    python evaluate_clevr_x_multi_ref.py
fi
