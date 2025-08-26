#!/bin/bash
# Script to install missing libraries and run CLEVR-X evaluation

echo "🚀 CLEVR-X Multi-Reference Explanation Evaluation Setup"
echo "📦 Installing required packages..."

# Change to FDR directory
cd /home/huytd/multi-agent/multi-agent/FDR

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "✅ Activating ma_vqa environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ma_vqa
    
    if [[ "$CONDA_DEFAULT_ENV" == "ma_vqa" ]]; then
        echo "✅ Successfully activated ma_vqa environment"
        
        # Install required packages
        echo "📦 Installing/updating required packages..."
        
        # Install NLTK and download required data
        echo "📥 Setting up NLTK..."
        pip install nltk
        python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
resources = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords']
for resource in resources:
    try:
        nltk.download(resource, quiet=False)
        print(f'✅ Downloaded: {resource}')
    except Exception as e:
        print(f'❌ Failed to download {resource}: {e}')
"
        
        # Install ROUGE-score
        echo "📥 Installing rouge-score..."
        pip install rouge-score
        
        # Install BERTScore
        echo "📥 Installing bert-score..."
        pip install bert-score
        
        # Install pycocoevalcap for CIDEr and SPICE
        echo "📥 Installing pycocoevalcap..."
        pip install pycocoevalcap
        
        # Install additional dependencies
        echo "📥 Installing additional dependencies..."
        pip install transformers torch torchvision
        
        echo ""
        echo "✅ All packages installed!"
        echo ""
        
        # Verify installations
        echo "🔍 Verifying package installations..."
        python -c "
try:
    import nltk
    print('✅ NLTK available')
    
    # Test NLTK tokenization
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize('This is a test.')
    print(f'✅ NLTK tokenization works: {tokens}')
except Exception as e:
    print(f'❌ NLTK issue: {e}')

try:
    from rouge_score import rouge_scorer
    print('✅ rouge-score available')
except Exception as e:
    print(f'❌ rouge-score issue: {e}')

try:
    from bert_score import score as bert_scorer
    print('✅ bert-score available')
except Exception as e:
    print(f'❌ bert-score issue: {e}')

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    print('✅ pycocoevalcap (CIDEr, SPICE) available')
except Exception as e:
    print(f'❌ pycocoevalcap issue: {e}')

try:
    import torch
    print(f'✅ PyTorch available, CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ PyTorch issue: {e}')
"
        
        echo ""
        echo "🔬 Running multi-reference explanation evaluation..."
        python evaluate_clevr_x_multi_ref.py
        
        echo ""
        echo "✅ Evaluation completed!"
        echo "📊 Check output/clevr_x_multi_ref_evaluation_results.json for results"
        
    else
        echo "❌ Failed to activate ma_vqa environment"
        exit 1
    fi
else
    echo "❌ Conda not found"
    exit 1
fi
