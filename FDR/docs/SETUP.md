# Setup Guide

## 🎯 Overview

This guide covers setting up the VQA Top-Down pipeline with CPU-friendly configuration and OpenAI backend support.

## 📋 Prerequisites

- **Python**: 3.8+ 
- **Conda**: Package manager
- **Hardware**: CPU (recommended) or GPU (optional)
- **OpenAI API**: Active account with API key

## 🚀 Quick Setup (Recommended)

### 1. Environment Activation
```bash
# Activate existing VQA environment
conda activate VQA

# If environment doesn't exist, create it
conda env create -f VQA_env.yaml
conda activate VQA
```

### 2. OpenAI API Configuration
```bash
# Option A: File-based (Recommended)
cd Top-Down
echo "sk-your-actual-api-key-here" > openai_key.txt

# Option B: Environment variable
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 3. Verify Setup
```bash
# Test all components
python tools/test_config.py

# Expected output:
# 🧪 VQA Pipeline Configuration Test
# ==================================================
# 🔍 Testing OpenAI Configuration...
# ✅ OpenAI API: Connected successfully
# ✅ Model: gpt-4o-mini
# 
# 🔍 Testing GroundingDINO Configuration...
# ✅ GroundingDINO model loaded successfully
# 
# 🎯 Overall Status: ✅ READY
```

### 4. Test Run
```bash
# Quick test (10 questions)
python main.py --config configs/vivqax_config.yaml --backend openai --test
```

## 🔧 Detailed Configuration

### Backend Selection

**Option 1: OpenAI (Recommended)**
- ✅ No local server setup required
- ✅ Reliable and stable
- ✅ CPU-friendly
- ❌ Requires API key and costs tokens

**Option 2: vLLM (Advanced)**
- ✅ Local inference
- ✅ No API costs
- ❌ Requires GPU and server setup
- ❌ More complex configuration

### Component Configuration

Edit `configs/vivqax_config.yaml`:

```yaml
agents_config:
  responder:
    model_name: "gpt-4o-mini"           # Or "Qwen/Qwen2.5-VL-7B-Instruct" for vLLM
    groundingdino_docker: false        # Native mode (CPU/GPU)
    enable_dam: false                   # Disable for simplicity
    temperature: 0.7
    max_tokens: 1000

data_config:
  num_questions: 10                     # Test with 10, use -1 for full dataset
```

## 🏗️ Component Setup

### GroundingDINO (Native Mode)
```bash
# Verify GroundingDINO is working
cd GroundingDINO
python -c "
from groundingdino.util.inference import load_model, load_image, predict, annotate
print('✅ GroundingDINO imported successfully')
"

# Check required files
ls groundingdino/config/GroundingDINO_SwinT_OGC.py
ls weights/groundingdino_swint_ogc.pth
```

### DAM (Optional)
```bash
# Test DAM import (disabled by default)
python -c "
import sys
sys.path.append('DAM')
from dam.describe_anything_model import DescribeAnythingModel
print('✅ DAM imported successfully')
"
```

## 🧪 Testing & Verification

### Configuration Test
```bash
# Comprehensive test
python tools/test_config.py

# Individual component tests
python -c "from utils.backend_manager import BackendManager; print('✅ Backend OK')"
```

### Pipeline Test
```bash
# Minimal test
python main.py --config configs/vivqax_config.yaml --backend openai --test

# Check outputs
ls output/vivqax_*.json
ls output/vivqax_*.txt
```

## 🐛 Troubleshooting

### OpenAI API Issues
```bash
# Test API key manually
python -c "
from openai import OpenAI
client = OpenAI(api_key='sk-your-key-here')
print(client.models.list())
"
```

### GroundingDINO Issues
```bash
# CPU mode warning (normal)
# "Failed to load custom C++ ops. Running on CPU mode Only!"
# This is expected and fine for CPU usage

# Memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Import Path Issues
```bash
# Fix Python path
export PYTHONPATH="/home/huynq/VQA:$PYTHONPATH"
```

## 🔄 Fallback Configurations

### CPU-Only Mode
```yaml
# In vivqax_config.yaml
agents_config:
  responder:
    groundingdino_docker: false  # Native CPU mode
    enable_dam: false            # Disable GPU-heavy components
```

### Minimal Mode
```yaml
# For testing with minimal resources
data_config:
  num_questions: 1               # Single question test

agents_config:
  responder:
    enable_groundingdino: false  # Skip object detection
    enable_dam: false            # Skip detailed analysis
```

## 📊 Performance Tuning

### Memory Optimization
```yaml
agents_config:
  responder:
    max_tokens: 500              # Reduce for memory
    temperature: 0.1             # More deterministic
```

### Speed Optimization
```bash
# Use smaller models if available
# Reduce number of test questions
# Enable only essential components
```

## 🎯 Production Deployment

### Environment Variables
```bash
# Production environment
export OPENAI_API_KEY="your-production-key"
export PYTHONPATH="/path/to/vqa:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"  # If using GPU
```

### Logging Configuration
```python
# In main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/pipeline.log'),
        logging.StreamHandler()
    ]
)
```

## 🔍 Next Steps

1. **Basic Testing**: Run test_config.py
2. **Small Dataset**: Test with 1-10 questions  
3. **Full Pipeline**: Run with complete dataset
4. **Custom Configuration**: Adapt for your use case

## 📚 References

- [Architecture Guide](ARCHITECTURE.md) - System design details
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Contributing](CONTRIBUTING.md) - Development guidelines

---

**Note**: This setup prioritizes simplicity and CPU compatibility. GPU acceleration is available but optional. 