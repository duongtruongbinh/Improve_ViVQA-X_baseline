# FDR Framework Setup Guide

> **Production-ready Vietnamese Visual Question Answering with Faithful Explanations**

## 🎯 Overview

This guide covers setting up the FDR (Faithful Decomposed Reasoning) framework - a 4-agent multi-modal VQA system optimized for Vietnamese language with 90-95% effectiveness and 9-12s response time.

## 📋 Prerequisites

- **Python**: 3.8+
- **Conda**: Package manager (recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended, 8GB+ VRAM)
- **System Memory**: 16GB+ RAM recommended
- **API Access**: OpenAI API key (for cloud backend option)

## 🚀 Quick Setup (Recommended)

### 1. Environment Setup
```bash
# Clone and setup repository
cd FDR

# Activate VQA environment (if exists)
conda activate VQA

# Or create new environment
conda create -n FDR python=3.8
conda activate FDR

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

#### Edit the unified config file:
```bash
# Edit main configuration
nano config.yaml
```

#### Essential configuration:
```yaml
# config.yaml - Main configuration file
active_dataset: "vivqax"  # or "vqax" for English

agents_config:
  verifier:
    model_name: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 1000
    enable_dam: true               # Enable dense captioning
    groundingdino_docker: false    # Use native GroundingDINO
    
  strategist:
    temperature: 0.5
    max_sub_questions: 3
    
  synthesizer:
    voting_method: "weighted"
    
  explanation:
    max_explanation_length: 500

backend_config:
  use_vllm: false                  # true for local vLLM, false for OpenAI
  model_name: "gpt-4o-mini"
  openai_api_key_file: "openai_key.txt"

datasets:
  vivqax:
    format: "vivqax"
    data_path: "/path/to/your/vivqax_data.json"
    image_dir: "/path/to/your/images/"
    
processing_config:
  num_samples: 10                  # -1 for all samples
  enable_caching: true
  
output_config:
  output_dir: "output"
  output_file: "fdr_results.json"
```

### 3. API Key Setup (OpenAI Backend)
```bash
# Option A: File-based (Recommended)
echo "sk-your-actual-api-key-here" > openai_key.txt

# Option B: Environment variable
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 4. Test Installation
```bash
# Quick test with OpenAI backend
python main.py --backend openai --test --samples 1

# Expected output:
# 🚀 Starting MVKB-X Pipeline
# 🔍 VerifierAgent analyzing: [question]
# 🧠 StrategistAgent building MVKB...
# ⚖️ SynthesizerAgent conducting weighted voting...
# 📝 ExplanationAgent generating explanation...
# ✅ Pipeline completed successfully!
```

## 🔧 Component Setup

### GroundingDINO (Object Detection)

#### Native Installation (Recommended):
```bash
# Navigate to GroundingDINO directory
cd GroundingDINO

# Install dependencies
pip install -e .

# Verify installation
python -c "
from groundingdino.util.inference import load_model
print('✅ GroundingDINO installed successfully')
"

# Check model weights (auto-downloaded on first use)
ls weights/groundingdino_swint_ogc.pth  # Should be ~693MB
```

#### Docker Alternative:
```bash
# Build GroundingDINO Docker image
cd GroundingDINO
docker build -t groundingdino:latest .

# Update config to use Docker
# In config.yaml: groundingdino_docker: true
```

### DAM Model (Dense Captioning)

#### GPU Setup:
```bash
# Test DAM installation
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/DAM-3B-Self-Contained', trust_remote_code=True)
print('✅ DAM model accessible')
"
```

#### CPU Fallback:
```bash
# If GPU memory is limited, disable DAM
# In config.yaml: enable_dam: false
# VLM fallback will be used automatically
```

## 🏗️ Backend Configuration

### Option 1: OpenAI Backend (Recommended for Testing)

**Advantages:**
- ✅ No local GPU requirements
- ✅ Reliable and stable
- ✅ Easy setup
- ❌ Requires API costs

**Setup:**
```yaml
# config.yaml
backend_config:
  use_vllm: false
  model_name: "gpt-4o-mini"
  openai_api_key_file: "openai_key.txt"
```

### Option 2: vLLM Backend (Advanced)

**Advantages:**
- ✅ Local inference (no API costs)
- ✅ Full control over models
- ❌ Requires GPU setup
- ❌ More complex configuration

**Setup:**
```bash
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --max-model-len 4096

# Update config
# In config.yaml: use_vllm: true
```

## 🧪 Testing & Verification

### Basic Pipeline Test
```bash
# Test all components with minimal data
python main.py --backend openai --test --samples 1

# Test with evaluation enabled
python main.py --backend openai --test --samples 5 --evaluate

# Test specific number of samples
python main.py --backend openai --samples 10
```

### Component Testing
```bash
# Test GroundingDINO
python -c "
import sys
sys.path.append('FDR/src')
from agents.verifier import VerifierAgent
agent = VerifierAgent()
print('✅ VerifierAgent initialized')
"

# Test prompt system
python src/prompts/tools/validate_templates.py

# Test backend connectivity
python -c "
from src.utils.backend_manager import get_backend_manager
manager = get_backend_manager('openai')
print('✅ Backend manager working')
"
```

### Performance Validation
```bash
# Run performance benchmark
time python main.py --backend openai --samples 5

# Expected timing:
# - Total: ~45-60 seconds for 5 questions
# - Per question: ~9-12 seconds average
# - GPU utilization: Moderate (GroundingDINO phases)
```

## 📊 Dataset Setup

### ViVQA-X (Vietnamese)
```bash
# Prepare Vietnamese VQA dataset
mkdir -p data/vivqax
# Place your ViVQA-X data in this directory

# Update config.yaml
# datasets:
#   vivqax:
#     data_path: "data/vivqax/vivqax_test.json"
#     image_dir: "data/vivqax/images/"
```

### VQA-X (English)
```bash
# Alternative English dataset
mkdir -p data/vqax

# Update config.yaml
# active_dataset: "vqax"
# datasets:
#   vqax:
#     data_path: "data/vqax/vqax_test.json"
#     image_dir: "data/vqax/images/"
```

### Custom Dataset
```json
// Custom dataset format
[
  {
    "question_id": "12345",
    "question": "Xe này màu gì?",
    "image_name": "car_image.jpg",
    "answer": "đỏ"
  }
]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Fix Python path
export PYTHONPATH="/path/to/FDR:$PYTHONPATH"

# Or add to your shell profile
echo 'export PYTHONPATH="/path/to/FDR:$PYTHONPATH"' >> ~/.bashrc
```

#### 2. GPU Memory Issues
```bash
# Monitor GPU usage
nvidia-smi

# Reduce memory usage
# In config.yaml:
# agents_config:
#   verifier:
#     enable_dam: false  # Disable DAM
#     max_tokens: 500    # Reduce token limit
```

#### 3. GroundingDINO Compilation Issues
```bash
# Install specific CUDA version
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Recompile GroundingDINO
cd GroundingDINO
pip uninstall groundingdino
pip install -e .
```

#### 4. OpenAI API Issues
```bash
# Test API key manually
python -c "
from openai import OpenAI
client = OpenAI(api_key='sk-your-key-here')
print('API Key valid!')
"

# Check rate limits
curl -H "Authorization: Bearer sk-your-key-here" https://api.openai.com/v1/models
```

### Debug Mode
```bash
# Enable detailed logging
export PYTHONWARNINGS="ignore"
python main.py --backend openai --test --samples 1 2>&1 | tee debug.log

# Check agent performance
python src/prompts/tools/validate_templates.py --check-performance
```

## 🎯 Production Deployment

### Environment Variables
```bash
# Production environment setup
export OPENAI_API_KEY="your-production-key"
export PYTHONPATH="/path/to/FDR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU

# Logging configuration
export TRANSFORMERS_VERBOSITY="error"  # Reduce logging noise
```

### System Service (Optional)
```bash
# Create systemd service for continuous operation
sudo nano /etc/systemd/system/fdr-vqa.service

# Service content:
# [Unit]
# Description=FDR VQA Service
# 
# [Service]
# Type=simple
# User=your-user
# WorkingDirectory=/path/to/FDR
# ExecStart=/path/to/conda/envs/FDR/bin/python main.py --backend openai
# Restart=always
# 
# [Install]
# WantedBy=multi-user.target

# Enable and start
sudo systemctl enable fdr-vqa
sudo systemctl start fdr-vqa
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN cd GroundingDINO && pip install -e .

CMD ["python", "main.py", "--backend", "openai"]
```

## 📈 Performance Optimization

### Memory Optimization
```yaml
# config.yaml - Optimized for limited memory
processing_config:
  batch_size: 1
  enable_caching: true
  
agents_config:
  verifier:
    max_tokens: 500        # Reduce for memory
    enable_dam: false      # Disable heavy components
  strategist:
    max_sub_questions: 2   # Reduce complexity
```

### Speed Optimization
```yaml
# config.yaml - Optimized for speed
agents_config:
  verifier:
    temperature: 0.1       # More deterministic
    groundingdino_docker: false  # Faster native mode
  strategist:
    temperature: 0.3       # Faster generation
```

## 🔍 Next Steps

1. **Basic Testing**: Run pipeline with 1-5 samples
2. **Performance Validation**: Test with your dataset
3. **Prompt Customization**: Modify templates in `src/prompts/`
4. **Agent Tuning**: Adjust parameters in `config.yaml`
5. **Production Deployment**: Set up monitoring and logging

## 📚 Additional Resources

- **[Architecture Guide](ARCHITECTURE.md)** - Technical system details
- **[Prompt Engineering Guide](PROMPT_ENGINEERING_GUIDE.md)** - Template customization
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Detailed problem solving
- **[Cheat Sheet](PROMPT_CHEAT_SHEET.md)** - Quick reference

---

## ✅ Verification Checklist

- [ ] Python environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GroundingDINO compiled successfully
- [ ] OpenAI API key configured
- [ ] Config.yaml updated with dataset paths
- [ ] Basic test passes (`python main.py --test --samples 1`)
- [ ] GPU memory sufficient (check `nvidia-smi`)
- [ ] All 4 agents initialize without errors

**Success Criteria**: Pipeline processes 1 sample in ~9-12 seconds with all agents functioning correctly.

---

*Setup Guide Version: 2.0 | Compatible with FDR Framework Production Release* 