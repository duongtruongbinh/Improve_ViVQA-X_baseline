# SIRI Framework Troubleshooting Guide

This guide covers common issues and their solutions when working with the SIRI VQA framework.

## 🚨 Common Issues & Solutions

### 1. vLLM Server Issues

#### Server Won't Start
**Symptoms:**
- `Connection refused` errors
- `CUDA out of memory` errors
- Server startup hangs

**Solutions:**
```bash
# Check GPU memory usage
nvidia-smi

# Start with reduced memory allocation
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096

# Alternative: Use CPU inference (slower)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --device cpu
```

#### Connection Timeout
**Symptoms:**
- `ReadTimeout` errors in pipeline
- Long response delays

**Solutions:**
```python
# Increase timeout in vllm_client.py
self.client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=300  # Increase to 5 minutes
)
```

#### Model Loading Errors
**Symptoms:**
- `Model not found` errors
- Download failures

**Solutions:**
```bash
# Pre-download model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# Check HuggingFace token
export HF_TOKEN="your_token_here"

# Alternative model
vllm serve microsoft/Phi-3.5-vision-instruct
```

### 2. GroundingDINO Issues

#### Compilation Errors
**Symptoms:**
- `No module named 'groundingdino'`
- CUDA compilation failures
- Version incompatibility errors

**Solutions:**
```bash
# Reinstall with proper CUDA support
cd GroundingDINO
pip uninstall groundingdino
pip install -e .

# Check CUDA version compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install compatible PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Missing Weights
**Symptoms:**
- `File not found: groundingdino_swint_ogc.pth`
- Zero detections

**Solutions:**
```bash
# Download weights manually
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Verify file integrity
ls -la groundingdino_swint_ogc.pth  # Should be ~693MB
```

#### Poor Detection Quality
**Symptoms:**
- No objects detected
- Incorrect bounding boxes
- Low confidence scores

**Solutions:**
```python
# Adjust detection thresholds in agents.py
boxes, logits, phrases = self.grounding_dino_model.predict_with_caption(
    image=image, 
    caption=detection_prompt,
    box_threshold=0.25,  # Lower for more detections
    text_threshold=0.20   # Lower for more phrases
)

# Improve detection prompts
detection_keywords = f"{main_keywords}. objects. things. items."
```

### 3. DAM Model Issues

#### Model Loading Warnings
**Symptoms:**
- Torchvision version warnings
- Model path warnings

**Solutions:**
```bash
# These warnings are typically harmless but can be suppressed
export PYTHONWARNINGS="ignore::UserWarning"

# Or install specific torchvision version
pip install torchvision==0.15.2
```

#### CUDA Memory Issues
**Symptoms:**
- `CUDA out of memory` during DAM inference
- System hangs

**Solutions:**
```python
# Add memory management in agents.py
torch.cuda.empty_cache()

# Reduce image resolution
image = image.resize((512, 512))  # Smaller than default

# Process images in smaller batches
```

#### Poor Analysis Quality
**Symptoms:**
- Generic responses
- Irrelevant descriptions

**Solutions:**
```python
# Improve DAM prompts in agents.py
prompt = f"""Analyze this image in detail, focusing on: {question}
Describe colors, positions, actions, and relationships.
Be specific and detailed."""
```

### 4. Pipeline Integration Issues

#### Agent Communication Errors
**Symptoms:**
- Empty responses between agents
- JSON parsing errors
- Missing explainability traces

**Solutions:**
```python
# Add robust error handling in pipeline.py
try:
    mvkb = seeker_agent.process(initial_response, answer_candidates)
except Exception as e:
    logger.error(f"SeekerAgent failed: {e}")
    # Use simplified MVKB
    mvkb = self._create_fallback_mvkb(initial_response, answer_candidates)
```

#### Configuration Issues
**Symptoms:**
- Config file not found
- Wrong model URLs
- Missing API keys

**Solutions:**
```bash
# Verify config file exists and is valid
python -c "import yaml; yaml.safe_load(open('Top-Down/config.yaml'))"

# Check configuration paths
ls Top-Down/configs/vivqa_config.yaml
ls Top-Down/config.yaml

# Validate vLLM endpoint
curl -X GET "http://localhost:9100/health"
```

### 5. Dataset & I/O Issues

#### Dataset Loading Errors
**Symptoms:**
- `FileNotFoundError` for images
- JSON parsing errors
- Path resolution issues

**Solutions:**
```python
# Check dataset paths in configs/vivqa_config.yaml
dataset_paths:
  questions_file: "/absolute/path/to/questions.json"  # Use absolute paths
  images_dir: "/absolute/path/to/images/"

# Verify files exist
import os
print(os.path.exists("/path/to/your/questions.json"))
print(os.path.exists("/path/to/your/images/"))
```

#### Output Directory Issues
**Symptoms:**
- Permission denied errors
- Missing output files
- Corrupted results

**Solutions:**
```bash
# Create output directory with proper permissions
mkdir -p Top-Down/output
chmod 755 Top-Down/output

# Check disk space
df -h

# Clear old results if needed
rm -f Top-Down/output/siri_pipeline_results.json
```

### 6. Memory & Performance Issues

#### GPU Memory Exhaustion
**Symptoms:**
- `CUDA out of memory` errors
- System crashes
- Slow inference

**Solutions:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Reduce batch sizes
num_questions: 1  # Process one at a time

# Use gradient checkpointing
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
```

#### Slow Performance
**Symptoms:**
- Long processing times
- High CPU usage
- Memory leaks

**Solutions:**
```python
# Add memory cleanup in pipeline.py
import gc
torch.cuda.empty_cache()
gc.collect()

# Use multiprocessing carefully
# Avoid too many concurrent processes
```

## 🔧 Debug Mode & Logging

### Enable Debug Logging
```python
# Add to main.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

### Diagnostic Commands
```bash
# System information
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi
free -h
df -h

# Model verification
python -c "from groundingdino.util.inference import load_model; print('GroundingDINO OK')"
python -c "import sys; sys.path.append('DAM'); from dam.describe_anything_model import DescribeAnythingModel; print('DAM OK')"

# Network connectivity
curl -X POST "http://localhost:9100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": [{"role": "user", "content": "test"}]}'
```

## 🚑 Emergency Procedures

### Complete Reset
```bash
# Stop all services
pkill -f "vllm serve"

# Clear GPU memory
sudo nvidia-smi --gpu-reset

# Reinstall environment
conda deactivate
conda env remove -n VQA_env
conda env create -f Top-Down/VQA_env.yaml
conda activate VQA_env

# Clear caches
rm -rf ~/.cache/huggingface/
rm -rf Top-Down/__pycache__/
rm -rf Top-Down/core/__pycache__/
```

### Fallback to Basic Mode
```python
# Minimal configuration for testing
# Edit main.py to skip enhanced features
if __name__ == "__main__":
    # Disable enhanced pipeline
    config = {
        "use_grounding_dino": False,
        "use_dam_analysis": False,
        "backend": "openai"  # If vLLM fails
    }
```

### Data Recovery
```bash
# Backup current results
cp Top-Down/output/siri_pipeline_results.json Top-Down/output/backup_$(date +%Y%m%d_%H%M%S).json

# Recover from partial results
python -c "
import json
with open('Top-Down/output/siri_pipeline_results.json') as f:
    data = json.load(f)
print(f'Processed: {len(data)} questions')
print(f'Last question ID: {data[-1].get(\"question_id\", \"unknown\")}')
"
```

## 📞 Getting Help

### Information to Collect
When reporting issues, please include:
- Operating system and version
- Python and CUDA versions
- GPU model and memory
- Complete error traceback
- Configuration files
- Steps to reproduce

### Log Files to Check
- vLLM server logs: Usually in terminal output
- Python application logs: Check console output
- System logs: `dmesg` for GPU/memory issues
- CUDA logs: Set `CUDA_LAUNCH_BLOCKING=1`

### Performance Benchmarks
```bash
# Quick performance test
time python Top-Down/main.py --config Top-Down/configs/vllm_test_config.yaml

# Monitor resources during run
htop &
nvidia-smi dmon &
python Top-Down/main.py --config Top-Down/configs/vllm_test_config.yaml
```

This troubleshooting guide should help resolve most common issues. For persistent problems, consider checking the GitHub issues or creating a new issue with detailed information. 