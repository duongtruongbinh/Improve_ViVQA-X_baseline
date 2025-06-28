# FDR Framework Troubleshooting Guide

> **Solutions for common issues in the FDR Multi-Agent VQA System**

This guide covers troubleshooting for the production-ready FDR framework with 4-agent architecture (VerifierAgent, StrategistAgent, SynthesizerAgent, ExplanationAgent).

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Quick system status
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# GPU status
nvidia-smi

# Test basic pipeline
python main.py --backend openai --test --samples 1
```

### Common Error Patterns
- **Import errors**: Python path issues
- **GPU memory errors**: Insufficient VRAM or memory leaks
- **GroundingDINO compilation**: CUDA version mismatches
- **API errors**: OpenAI key or rate limit issues
- **Agent failures**: Config or dependency problems

## 🔧 Agent-Specific Issues

### 1. VerifierAgent Issues

#### GroundingDINO Compilation Errors
**Symptoms:**
- `name '_C' is not defined`
- CUDA compilation failures
- `RuntimeError: No such operator`

**Solutions:**
```bash
# Check CUDA compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall with proper CUDA support
cd GroundingDINO
pip uninstall groundingdino
pip install -e .

# For compilation issues, patch deprecated API
# Replace `.type().is_cuda()` with `.is_cuda()` in:
# - groundingdino/models/GroundingDINO/ms_deform_attn.py
# - GroundingDINO_ops/src/vision.cpp

# Install compatible PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Missing GroundingDINO Weights
**Symptoms:**
- `FileNotFoundError: groundingdino_swint_ogc.pth`
- Zero object detections

**Solutions:**
```bash
# Download weights manually
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Verify download (should be ~693MB)
ls -la groundingdino_swint_ogc.pth

# Test detection
python -c "
from groundingdino.util.inference import load_model
model = load_model('groundingdino/config/GroundingDINO_SwinT_OGC.py', 'weights/groundingdino_swint_ogc.pth')
print('✅ GroundingDINO model loaded successfully')
"
```

#### DAM Model Integration Issues
**Symptoms:**
- `RuntimeError: mat1 and mat2 must have the same dtype`
- DAM loading failures
- Memory errors during dense captioning

**Solutions:**
```bash
# Disable DAM if problematic (common solution)
# In config.yaml:
# agents_config:
#   verifier:
#     enable_dam: false

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# Test DAM separately
python -c "
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained('nvidia/DAM-3B-Self-Contained', trust_remote_code=True)
    print('✅ DAM model accessible')
except Exception as e:
    print(f'❌ DAM error: {e}')
"
```

### 2. StrategistAgent Issues

#### Sub-question Generation Failures
**Symptoms:**
- Empty MVKB entries
- Generic or irrelevant sub-questions
- Timeout errors

**Solutions:**
```yaml
# Adjust config.yaml
agents_config:
  strategist:
    temperature: 0.5              # Lower for more focused questions
    max_sub_questions: 2          # Reduce if memory constrained
    confidence_threshold: 0.6     # Adjust quality threshold
```

#### MVKB Construction Errors
**Symptoms:**
- Malformed hypothesis structures
- Missing confidence scores
- JSON parsing errors

**Solutions:**
```bash
# Test strategist templates
python src/prompts/tools/validate_templates.py --agent strategist

# Check prompt rendering
python -c "
from prompts import PromptManager
pm = PromptManager()
result = pm.render('agents/strategist/fdr_strategist_hypothesis.jinja', 
                   question='Test question',
                   answer_candidate='test answer')
print('Template renders successfully')
"
```

### 3. SynthesizerAgent Issues

#### Weighted Voting Failures
**Symptoms:**
- All candidates get equal scores
- Voting algorithm errors
- Inconsistent final answers

**Solutions:**
```yaml
# Optimize synthesizer config
agents_config:
  synthesizer:
    voting_method: "weighted"     # Ensure weighted voting
    normalization: true          # Normalize scores
    min_confidence: 0.1          # Minimum confidence threshold
```

#### Score Calculation Errors
**Symptoms:**
- NaN or infinite scores
- Negative confidence values
- Division by zero errors

**Solutions:**
```python
# Debug voting process
# Add to synthesizer debugging:
import logging
logging.basicConfig(level=logging.DEBUG)

# Check MVKB structure before voting
for entry in mvkb:
    assert 'confidence_score' in entry
    assert 0 <= entry['confidence_score'] <= 1
```

### 4. ExplanationAgent Issues

#### Poor Explanation Quality
**Symptoms:**
- Generic or template-like explanations
- Missing reasoning steps
- Inconsistent with actual decision process

**Solutions:**
```yaml
# Improve explanation config
agents_config:
  explanation:
    max_explanation_length: 500    # Increase for more detail
    include_confidence: true       # Include confidence rationale
    temperature: 0.3               # More creative explanations
```

## 🛠️ System-Level Issues

### GPU Memory Management

#### CUDA Out of Memory
**Symptoms:**
- `RuntimeError: CUDA out of memory`
- System crashes during inference
- Slow performance

**Solutions:**
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Optimize memory usage in config.yaml
agents_config:
  verifier:
    enable_dam: false              # Disable heavy components
    max_tokens: 500               # Reduce token limits
  strategist:
    max_sub_questions: 2          # Reduce parallel processing
    
processing_config:
  batch_size: 1                   # Process one at a time
  enable_caching: false           # Disable if memory tight
```

#### Memory Leaks
**Symptoms:**
- Gradually increasing memory usage
- Performance degradation over time
- System becomes unresponsive

**Solutions:**
```python
# Add memory cleanup to pipeline
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Call after each sample processing
cleanup_memory()
```

### Backend Issues

#### OpenAI API Problems
**Symptoms:**
- `Connection error` or `API key invalid`
- Rate limit exceeded
- Timeout errors

**Solutions:**
```bash
# Test API connectivity
curl -H "Authorization: Bearer $(cat openai_key.txt)" \
     "https://api.openai.com/v1/models" | head -20

# Check rate limits
python -c "
from openai import OpenAI
client = OpenAI(api_key=open('openai_key.txt').read().strip())
try:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=10
    )
    print('✅ OpenAI API working')
except Exception as e:
    print(f'❌ API Error: {e}')
"

# Add retry logic in config
backend_config:
  max_retries: 3
  retry_delay: 2
  timeout: 60
```

#### vLLM Server Issues
**Symptoms:**
- Connection refused errors
- Server startup failures
- Model loading errors

**Solutions:**
```bash
# Check vLLM server status
curl -X GET "http://localhost:9100/health"

# Restart with reduced memory
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048

# Switch to OpenAI fallback
python main.py --backend openai  # Immediate fallback
```

### Configuration Issues

#### Config File Problems
**Symptoms:**
- `FileNotFoundError: config.yaml`
- YAML parsing errors
- Missing required fields

**Solutions:**
```bash
# Validate config file
python -c "
import yaml
try:
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    print('✅ Config file valid')
except Exception as e:
    print(f'❌ Config error: {e}')
"

# Check required fields
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
required = ['active_dataset', 'agents_config', 'backend_config']
missing = [r for r in required if r not in config]
if missing:
    print(f'❌ Missing required fields: {missing}')
else:
    print('✅ All required fields present')
"
```

#### Path Resolution Issues
**Symptoms:**
- `FileNotFoundError` for datasets
- Image loading failures
- Template not found errors

**Solutions:**
```bash
# Use absolute paths in config.yaml
datasets:
  vivqax:
    data_path: "/absolute/path/to/data.json"
    image_dir: "/absolute/path/to/images/"

# Verify paths exist
python -c "
import os
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
dataset = config['datasets'][config['active_dataset']]
print(f'Data file exists: {os.path.exists(dataset[\"data_path\"])}')
print(f'Image dir exists: {os.path.exists(dataset[\"image_dir\"])}')
"
```

## 🔍 Debug Mode & Logging

### Enable Comprehensive Logging
```python
# Add to main.py or debugging script
import logging
import os

# Set debug environment
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA for better error traces
os.environ['PYTHONPATH'] = '/path/to/FDR:' + os.environ.get('PYTHONPATH', '')

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Reduce noise from HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
```

### Performance Profiling
```python
# Add timing to pipeline
import time

def profile_agent_performance():
    times = {}
    
    # Time each agent
    start = time.time()
    verifier_result = verifier.generate_initial_response(question, image_path)
    times['verifier'] = time.time() - start
    
    start = time.time()
    mvkb = strategist.build_mvkb(question, image_path, candidates, caption)
    times['strategist'] = time.time() - start
    
    start = time.time()
    voting_result = synthesizer.conduct_weighted_voting(question, image_path, candidates, mvkb)
    times['synthesizer'] = time.time() - start
    
    start = time.time()
    explanation = explanation_agent.generate_explanation(...)
    times['explanation'] = time.time() - start
    
    print(f"Performance breakdown: {times}")
    return times
```

## 🚑 Emergency Recovery

### Complete System Reset
```bash
# Stop all processes
pkill -f "python main.py"
pkill -f "vllm serve"

# Clear GPU memory
sudo nvidia-smi --gpu-reset

# Clean Python cache
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Recreate environment
conda deactivate
conda create -n FDR_clean python=3.8 -y
conda activate FDR_clean
pip install -r requirements.txt
cd GroundingDINO && pip install -e .
```

### Minimal Working Configuration
```yaml
# emergency_config.yaml - Minimal working setup
active_dataset: "vivqax"

agents_config:
  verifier:
    model_name: "gpt-4o-mini"
    enable_dam: false              # Disable for simplicity
    groundingdino_docker: false
    temperature: 0.7
    max_tokens: 300
    
  strategist:
    temperature: 0.5
    max_sub_questions: 1           # Minimal complexity
    
  synthesizer:
    voting_method: "simple"        # Fallback voting
    
  explanation:
    max_explanation_length: 200    # Short explanations

backend_config:
  use_vllm: false                  # Force OpenAI
  model_name: "gpt-4o-mini"
  openai_api_key_file: "openai_key.txt"

processing_config:
  num_samples: 1                   # Single sample test
  batch_size: 1
  enable_caching: false

output_config:
  output_dir: "output"
  output_file: "emergency_test.json"
```

### Data Recovery
```bash
# Backup existing results
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
cp -r output/* backup/$(date +%Y%m%d_%H%M%S)/

# Check partial results
python -c "
import json
import os
if os.path.exists('output/fdr_results.json'):
    with open('output/fdr_results.json') as f:
        data = json.load(f)
    print(f'Found {len(data)} processed samples')
    if data:
        print(f'Last question ID: {data[-1].get(\"question_id\", \"unknown\")}')
else:
    print('No results file found')
"
```

## 📞 Getting Help

### Information to Collect
When reporting issues, include:

1. **System Information:**
   ```bash
   python --version
   nvidia-smi
   pip list | grep -E "(torch|transformers|opencv|groundingdino)"
   ```

2. **Error Context:**
   - Complete error traceback
   - Agent that failed (Verifier/Strategist/Synthesizer/Explanation)
   - Config file content (sanitized)
   - Sample question/image that caused error

3. **Environment Details:**
   - Operating system and version
   - CUDA version (`nvcc --version`)
   - Available GPU memory
   - Python environment details

### Log Files to Check
- **Pipeline logs**: `debug.log` or console output
- **GPU monitoring**: `nvidia-smi dmon` output during run
- **System logs**: `dmesg` for GPU/memory issues
- **Agent-specific**: Check individual agent error messages

### Quick Fixes Checklist
- [ ] Python path includes FDR directory
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] GroundingDINO compiled without errors
- [ ] OpenAI API key valid and has credits
- [ ] Config.yaml syntax is valid YAML
- [ ] Dataset paths point to existing files
- [ ] GPU has sufficient memory (check `nvidia-smi`)
- [ ] No other processes consuming GPU memory

---

## 📈 Performance Optimization

### Expected Performance Baselines
- **Single question processing**: 9-12 seconds end-to-end
- **GroundingDINO inference**: ~2 seconds per image
- **VLM API calls**: ~1-3 seconds per call
- **GPU memory usage**: 4-8GB peak (with DAM enabled)
- **System memory**: 8-16GB during processing

### Monitoring Commands
```bash
# Continuous monitoring during run
nvidia-smi dmon -s u -d 1 &  # GPU utilization
htop &                       # CPU and memory
python main.py --backend openai --test --samples 5
```

If performance is below these baselines, check for:
- GPU memory fragmentation
- API rate limiting
- Network connectivity issues
- Concurrent processes competing for resources

---

*Troubleshooting Guide Version: 2.0 | Updated for FDR Production Release* 