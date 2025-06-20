# Quick Setup Guide

This guide helps you get the SIRI VQA framework running quickly.

## 🚀 Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090 or better)
- **CUDA**: Version 11.8 or 12.1
- **Python**: 3.8 - 3.11
- **Conda**: Miniconda or Anaconda

## 📦 Quick Installation

### 1. Environment Setup
```bash
# Navigate to workspace
cd /path/to/VQA

# Create environment
conda env create -f Top-Down/VQA_env.yaml
conda activate VQA_env
```

### 2. Model Setup
```bash
# Verify GroundingDINO weights (should be ~693MB)
ls -lh GroundingDINO/weights/groundingdino_swint_ogc.pth

# If missing, download:
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..

# Install GroundingDINO
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Start vLLM Server
```bash
# Start in background (adjust memory if needed)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --api-key dummy-key \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    > vllm.log 2>&1 &

# Wait for server to start (check log)
tail -f vllm.log
```

### 4. Test Connection
```bash
# Test vLLM server
curl -X POST "http://localhost:9100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 5. Configure Dataset
```bash
# Edit dataset configuration
nano Top-Down/configs/vivqa_config.yaml

# Update paths to your dataset:
dataset_paths:
  questions_file: "/absolute/path/to/your/questions.json"
  images_dir: "/absolute/path/to/your/images/"
```

### 6. Run Quick Test
```bash
# Test with 5 questions
python Top-Down/main.py --config Top-Down/configs/vllm_test_config.yaml

# Check results
ls -la Top-Down/output/
cat Top-Down/output/siri_summary.txt
```

## 🛠️ Common Issues

### GPU Memory Error
```bash
# Reduce model memory usage
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096
```

### GroundingDINO Compilation Error
```bash
# Reinstall with proper CUDA
cd GroundingDINO
pip uninstall groundingdino
pip install -e .
```

### vLLM Server Not Starting
```bash
# Check GPU status
nvidia-smi

# Check available models
vllm models | grep Qwen
```

## 📈 Performance Tips

- **GPU Memory**: Monitor with `nvidia-smi`
- **Processing Speed**: Start with `num_questions: 1` for testing
- **Quality vs Speed**: Adjust detection thresholds in agents.py

## 🆘 Need Help?

- Check `Top-Down/docs/TROUBLESHOOTING.md` for detailed solutions
- Review logs in `vllm.log` for server issues
- Use `--backend openai` as fallback (requires OpenAI API key)

## ✅ Verification Checklist

- [ ] Conda environment activated
- [ ] GroundingDINO weights downloaded (693MB)
- [ ] vLLM server running on port 9100
- [ ] Dataset paths configured correctly
- [ ] Test run completes successfully
- [ ] Results saved to `Top-Down/output/`

## 🚀 Next Steps

Once basic setup works:
1. Configure your own dataset
2. Adjust parameters for your use case
3. Scale up to full dataset processing
4. Explore advanced features in documentation

Happy questioning! 🎯 