# vLLM Server Setup for FDR Multi-Agent Pipeline

## 📋 Overview

This guide helps you host the Qwen2.5-VL-7B-Instruct model using vLLM for the FDR (Faithful Decomposed Reasoning) multi-agent pipeline.

## 🚀 Quick Start

### 1. Start vLLM Server

```bash
# Navigate to FDR directory
cd /home/huytd/multi-agent/multi-agent/FDR

# Start the vLLM server (this will activate vllm_env automatically)
./launch_vllm.sh
```

### 2. Test Server Connection

In a new terminal:

```bash
# Test if server is working
python test_vllm.py
```

### 3. Run the FDR Pipeline

In another terminal:

```bash
# Run the pipeline with vLLM backend
python main.py --backend vllm
```

## 🔧 Configuration Details

### Server Configuration

The vLLM server is configured in `config.yaml`:

```yaml
backend_config:
  type: "vllm"
  vllm_settings:
    model_path: "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
    host: "0.0.0.0"
    port: 9100
    api_key: "dummy-key"
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    tensor_parallel_size: 1
```

### Model Information

- **Model**: Qwen2.5-VL-7B-Instruct
- **Size**: ~7B parameters  
- **Type**: Vision-Language Model
- **Use Case**: Visual Question Answering with Vietnamese support

## 📊 Server Endpoints

Once running, the server provides OpenAI-compatible endpoints:

- **Base URL**: `http://localhost:9100/v1`
- **Health Check**: `http://localhost:9100/health`
- **Models**: `http://localhost:9100/v1/models`
- **Chat Completions**: `http://localhost:9100/v1/chat/completions`

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill existing process on port 9100
   sudo lsof -ti:9100 | xargs sudo kill -9
   ```

2. **GPU Memory Error**
   - Reduce `gpu_memory_utilization` in config.yaml
   - Check available GPU memory: `nvidia-smi`

3. **Model not found**
   - Verify model path exists: `ls /mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct`
   - Check permissions: `ls -la /mnt/dataset1/pretrained_fm/`

4. **Environment not activated**
   ```bash
   # Manually activate environment
   source ~/miniconda3/bin/activate
   conda activate vllm_env
   ```

### Performance Tuning

1. **For better performance**:
   - Increase `gpu_memory_utilization` to 0.9
   - Use `tensor_parallel_size` > 1 if you have multiple GPUs

2. **For memory constraints**:
   - Reduce `max_model_len` to 2048
   - Decrease `gpu_memory_utilization` to 0.7

## 📝 Usage Examples

### Direct API Call

```python
import requests

response = requests.post(
    "http://localhost:9100/v1/chat/completions",
    headers={"Authorization": "Bearer dummy-key"},
    json={
        "model": "Qwen2.5-VL-7B-Instruct",
        "messages": [{"role": "user", "content": "What is in this image?"}],
        "max_tokens": 100
    }
)
```

### Using OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:9100/v1"
)

response = client.chat.completions.create(
    model="Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🔍 Monitoring

### Check Server Status

```bash
# Check if server is running
curl http://localhost:9100/health

# List available models
curl -H "Authorization: Bearer dummy-key" http://localhost:9100/v1/models
```

### View Logs

Server logs are displayed in the terminal where you started `./launch_vllm.sh`

## 🚪 Stopping the Server

Press `Ctrl+C` in the terminal where the server is running, or:

```bash
# Find and kill the process
sudo lsof -ti:9100 | xargs sudo kill -9
```

## 📚 Next Steps

1. **Run FDR Pipeline**: Use `python main.py --backend vllm` to start the multi-agent VQA pipeline
2. **Evaluate Results**: Add `--evaluate` flag for comprehensive evaluation  
3. **Custom Datasets**: Modify `config.yaml` to use your own VQA datasets

For more detailed documentation, see the main [FDR README](README.md).
