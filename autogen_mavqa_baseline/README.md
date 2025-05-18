# Usage Instructions

## Environment Setup

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
```

## Model Name

### Visual Language Models (VLMs)

```bash
Qwen/Qwen2-VL-2B-Instruct
Qwen/Qwen2-VL-7B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct
```

### Large Language Models (LLMs)

```bash
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-8B
```

```bash
Qwen/Qwen3-4B
```

## Serve Visual Language Models

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-VL-2B-Instruct --dtype bfloat16 --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-VL-7B-Instruct --dtype bfloat16 --port 8001
```

## GPU Monitoring & Process Management

- Monitor GPU usage:
  - `nvtop`
  - `gpustat`
- Kill a process (replace `<PID>` with the actual process ID):
  - `kill -9 <PID>`

## Running Main Script

```bash
python3 main.py
# or with test data options
python main.py --use_num_test_data --num_test_data 20
```

## Python Environment Checks

```bash
pip check
pipdeptree
pipdeptree -w silence
```
