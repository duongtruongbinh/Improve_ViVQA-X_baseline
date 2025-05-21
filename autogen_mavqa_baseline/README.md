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
## Other Common Parameters for `vllm serve`

Below are some other important parameters you can use with `vllm serve`, based on the documentation:

- `--model <model_name_or_path>`  
    **Description:** Name or path to the Hugging Face model to serve. This is a required parameter.  
    **Example:** `--model meta-llama/Llama-2-7b-chat-hf`

- `--served-model-name <name>`  
    **Description:** The model name to use in API requests. Useful if you want to refer to the model by a different name than on Hugging Face.  
    **Example:** `--served-model-name my-custom-llama`

- `--host <host_address>`  
    **Description:** The IP address the server will listen on. Defaults to localhost. Set to `0.0.0.0` to allow access from other machines.  
    **Example:** `--host 0.0.0.0`

- `--tensor-parallel-size <num_gpus>`  
    **Description:** Number of GPUs to use for tensor parallelism. Useful for large models that don't fit on a single GPU.  
    **Example:** `--tensor-parallel-size 4`

- `--pipeline-parallel-size <num_gpus>`  
    **Description:** Number of GPUs to use for pipeline parallelism, splitting model layers across multiple GPUs.  
    **Example:** `--pipeline-parallel-size 2`

- `--quantization <method>`  
    **Description:** Quantization method to reduce model size and speed up inference (e.g., awq, gptq, aqlm, squeezellm, fp8).  
    **Example:** `--quantization awq`

- `--max-model-len <length>`  
    **Description:** Maximum context length the model can handle (in tokens).  
    **Example:** `--max-model-len 4096`

- `--max-num-seqs <num>`  
    **Description:** Maximum number of sequences (requests) that can be processed concurrently in a batch.  
    **Example:** `--max-num-seqs 16`

- `--max-num-batched-tokens <num>`  
    **Description:** Maximum total number of tokens (including prompt and generated tokens) in a batch. Important for tuning performance, especially with chunked prefill.  
    **Example:** `--max-num-batched-tokens 8192`

- `--gpu-memory-utilization <fraction>`  
    **Description:** Fraction of GPU memory used by vLLM for KV cache. Value from 0.0 to 1.0.  
    **Example:** `--gpu-memory-utilization 0.9` (uses 90% of GPU memory)

- `--trust-remote-code`  
    **Description:** Allow execution of remote code in the model repository on Hugging Face. Required for some models with custom code.  
    **Example:** `--trust-remote-code`

- `--enforce-eager`  
    **Description:** Force eager execution mode, disabling CUDAGraph optimization. Useful for debugging.  
    **Example:** `--enforce-eager`

- `--enable-lora`  
    **Description:** Enable support for LoRA (Low-Rank Adaptation).  
    **Example:** `--enable-lora --max-loras 4 --max-lora-rank 8`

- `--chat-template <template_path>`  
    **Description:** Path to a Jinja template file for chat formatting, or the name of a known template. Important for chat models, especially when using tool calling.  
    **Example:** `--chat-template examples/tool_chat_template_mistral.jinja`

- `--enable-auto-tool-choice`  
    **Description:** Allow the model to automatically decide when and which tool to call.  
    **Example:** `--enable-auto-tool-choice`

- `--guided-decoding-backend <backend_name>`  
    **Description:** Specify the backend for guided decoding, e.g., outlines.  
    **Example:** `--guided-decoding-backend outlines`

- `--api-key <your_api_key>`  
    **Description:** Set an API key for the server. The server will check this key in the request header.  
    **Example:** `--api-key mysecretkey`

- `--generation-config <config_name_or_path>`  
    **Description:** Specify the generation config. Can be set to `vllm` to use vLLM's default instead of the model's.  
    **Example:** `--generation-config vllm`

You can find the full list of parameters by running `vllm serve --help` or `uv run --with vllm vllm serve --help`. Example scripts often include an argument parser, and you can run the script with `--help` to see all available arguments compatible with LLMs, many of which also apply to `vllm serve`.
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
