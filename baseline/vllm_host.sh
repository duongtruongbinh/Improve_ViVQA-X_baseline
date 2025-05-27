#!/bin/bash

# --- General Config ---
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
GPU_ID="0" 
PYTHON_CMD="python3" 
TRUST_REMOTE_CODE_ARG="--trust-remote-code"
QUANTIZATION_ARG="--quantization gptq"
DTYPE_ARG="--dtype auto"

# --- Server 1 Config ---
PORT1="8000"
GPU_MEM_UTIL1="0.40" 

# --- Server 2 Config ---
PORT2="8001"
GPU_MEM_UTIL2="0.40" 

# --- Launch Server 1 ---
echo "Launching Server 1: Port ${PORT1}, Model ${MODEL_NAME}"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_CMD} -m vllm.entrypoints.openai.api_server --model "${MODEL_NAME}" --port "${PORT1}" ${QUANTIZATION_ARG} ${DTYPE_ARG} --gpu-memory-utilization "${GPU_MEM_UTIL1}" ${TRUST_REMOTE_CODE_ARG} &

# --- Launch Server 2 ---
echo "Launching Server 2: Port ${PORT2}, Model ${MODEL_NAME}"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_CMD} -m vllm.entrypoints.openai.api_server --model "${MODEL_NAME}" --port "${PORT2}" ${QUANTIZATION_ARG} ${DTYPE_ARG} --gpu-memory-utilization "${GPU_MEM_UTIL2}" ${TRUST_REMOTE_CODE_ARG} &

echo -e "\nBoth servers have been requested to launch in the background."
echo "Their output will go to this terminal's stdout/stderr if they produce any before backgrounding fully."
echo "The script will exit now. Servers (if started successfully) will continue running."
