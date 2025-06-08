#!/bin/bash

# Multi-Agent ViVQA-X Evaluation Script
# This script starts vLLM server and runs multi-agent evaluation

set -e

# Configuration
MODEL_PATH="/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
CONFIG_PATH="configs/vivqa_x_config.yaml"
CONDA_ENV="ma_vqa"
GPU_ID=1
VLLM_PORT=8000

echo "🚀 Starting Multi-Agent ViVQA-X Evaluation"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_PATH"
echo "GPU: $GPU_ID"
echo "Port: $VLLM_PORT"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "❌ Conda environment '$CONDA_ENV' not found!"
    echo "Please create it first: conda create -n $CONDA_ENV python=3.9"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "✅ All checks passed. Starting evaluation..."
echo ""

# Function to cleanup
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    # Kill vLLM server if running
    pkill -f "vllm.entrypoints.openai.api_server" || true
    echo "✅ Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Start vLLM server in background
echo "🔥 Starting vLLM server..."
conda activate $CONDA_ENV
export CUDA_VISIBLE_DEVICES=$GPU_ID

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --enforce-eager \
    --trust-remote-code > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "📡 vLLM server started (PID: $VLLM_PID)"
echo "📋 Server logs: vllm_server.log"

# Wait for server to be ready
echo "⏳ Waiting for vLLM server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "✅ vLLM server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ vLLM server failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

echo ""
echo "🤖 Starting Multi-Agent Evaluation..."

# Run evaluation
cd scripts
python simple_multiagent_vivqa_x.py --config "../$CONFIG_PATH" --verbose

echo ""
echo "🎉 Evaluation completed successfully!"
echo "📊 Check the outputs/ directory for results"
