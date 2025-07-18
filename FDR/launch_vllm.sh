#!/bin/bash

# FDR vLLM Server Launcher Script
# This script activates vllm_env and starts the vLLM server

echo "🚀 Starting FDR vLLM Server..."
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if vllm_env exists
if ! conda env list | grep -q "vllm_env"; then
    echo "❌ Error: vllm_env environment does not exist"
    echo "Please create it first with: conda create -n vllm_env python=3.10"
    exit 1
fi

echo "📦 Activating vllm_env environment..."

# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate vllm_env

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" != "vllm_env" ]]; then
    echo "❌ Error: Failed to activate vllm_env"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ Error: vLLM is not installed in vllm_env"
    echo "Please install it with: pip install vllm"
    exit 1
fi

echo "✅ vLLM is available"

# Change to the script directory
cd "$(dirname "$0")"

# Start the vLLM server
echo "🔥 Starting vLLM server..."
python start_vllm_server.py "$@"

echo "🛑 vLLM server stopped"
