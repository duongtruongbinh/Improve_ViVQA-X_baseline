#!/bin/bash
#
# GQA-REX Pipeline Server Startup Script
# Based on CLEVR-X execution pattern
#

echo "🚀 Starting vLLM servers for GQA-REX Pipeline"
echo "============================================="

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure CUDA is available."
    exit 1
fi

echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🔧 Starting Vision-Language Model Server (Port 9100)..."
echo "   Model: Qwen2.5-VL-7B-Instruct"
echo "   GPU: Device 2"

# Start VL model server in background
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct \
    --port 9100 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 &

VL_PID=$!
echo "   ✅ VL Server started (PID: $VL_PID)"

echo ""
echo "⏳ Waiting 30 seconds for VL server to initialize..."
sleep 30

echo ""
echo "🔧 Starting Language Model Server (Port 9200)..."
echo "   Model: Qwen2.5-72B-Instruct"
echo "   GPU: Device 3"

# Start LLM server in background
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-72B-Instruct \
    --port 9200 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &

LLM_PID=$!
echo "   ✅ LLM Server started (PID: $LLM_PID)"

echo ""
echo "⏳ Waiting 60 seconds for LLM server to initialize..."
sleep 60

echo ""
echo "🔍 Checking server status..."

# Check if servers are running
if curl -s http://localhost:9100/v1/models > /dev/null; then
    echo "   ✅ VL Server (Port 9100) is responding"
else
    echo "   ❌ VL Server (Port 9100) is not responding"
fi

if curl -s http://localhost:9200/v1/models > /dev/null; then
    echo "   ✅ LLM Server (Port 9200) is responding"
else
    echo "   ❌ LLM Server (Port 9200) is not responding"
fi

echo ""
echo "🎯 Servers are ready for GQA-REX Pipeline!"
echo "   • VL Server PID: $VL_PID"
echo "   • LLM Server PID: $LLM_PID"
echo ""
echo "📝 To stop servers, run:"
echo "   kill $VL_PID $LLM_PID"
echo ""
echo "🚀 You can now run: bash run_gqa_rex_pipeline.sh"

# Keep script running and wait for user input
echo ""
echo "Press Ctrl+C to stop all servers..."
trap "echo ''; echo '🛑 Stopping servers...'; kill $VL_PID $LLM_PID 2>/dev/null; exit 0" INT

# Wait indefinitely
while true; do
    sleep 10
done
