CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code --dtype auto

chmod +x vllm_host.sh

./vllm_host.sh


