CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2-VL-2B-Instruct

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-VL-7B-Instruct

./vllm_host.sh
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
nvtop
gpustat (check gpu)
kill -9 305879


python3 main.py
or
python main.py --use_num_test_data --num_test_data 5


pip check 
pipdeptree
pipdeptree -w silence
