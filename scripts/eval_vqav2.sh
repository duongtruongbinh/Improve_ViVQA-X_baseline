# Run QwenVL for VQAv2
cd src/inference
CUDA_VISIBLE_DEVICES=0 python qwenvl_vqav2.py

# Eval predictions
cd ../evaluation
CUDA_VISIBLE_DEVICES=0 python eval_vqa.py --preds ../../results/VQAv2_Qwen2.5-VL-7B-Instruct.json

