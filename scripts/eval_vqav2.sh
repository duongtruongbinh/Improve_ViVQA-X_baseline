# Run QwenVL for VQAv2
cd src/inference
CUDA_VISIBLE_DEVICES=0 python qwenvl_vqav2.py

# Eval predictions
cd ../evaluation
python eval_vqa.py --predict ../../results/VQAv2_Qwen2-VL-2B-Instruct.json
