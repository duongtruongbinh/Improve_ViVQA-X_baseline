# GQA-REX Pipeline - Validation Set Configuration

## ✅ Xác nhận: Pipeline được cấu hình để chạy trên tập VALIDATION của GQA-REX

### 📊 Thông tin Dataset Validation:
- **Tổng số samples**: 127,900 (đã link giữa GQA và GQA-REX)
- **GQA questions**: 132,062 
- **REX explanations**: 127,900
- **Scene graphs**: 10,696
- **Dataset split**: validation

### 🚀 Cách chạy Pipeline trên tập Validation:

#### Phương pháp 1: Script đơn giản
```bash
cd /home/huytd/multi-agent/multi-agent/FDR/gqa_rex_pipeline
python run_validation.py
```

#### Phương pháp 2: Script với shell
```bash
cd /home/huytd/multi-agent/multi-agent/FDR/gqa_rex_pipeline
bash run_gqa_rex_pipeline.sh --split val --num_samples 100
```

#### Phương pháp 3: Main script trực tiếp
```bash
cd /home/huytd/multi-agent/multi-agent/FDR
python main_gqa_rex.py --dataset gqa_rex_val --backend vllm --evaluate --samples 100
```

### 🔧 Cấu hình mặc định:
- **Active dataset**: `gqa_rex_val` (validation set)
- **Default samples**: 100 (có thể thay đổi)
- **Backend**: vLLM
- **Evaluation**: Enabled
- **Output**: `results_gqa_rex_val/`

### 📋 Test & Validation:
```bash
# Test dataset loading
cd /home/huytd/multi-agent/multi-agent/FDR/gqa_rex_pipeline
python confirm_val_dataset.py

# Test with few samples
python run_validation.py
```

### 🎯 Kết quả:
Pipeline sẽ xử lý các câu hỏi từ tập validation GQA với explanations từ GQA-REX, đánh giá accuracy và quality của reasoning explanations.
