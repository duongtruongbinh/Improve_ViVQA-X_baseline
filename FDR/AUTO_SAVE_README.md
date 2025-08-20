# Auto Save FDR Pipeline

Pipeline tự động lưu output và hình ảnh vào thư mục correct/wrong với đủ 3 cặp đúng/sai rồi tự dừng.

## 🎯 Tính năng

- **Tự động phân loại**: Tự động phân loại kết quả đúng/sai dựa trên ground truth
- **Lưu trữ có tổ chức**: Lưu vào thư mục `correct/` và `wrong/` riêng biệt
- **Tự động dừng**: Dừng khi đủ số lượng mẫu đúng/sai theo yêu cầu
- **Visualization**: Tạo hình ảnh output với thông tin đầy đủ
- **Random sampling**: Lấy mẫu ngẫu nhiên từ dataset

## 📁 Cấu trúc Output

```
auto_output/
├── correct/
│   ├── sample_1/
│   │   ├── info.json          # Thông tin text đầy đủ
│   │   ├── input_image.jpg    # Hình ảnh input
│   │   └── output_visualization.png  # Hình ảnh output
│   ├── sample_2/
│   └── sample_3/
├── wrong/
│   ├── sample_1/
│   ├── sample_2/
│   └── sample_3/
└── summary.json               # Tóm tắt kết quả
```

## 📋 Nội dung lưu trữ

### Correct/Wrong folders chứa:

**1. Text Information (info.json):**
- ID + Question + Answer + Explanation
- Evidence (bằng chứng) + Hypothesis (giả thuyết)
- Ground truth + Synthesis status
- Type (correct/wrong)

**2. Images:**
- Input image (hình ảnh gốc)
- Output visualization (hình ảnh kết quả với text overlay)

## 🚀 Cách sử dụng

### 1. Chạy với tham số mặc định (3 đúng, 3 sai)

```bash
cd FDR
python run_auto_save.py
```

### 2. Tùy chỉnh số lượng mẫu

```bash
# Lấy 5 mẫu đúng, 5 mẫu sai
python run_auto_save.py --correct 5 --wrong 5

# Chỉ lấy 2 mẫu đúng, 1 mẫu sai
python run_auto_save.py --correct 2 --wrong 1
```

### 3. Chọn backend

```bash
# Sử dụng vLLM (mặc định)
python run_auto_save.py --backend vllm

# Sử dụng OpenAI
python run_auto_save.py --backend openai
```

### 4. Chọn dataset

```bash
# VQA-X (English)
python run_auto_save.py --dataset vqax

# ViVQA-X (Vietnamese)
python run_auto_save.py --dataset vivqax
```

### 5. Tùy chỉnh thư mục output

```bash
python run_auto_save.py --output-dir my_results
```

### 6. Kết hợp các tham số

```bash
python run_auto_save.py \
    --backend openai \
    --correct 4 \
    --wrong 4 \
    --dataset vivqax \
    --output-dir vietnamese_results
```

## 📊 Tham số có sẵn

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--backend` | Backend: vllm/openai | vllm |
| `--correct` | Số mẫu đúng cần lấy | 3 |
| `--wrong` | Số mẫu sai cần lấy | 3 |
| `--output-dir` | Thư mục output | auto_output |
| `--dataset` | Dataset: vqax/vivqax | Từ config |
| `--config` | File config tùy chỉnh | config.yaml |

## 🔍 Ví dụ Output

### info.json
```json
{
  "id": "question_123",
  "question": "What color is the car?",
  "answer": "red",
  "explanation": "The car in the image is clearly red...",
  "evidence": ["visual_evidence_1", "visual_evidence_2"],
  "hypothesis": 2,
  "ground_truth": "red",
  "synthesis_status": "SUCCESS",
  "type": "correct"
}
```

### summary.json
```json
{
  "total_processed": 15,
  "correct_samples": 3,
  "wrong_samples": 3,
  "correct_samples_saved": [1, 2, 3],
  "wrong_samples_saved": [1, 2, 3],
  "target_reached": true
}
```

## ⚡ Lưu ý

1. **Tự động dừng**: Pipeline sẽ tự động dừng khi đủ số lượng mẫu đúng/sai
2. **Random sampling**: Mẫu được lấy ngẫu nhiên từ dataset
3. **Error handling**: Bỏ qua mẫu lỗi và tiếp tục xử lý
4. **Visualization**: Tự động tạo hình ảnh output với matplotlib
5. **UTF-8 support**: Hỗ trợ tiếng Việt trong output

## 🛠️ Yêu cầu

- Python 3.8+
- matplotlib
- PIL (Pillow)
- Các thư viện FDR khác (xem requirements.txt)

## 📝 Log

Pipeline sẽ hiển thị log chi tiết:
- Tiến độ xử lý
- Số lượng mẫu đã lưu
- Thông báo lỗi (nếu có)
- Tóm tắt cuối cùng 