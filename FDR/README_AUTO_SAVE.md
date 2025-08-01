# Auto Save FDR Pipeline - Tóm tắt

## 🎯 Mục đích
Pipeline tự động lưu output và hình ảnh vào thư mục correct/wrong với đủ 3 cặp đúng/sai rồi tự dừng.

## 📁 Files đã tạo

### Core Pipeline
- `src/auto_save_pipeline.py` - Pipeline chính với logic tự động lưu
- `run_auto_save.py` - Script chính với command line arguments
- `quick_run.py` - Script chạy nhanh với cấu hình mặc định

### Utilities
- `view_results.py` - Xem và phân tích kết quả
- `test_auto_save.py` - Test pipeline
- `install_deps.py` - Cài đặt dependencies

### Documentation
- `AUTO_SAVE_README.md` - Hướng dẫn chi tiết
- `USAGE_GUIDE.md` - Hướng dẫn sử dụng ngắn gọn
- `README_AUTO_SAVE.md` - File này

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies
```bash
cd FDR
python install_deps.py
```

### 2. Chạy nhanh
```bash
python quick_run.py
```

### 3. Chạy với tùy chỉnh
```bash
python run_auto_save.py --correct 5 --wrong 5 --backend openai
```

### 4. Xem kết quả
```bash
python view_results.py --list
python view_results.py --show correct:1
python view_results.py --images correct:1
```

## 📊 Output Structure

```
auto_output/
├── correct/
│   ├── sample_1/
│   │   ├── info.json          # ID + Question + Answer + Explanation
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

## ⚡ Tính năng chính

- ✅ **Tự động phân loại**: Dựa trên ground truth
- ✅ **Tự động dừng**: Khi đủ số lượng mẫu đúng/sai
- ✅ **Random sampling**: Lấy mẫu ngẫu nhiên từ dataset
- ✅ **Visualization**: Tự động tạo hình ảnh output
- ✅ **UTF-8 support**: Hỗ trợ tiếng Việt
- ✅ **Error handling**: Bỏ qua mẫu lỗi và tiếp tục
- ✅ **Flexible**: Có thể tùy chỉnh số lượng, backend, dataset

## 🔧 Tham số có sẵn

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--backend` | Backend: vllm/openai | vllm |
| `--correct` | Số mẫu đúng cần lấy | 3 |
| `--wrong` | Số mẫu sai cần lấy | 3 |
| `--output-dir` | Thư mục output | auto_output |
| `--dataset` | Dataset: vqax/vivqax | Từ config |

## 📝 Ví dụ sử dụng

```bash
# Chạy với cấu hình mặc định
python quick_run.py

# Tùy chỉnh số lượng mẫu
python run_auto_save.py --correct 5 --wrong 5

# Sử dụng OpenAI backend
python run_auto_save.py --backend openai

# Chọn dataset ViVQA-X
python run_auto_save.py --dataset vivqax

# Kết hợp nhiều tham số
python run_auto_save.py --backend openai --correct 4 --wrong 4 --dataset vivqax --output-dir vietnamese_results
```

## 🎉 Hoàn thành!

Pipeline đã được tạo hoàn chỉnh với:
- Logic tự động phân loại và lưu trữ
- Visualization tự động
- Hỗ trợ đầy đủ command line arguments
- Tools để xem và phân tích kết quả
- Documentation chi tiết

Bạn có thể bắt đầu sử dụng ngay với `python quick_run.py`! 