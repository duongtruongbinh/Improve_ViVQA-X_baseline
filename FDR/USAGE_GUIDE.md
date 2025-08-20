# Hướng dẫn sử dụng Auto Save Pipeline

## 🚀 Chạy nhanh

```bash
cd FDR
python quick_run.py
```

## 📋 Các lệnh chính

### 1. Chạy pipeline
```bash
# Mặc định (3 đúng, 3 sai)
python run_auto_save.py

# Tùy chỉnh số lượng
python run_auto_save.py --correct 5 --wrong 5

# Chọn backend
python run_auto_save.py --backend openai

# Chọn dataset
python run_auto_save.py --dataset vivqax
```

### 2. Xem kết quả
```bash
# Tóm tắt
python view_results.py

# Liệt kê tất cả mẫu
python view_results.py --list

# Xem chi tiết mẫu cụ thể
python view_results.py --show correct:1

# Xem hình ảnh
python view_results.py --images correct:1

# Phân tích
python view_results.py --analyze
```

### 3. Test
```bash
python test_auto_save.py
```

## 📁 Cấu trúc output

```
auto_output/
├── correct/
│   ├── sample_1/
│   │   ├── info.json          # Thông tin đầy đủ
│   │   ├── input_image.jpg    # Hình gốc
│   │   └── output_visualization.png  # Hình kết quả
│   ├── sample_2/
│   └── sample_3/
├── wrong/
│   ├── sample_1/
│   ├── sample_2/
│   └── sample_3/
└── summary.json               # Tóm tắt
```

## 📊 Nội dung lưu trữ

**Correct/Wrong folders chứa:**
- **ID + Question + Answer + Explanation**
- **Evidence (bằng chứng) + Hypothesis (giả thuyết)**
- **Ground truth + Synthesis status**
- **Input image + Output visualization**

## ⚡ Tính năng

- ✅ Tự động phân loại đúng/sai
- ✅ Tự động dừng khi đủ mẫu
- ✅ Random sampling
- ✅ Visualization tự động
- ✅ Hỗ trợ tiếng Việt
- ✅ Error handling 