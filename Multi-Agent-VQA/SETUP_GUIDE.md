# Multi-Agent VQA Setup Guide

Hướng dẫn chi tiết để thiết lập và chạy hệ thống Multi-Agent VQA trên máy Linux với CUDA.

## Yêu cầu hệ thống

- Ubuntu/Linux với CUDA 12.x
- Python 3.10
- Conda environment

## Bước 1: Thiết lập môi trường Conda

```bash
# Tạo conda environment mới
conda create -n Leo_VQA python=3.10 -y
conda activate Leo_VQA
```

## Bước 2: Cài đặt PyTorch với CUDA support

```bash
# Cài đặt PyTorch với CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Kiểm tra CUDA hoạt động
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## Bước 3: Thiết lập submodules

```bash
cd Multi-Agent-VQA

# Khởi tạo và cập nhật submodules
git submodule update --init --recursive
```

## Bước 4: Cài đặt dependencies cơ bản

```bash
# Cài đặt dependencies bị thiếu
pip install idna certifi packaging python-dateutil wheel

# Cài đặt NumPy phiên bản tương thích
pip install "numpy<2.0"

# Cài đặt gradio và openai
pip install gradio openai
```

## Bước 5: Thiết lập Grounded-Segment-Anything

```bash
cd Grounded-Segment-Anything

# Cài đặt Segment Anything
pip install -e segment_anything

# Cài đặt GroundingDINO
pip install --no-build-isolation -e GroundingDINO

# Build extensions
cd GroundingDINO
python setup.py build_ext --inplace
cd ../..
```

## Bước 6: Thiết lập CLIP-Count

```bash
cd CLIP_Count

# Cài đặt requirements
pip install -r requirements.txt
pip install ftfy regex tqdm imgaug einops pytorch-lightning

# Cài đặt CLIP từ GitHub
pip install git+https://github.com/openai/CLIP.git

cd ..
```

## Bước 7: Sửa lỗi tương thích

### 7.1. Sửa lỗi torch._six trong CLIP_Count/util/misc.py

```python
# Thay thế dòng import torch._six
import torch
import torch.distributed as dist
import math
inf = math.inf
import numpy as np
```

### 7.2. Sửa import trong inference.py

```python
# Thêm vào đầu file inference.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'CLIP_Count'))
from run import Model as CLIP_Count
```

### 7.3. Sửa main.py để sử dụng GPU cụ thể

```python
# Thêm vào main.py trước device initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

## Bước 8: Tải các model weights

### 8.1. Tải GroundingDINO và SAM models

```bash
# Tải GroundingDINO model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P Grounded-Segment-Anything/

# Tải SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P Grounded-Segment-Anything/
```

### 8.2. Tải CLIP-Count checkpoint

```bash
# Tạo thư mục checkpoint
mkdir -p CLIP_Count/ckpt

# Tải checkpoint từ Google Drive
gdown 17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR -O CLIP_Count/ckpt/clipcount_pretrained.ckpt
```

## Bước 9: Cấu hình dataset paths

Cập nhật file `config.yaml` với đường dẫn dataset chính xác:

```yaml
datasets:
  dataset: 'vqa-v2'
  vqa_v2_dataset_split: 'rest-val'  # hoặc 'val', 'val1000', 'test-dev'
  vqa_v2_val_images_dir: '/mnt/VLAI_data/COCO_Images/val2014'
  vqa_v2_val_questions_file: '/mnt/VLAI_data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
  vqa_v2_val_annotations_file: '/mnt/VLAI_data/VQAv2/v2_mscoco_val2014_annotations.json'
  vqa_v2_rest_val_questions_file: '/mnt/VLAI_data/VQAv2/v2_OpenEnded_mscoco_rest_val2014_questions.json'
  vqa_v2_rest_val_annotations_file: '/mnt/VLAI_data/VQAv2/v2_mscoco_rest_val2014_annotations.json'
```

## Bước 10: Thiết lập OpenAI API Key

```bash
# Tạo file API key (thay YOUR_API_KEY bằng key thực tế)
echo "YOUR_OPENAI_API_KEY" > openai_key.txt

# Đảm bảo không có ký tự xuống dòng thừa
python -c "with open('openai_key.txt', 'r') as f: key = f.read().strip(); with open('openai_key.txt', 'w') as f: f.write(key)"
```

## Bước 11: Chạy hệ thống

```bash
# Tạo thư mục outputs
mkdir -p outputs

# Chạy với VQA-v2 rest-val dataset
python main.py --vlm_model gpt4 --dataset vqa-v2 --split rest-val --verbose

# Hoặc chạy với subset nhỏ hơn
python main.py --vlm_model gpt4 --dataset vqa-v2 --split val1000 --verbose
```

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**: Giảm batch_size hoặc sử dụng subset nhỏ hơn
2. **API key error**: Kiểm tra file `openai_key.txt` không có ký tự thừa
3. **Import errors**: Đảm bảo tất cả dependencies đã được cài đặt
4. **Model not found**: Kiểm tra các model weights đã được tải đúng vị trí

### Kiểm tra hệ thống:

```bash
# Kiểm tra CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# Kiểm tra models
ls -la Grounded-Segment-Anything/*.pth
ls -la CLIP_Count/ckpt/*.ckpt

# Kiểm tra API key
ls -la openai_key.txt
```

## Kết quả mong đợi

Khi chạy thành công, bạn sẽ thấy:
- Hệ thống load các models (GroundingDINO, SAM, CLIP-Count)
- Bắt đầu xử lý từng câu hỏi VQA
- Lưu kết quả vào file JSON trong thư mục `outputs/`

## Performance

- **Dataset**: VQA-v2 rest-val (~5,000 câu hỏi)
- **Model**: GPT-4 Vision + Multi-Agent approach
- **Estimated time**: Phụ thuộc vào API rate limit của OpenAI

## Credits

- Multi-Agent VQA paper và implementation
- Grounded-Segment-Anything
- CLIP-Count
- OpenAI GPT-4 Vision API 