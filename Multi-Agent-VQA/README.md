# Multi-Agent VQA: Setup and Run Guide

This project explores zero-shot capabilities of foundation models in Visual Question Answering (VQA) tasks using an adaptive multi-agent system. The system uses multiple specialized agents working together to answer complex visual questions.

![Pipeline](pipeline.png)

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: Docker Setup (Recommended)](#method-1-docker-setup-recommended)
  - [Method 2: Normal Setup](#method-2-normal-setup)
- [Configuration](#configuration)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- At least 16GB RAM
- At least 50GB free disk space for models and datasets

### Software Requirements
- Linux/Unix-based operating system (tested on Ubuntu 20.04+)
- NVIDIA drivers (version 470+ recommended)
- Git

---

## Installation Methods

### Method 1: Docker Setup (Recommended)

Docker setup provides an isolated environment with all dependencies pre-configured.

#### Prerequisites
- Docker installed ([Installation Guide](https://docs.docker.com/engine/install/))
- NVIDIA Container Toolkit for GPU support ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

#### Step 1: Clone the Repository
```bash
git clone --recursive https://github.com/your-repo/Multi-Agent-VQA.git
cd Multi-Agent-VQA
```

#### Step 2: Configure Host Directory Mapping
The default Makefile mounts `/raid0/docker-raid/bwjiang` from the host to `/usr/src/app` in the container. You need to either:

**Option A: Use the default path**
```bash
sudo mkdir -p /raid0/docker-raid/bwjiang
sudo cp -r . /raid0/docker-raid/bwjiang/
```

**Option B: Modify the Makefile for your preferred directory**
Edit the `Makefile` and change line 32:
```makefile
# Change this line:
-v /raid0/docker-raid/bwjiang:/usr/src/app \
# To your preferred path, for example:
-v $(PWD):/usr/src/app \
```

#### Step 3: Build Docker Image
```bash
make build-image
```

This will:
- Automatically detect your CUDA version
- Build the image with appropriate CUDA support
- Install all required dependencies

#### Step 4: Run Docker Container
```bash
make run
```

This will start an interactive container with:
- GPU access enabled
- X11 forwarding for GUI applications
- All dependencies pre-installed
- Project directory mounted

#### Step 5: Verify Installation (Inside Container)
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if all dependencies are installed
python -c "import transformers, diffusers, opencv; print('All dependencies installed successfully')"
```

---

### Method 2: Normal Setup

Manual installation on your local machine.

#### Prerequisites
- Python 3.8 or higher
- CUDA 12.1+ (for GPU support)
- Git

#### Step 1: Clone the Repository
```bash
git clone --recursive https://github.com/your-repo/Multi-Agent-VQA.git
cd Multi-Agent-VQA
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install PyTorch
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4: Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 git build-essential python3-dev

# CentOS/RHEL
sudo yum install -y ffmpeg libSM libXext git gcc gcc-c++ python3-devel
```

#### Step 5: Install Grounded-Segment-Anything
```bash
cd Grounded-Segment-Anything

# Install Segment Anything
pip install -e segment_anything

# Install GroundingDINO
pip install wheel
pip install --no-build-isolation -e GroundingDINO

# Build GroundingDINO extensions
cd GroundingDINO
python setup.py build_ext --inplace
cd ../..
```

#### Step 6: Install CLIP-Count Requirements
```bash
cd CLIP_Count
pip install -r requirements.txt
pip install ftfy regex tqdm imgaug einops pytorch-lightning
pip install git+https://github.com/openai/CLIP.git
cd ..
```

#### Step 7: Install Main Requirements
```bash
pip install -r requirements.txt
```

#### Step 8: Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers, diffusers, cv2; print('Installation successful')"
```

---

## Configuration

### 1. API Keys Setup
Create API key files for the language models:

#### For GPT-4 (OpenAI)
Create `LLM_api_keys/openai_api_key.txt` and add your OpenAI API key.

#### For Gemini (Google Cloud)
1. Create a Google Cloud project
2. Enable the Vertex AI API
3. Create a service account and download the JSON key file
4. Update the credential path in `main.py` (line 19) and `config.yaml`

### 2. Configuration File
Edit `config.yaml` to customize:
- Dataset paths
- Model parameters
- Inference settings
- Output configurations

### 3. Download Required Models```bash
# Create model directory
cd Grounded-Segment-Anything

# Download GroundingDINO model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P Grounded-Segment-Anything/

# Download SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P Grounded-Segment-Anything/
```

---

## Dataset Setup

### GQA Dataset
```bash
mkdir -p /tmp/datasets/gqa
cd /tmp/datasets/gqa

# Download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# Download questions
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip questions1.2.zip
```

### VQA v2 Dataset
```bash
mkdir -p /tmp/datasets/coco
cd /tmp/datasets/coco

# Download COCO images
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip
unzip val2014.zip
unzip test2015.zip

# Download VQA annotations
mkdir vqa
cd vqa
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
```

---

## Usage

### Basic Usage

#### For Docker Setup
```bash
# Start the container
make run

# Inside container, run inference
python main.py --vlm_model gpt4 --dataset vqa-v2 --split val1000
```

#### For Normal Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Run inference
python main.py --vlm_model gpt4 --dataset vqa-v2 --split val1000
```

### Command Line Arguments
- `--vlm_model`: Choose VLM model (`gpt4`, `gemini`)
- `--dataset`: Choose dataset (`gqa`, `vqa-v2`)
- `--split`: Choose dataset split
  - For GQA: `val`, `val-subset`, `test`
  - For VQA-v2: `val`, `rest-val`, `val1000`, `test-dev`, `test-std`
- `--verbose`: Enable verbose output

### Example Commands
```bash
# Run on GQA validation set with GPT-4
python main.py --vlm_model gpt4 --dataset gqa --split val --verbose

# Run on VQA-v2 with Gemini
python main.py --vlm_model gemini --dataset vqa-v2 --split val1000

# Quick test with small subset
python main.py --vlm_model gpt4 --dataset vqa-v2 --split val1000
```

---

## Project Structure

```
Multi-Agent-VQA/
├── main.py                     # Main entry point
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── Makefile                    # Build and run scripts
├── inference.py                # Inference logic
├── dataloader.py              # Dataset loading
├── query_vlm.py               # VLM querying
├── query_llm.py               # LLM querying
├── utils.py                   # Utility functions
├── Grounded-Segment-Anything/ # Submodule for segmentation
├── CLIP_Count/                # Submodule for counting
├── src/                       # Source code
├── utils_func/                # Utility functions
└── outputs/                   # Output results
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size in config.yaml
- Use smaller models
- Clear GPU cache: `torch.cuda.empty_cache()`

#### Permission Errors (Docker)
```bash
# Fix ownership issues
sudo chown -R $(id -u):$(id -g) /path/to/your/directory
```

#### Missing Models
Ensure all required models are downloaded:
```bash
ls -la Grounded-Segment-Anything/
# Should contain: groundingdino_swint_ogc.pth, sam_vit_h_4b8939.pth
```

#### API Key Issues
- Verify API keys are correctly set
- Check credential file paths in main.py and config.yaml
- Ensure proper permissions on credential files

#### Build Errors (Normal Setup)
```bash
# Install build dependencies
sudo apt-get install build-essential python3-dev

# Reinstall with verbose output
pip install -v -e GroundingDINO
```

### Getting Help
- Check the [Issues](https://github.com/your-repo/Multi-Agent-VQA/issues) page
- Ensure all submodules are properly initialized: `git submodule update --init --recursive`
- Verify CUDA compatibility with PyTorch version

---

## License
This project is licensed under the terms specified in the LICENSE file.

## Citation
If you use this work in your research, please cite our paper:
```bibtex
@article{multi-agent-vqa,
  title={Multi-Agent VQA: Adaptive Multi-Agent System for Visual Question Answering},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
