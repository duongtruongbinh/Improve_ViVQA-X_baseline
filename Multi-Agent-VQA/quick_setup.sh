#!/bin/bash

# Multi-Agent VQA Quick Setup Script
# Run with: bash quick_setup.sh

set -e

echo "🚀 Starting Multi-Agent VQA Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda/Miniconda first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "Please run this script from the Multi-Agent-VQA directory"
    exit 1
fi

print_status "Step 1: Installing basic dependencies..."
pip install idna certifi packaging python-dateutil wheel
pip install "numpy<2.0"
pip install gradio openai huggingface_hub

print_status "Step 2: Setting up Grounded-Segment-Anything..."
cd Grounded-Segment-Anything

# Install Segment Anything
print_status "Installing Segment Anything..."
pip install -e segment_anything

# Install GroundingDINO
print_status "Installing GroundingDINO..."
pip install --no-build-isolation -e GroundingDINO

# Build extensions
print_status "Building GroundingDINO extensions..."
cd GroundingDINO
python setup.py build_ext --inplace
cd ../..

print_status "Step 3: Setting up CLIP-Count..."
cd CLIP_Count

# Install requirements
pip install -r requirements.txt
pip install ftfy regex tqdm imgaug einops pytorch-lightning

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

cd ..

print_status "Step 4: Downloading model weights..."

# Download GroundingDINO model
if [ ! -f "Grounded-Segment-Anything/groundingdino_swint_ogc.pth" ]; then
    print_status "Downloading GroundingDINO model..."
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P Grounded-Segment-Anything/
else
    print_status "GroundingDINO model already exists"
fi

# Download SAM model
if [ ! -f "Grounded-Segment-Anything/sam_vit_h_4b8939.pth" ]; then
    print_status "Downloading SAM model..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P Grounded-Segment-Anything/
else
    print_status "SAM model already exists"
fi

# Create CLIP-Count checkpoint directory
mkdir -p CLIP_Count/ckpt

# Download CLIP-Count checkpoint
if [ ! -f "CLIP_Count/ckpt/clipcount_pretrained.ckpt" ]; then
    print_status "Downloading CLIP-Count checkpoint..."
    if command -v gdown &> /dev/null; then
        gdown 17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR -O CLIP_Count/ckpt/clipcount_pretrained.ckpt
    else
        print_warning "gdown not installed. Installing it first..."
        pip install gdown
        gdown 17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR -O CLIP_Count/ckpt/clipcount_pretrained.ckpt
    fi
else
    print_status "CLIP-Count checkpoint already exists"
fi

print_status "Step 5: Applying code fixes..."

# Fix torch._six import in misc.py
python -c "
import os
file_path = 'CLIP_Count/util/misc.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace torch._six import
    if 'torch._six' in content:
        content = content.replace('import torch._six', '# import torch._six  # deprecated')
        content = content.replace('torch._six.inf', 'math.inf')
        
        # Ensure math is imported
        if 'import math' not in content:
            content = 'import math\n' + content
        
        with open(file_path, 'w') as f:
            f.write(content)
        print('Fixed torch._six import in misc.py')
    else:
        print('torch._six import already fixed or not found')
else:
    print('misc.py not found')
"

# Fix imports in inference.py
python -c "
import os
file_path = 'inference.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if sys.path fix is already applied
    if 'sys.path.append' not in content:
        # Add sys.path fix after imports
        lines = content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('from') or line.startswith('import'):
                import_end = i
        
        # Insert sys.path fix
        lines.insert(import_end + 1, '')
        lines.insert(import_end + 2, '# Add CLIP_Count to Python path')
        lines.insert(import_end + 3, 'sys.path.append(os.path.join(os.path.dirname(__file__), \"CLIP_Count\"))')
        
        content = '\n'.join(lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print('Fixed imports in inference.py')
    else:
        print('Import fixes already applied to inference.py')
else:
    print('inference.py not found')
"

# Fix main.py for single GPU usage
python -c "
import os
file_path = 'main.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if GPU fix is already applied
    if 'CUDA_VISIBLE_DEVICES' not in content:
        # Add GPU fix before device initialization
        content = content.replace(
            'device = torch.device(',
            '# Set CUDA device to GPU 0 only\n    os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n    device = torch.device('
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        print('Fixed GPU settings in main.py')
    else:
        print('GPU settings already fixed in main.py')
else:
    print('main.py not found')
"

print_status "Step 6: Creating outputs directory..."
mkdir -p outputs

print_status "Setup completed! 🎉"
echo ""
print_status "Next steps:"
echo "1. Set up your OpenAI API key:"
echo "   echo 'YOUR_OPENAI_API_KEY' > openai_key.txt"
echo ""
echo "2. Update config.yaml with your dataset paths"
echo ""
echo "3. Run the system:"
echo "   python main.py --vlm_model gpt4 --dataset vqa-v2 --split rest-val --verbose"
echo ""
print_warning "Make sure your dataset paths in config.yaml are correct!"
print_warning "Don't forget to add your OpenAI API key to openai_key.txt" 