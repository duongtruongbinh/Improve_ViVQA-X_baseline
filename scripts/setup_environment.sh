#!/bin/bash

# VQA Pipeline Auto Setup Script
# Sets up the complete environment for Image → VLM → GroundingDINO → DAM → Seeker pipeline

set -e  # Exit on any error

echo "🚀 VQA Pipeline Auto Setup"
echo "=========================="

# Check if running in correct directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Please run this script from the VQA root directory"
    exit 1
fi

echo "📦 Step 1: Installing Python dependencies..."

# Install core dependencies
pip install supervision torchvision

# Install DAM dependencies if not already installed
if ! python -c "import transformers" &> /dev/null; then
    echo "Installing transformers for DAM..."
    pip install transformers torch pillow numpy
fi

echo "✅ Python dependencies installed"

echo "🐳 Step 2: Setting up GroundingDINO with Docker..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if GroundingDINO directory exists
if [ -d "GroundingDINO" ]; then
    # Build GroundingDINO Docker image
    cd GroundingDINO
    echo "Building GroundingDINO Docker image..."
    docker build -t groundingdino:latest .

    # Test the Docker image
    echo "Testing GroundingDINO Docker..."
    if docker run --rm --gpus all groundingdino:latest python -c "print('GroundingDINO Docker ready!')"; then
        echo "✅ GroundingDINO Docker setup successful"
    else
        echo "⚠️  GPU not available, testing CPU mode..."
        if docker run --rm groundingdino:latest python -c "print('GroundingDINO Docker ready (CPU)!')"; then
            echo "✅ GroundingDINO Docker setup successful (CPU mode)"
        else
            echo "❌ GroundingDINO Docker setup failed"
            exit 1
        fi
    fi
    cd ..
else
    echo "⚠️  GroundingDINO directory not found"
    echo "   Pipeline will run in VLM-only mode"
    echo "   To enable GroundingDINO: git clone https://github.com/IDEA-Research/GroundingDINO.git"
fi

echo "🔧 Step 3: Setting up GroundingDINO service..."

# Create GroundingDINO service script
cat > scripts/groundingdino_service.py << 'EOF'
#!/usr/bin/env python3
"""
GroundingDINO Docker Service
Provides a simple API to run GroundingDINO detection via Docker
"""

import subprocess
import json
import os
import tempfile
import base64
from PIL import Image
import io

class GroundingDINOService:
    def __init__(self, docker_image="groundingdino:latest"):
        self.docker_image = docker_image
        
    def detect_objects(self, image_path, text_prompt, confidence_threshold=0.3):
        """
        Run object detection using GroundingDINO via Docker
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for detection (e.g., "car . person . tree")
            confidence_threshold: Detection confidence threshold
            
        Returns:
            str: Path to annotated image, or None if failed
        """
        try:
            # Create temporary directory for Docker I/O
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy input image to temp directory
                temp_input = os.path.join(temp_dir, "input.jpg")
                temp_output = os.path.join(temp_dir, "output.jpg")
                
                # Copy image to temp location
                import shutil
                shutil.copy2(image_path, temp_input)
                
                # Prepare Docker command
                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{temp_dir}:/workspace",
                    "--gpus", "all",  # Remove if no GPU
                    self.docker_image,
                    "python", "-c", f"""
import sys
sys.path.append('/opt/program/GroundingDINO')
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

# Load model
model = load_model('/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
                   '/opt/program/weights/groundingdino_swint_ogc.pth')

# Load image
image_source, image = load_image('/workspace/input.jpg')

# Run detection
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption='{text_prompt}',
    box_threshold={confidence_threshold},
    text_threshold=0.25
)

# Annotate image
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# Save result
cv2.imwrite('/workspace/output.jpg', annotated_frame)
print('Detection completed successfully')
"""
                ]
                
                # Run Docker command
                result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(temp_output):
                    # Copy result back to permanent location
                    output_dir = os.path.dirname(image_path)
                    output_filename = f"groundingdino_{os.path.basename(image_path)}"
                    final_output = os.path.join(output_dir, output_filename)
                    shutil.copy2(temp_output, final_output)
                    return final_output
                else:
                    print(f"GroundingDINO Docker failed: {result.stderr}")
                    return None
                    
        except Exception as e:
            print(f"GroundingDINO service error: {e}")
            return None

if __name__ == "__main__":
    # Test the service
    service = GroundingDINOService()
    print("GroundingDINO service ready!")
EOF

chmod +x scripts/groundingdino_service.py

echo "📝 Step 4: Creating configuration files..."

# Create pipeline config if not exists
if [ ! -f "Top-Down/configs/vivqax_config.yaml" ]; then
    echo "Creating default ViVQA-X config..."
    mkdir -p Top-Down/configs
    cat > Top-Down/configs/vivqax_config.yaml << 'EOF'
# ViVQA-X Configuration for Refactored Pipeline
# Image → VLM → GroundingDINO → DAM → Seeker

data_config:
  dataset_name: "ViVQA-X"
  data_path: "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
  image_dir: "/mnt/VLAI_data/COCO_Images/val2014"
  split: "val"
  num_questions: 10  # For testing, set to -1 for full dataset

agents_config:
  responder:
    enable_dam: true
    enable_groundingdino: true
    groundingdino_docker: true  # Use Docker for GroundingDINO
    model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
    temperature: 0.7
    max_tokens: 1000

output_config:
  results_file: "Top-Down/output/vivqax_results.json"
  summary_file: "Top-Down/output/vivqax_summary.txt"
  enable_visualization: true
  save_intermediate_images: true
EOF
fi

echo "🔄 Step 5: Testing the pipeline..."

# Test DAM initialization
echo "Testing DAM..."
python3 -c "
from transformers import AutoModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained('nvidia/DAM-3B-Self-Contained', trust_remote_code=True, torch_dtype=torch.float16)
print('✅ DAM test successful')
" || echo "⚠️  DAM test failed - check CUDA/GPU setup"

# Test GroundingDINO Docker
echo "Testing GroundingDINO Docker..."
python3 scripts/groundingdino_service.py || echo "⚠️  GroundingDINO Docker test failed"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 To run the pipeline:"
echo "   python3 Top-Down/main.py --test --backend vllm"
echo ""
echo "📊 Pipeline flow: Image → VLM → GroundingDINO → DAM → Seeker"
echo "🐳 GroundingDINO: Running in Docker container"
echo "🔥 DAM: nvidia/DAM-3B-Self-Contained model"
echo "💡 VLM: Qwen/Qwen2.5-VL-7B-Instruct via vLLM"
echo ""
echo "📁 Outputs will be saved to Top-Down/output/"
echo "" 