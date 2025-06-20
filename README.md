# VQA Pipeline - Refactored Architecture

## 🎯 **Sequential Data Flow**
**Image → VLM → GroundingDINO → DAM → Seeker**

A production-ready Visual Question Answering pipeline with context-aware sequential processing.

---

## 🚀 **Quick Start**

### 1. Auto Setup (Recommended)
```bash
# One-command setup
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 2. Run Pipeline
```bash
# Test with 10 questions
python3 Top-Down/main.py --test --backend vllm

# Full dataset
python3 Top-Down/main.py --backend vllm
```

### 3. Check Results
```bash
# View results
cat Top-Down/output/vivqax_summary.txt
ls Top-Down/output/
```

---

## 🏗️ **Architecture**

### Pipeline Components
- **VLM**: `Qwen/Qwen2.5-VL-7B-Instruct` via vLLM server
- **GroundingDINO**: Object detection via Docker container
- **DAM**: `nvidia/DAM-3B-Self-Contained` for enhanced analysis
- **Seeker**: Multi-View Knowledge Base construction

### Data Flow
```
📸 Image Input
    ↓
📝 VLM.process() → Description with object details
    ↓
🎯 GroundingDINO.generate(BBox) → Annotated image from description
    ↓
🔍 DAM.process() → Enhanced analysis with bounding box context
    ↓
🧠 Seeker.receive() → Multi-View Knowledge Base → Final answer
```

### Key Features
- ✅ **Context-aware**: Each step builds on previous step's output
- ✅ **Docker Integration**: GroundingDINO runs in isolated container
- ✅ **Auto-fallback**: GPU → CPU, Native → Docker transitions
- ✅ **Production-ready**: Complete error handling and logging
- ✅ **Modular**: Each component can be used independently

---

## 📁 **Project Structure**

```
VQA/
├── Top-Down/                    # Main SIRI framework
│   ├── core/
│   │   ├── agents.py           # Refactored pipeline (MAIN)
│   │   └── pipeline.py         # Pipeline orchestration
│   ├── configs/
│   │   └── vivqax_config.yaml  # ViVQA-X configuration
│   ├── main.py                 # Entry point
│   └── output/                 # Results directory
├── GroundingDINO/              # Object detection service
│   ├── Dockerfile              # Docker container setup
│   └── groundingdino/          # Core detection model
├── DAM/                        # Describe Anything Model
│   ├── single_inference.py     # Reference implementation
│   └── dam/                    # Core DAM model
├── scripts/
│   ├── setup_environment.sh    # Auto setup script (NEW)
│   └── groundingdino_service.py # Docker service wrapper
├── REFACTOR_LOG.md             # Detailed change log
└── README.md                   # This file
```

---

## 🔧 **Configuration**

### Main Config: `Top-Down/configs/vivqax_config.yaml`
```yaml
data_config:
  dataset_name: "ViVQA-X" 
  data_path: "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
  image_dir: "/mnt/VLAI_data/COCO_Images/val2014"
  num_questions: 10  # -1 for full dataset

agents_config:
  responder:
    enable_dam: true
    enable_groundingdino: true
    groundingdino_docker: true  # Use Docker service
    model_name: "Qwen/Qwen2.5-VL-7B-Instruct"

output_config:
  results_file: "Top-Down/output/vivqax_results.json"
  summary_file: "Top-Down/output/vivqax_summary.txt"
  enable_visualization: true
```

---

## 🛠️ **Manual Setup** (if auto setup fails)

### Dependencies
```bash
# Core dependencies  
pip install supervision torchvision transformers torch pillow numpy

# vLLM server (separate terminal)
pip install vllm
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 9100 --host localhost
```

### GroundingDINO Docker
```bash
cd GroundingDINO
docker build -t groundingdino:latest .

# Test
docker run --rm --gpus all groundingdino:latest \
    python -c "print('GroundingDINO ready!')"
```

### DAM Model
```bash
# Will auto-download on first run
python3 -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/DAM-3B-Self-Contained', 
                                 trust_remote_code=True)
print('DAM ready!')
"
```

---

## 📊 **Testing & Validation**

### Test Pipeline
```bash
# Quick test (10 questions)
python3 Top-Down/main.py --test --backend vllm

# Expected output:
# ✅ VLM initialized
# ✅ GroundingDINO Docker ready  
# ✅ DAM loaded successfully
# Pipeline processing: 100%|██████████| 10/10
```

### Check Components
```bash
# Test individual components
python3 scripts/groundingdino_service.py  # GroundingDINO
python3 DAM/single_inference.py           # DAM
```

### Troubleshooting
- **GPU Issues**: Pipeline auto-falls back to CPU
- **Docker Issues**: Check `docker ps` and `docker images`
- **vLLM Issues**: Ensure server running on localhost:9100
- **Memory Issues**: Use smaller batch sizes in config

---

## 📈 **Performance & Monitoring**

### Logs
```bash
# Real-time monitoring
tail -f Top-Down/output/pipeline.log

# Key indicators:
# "📝 Step 1: VLM.process()" - VLM working
# "🎯 Step 2: GroundingDINO.generate(BBox)" - Detection working  
# "🔍 Step 3: DAM.process()" - DAM working
# "✅ Step 4: Seeker.receive()" - Pipeline complete
```

### Output Files
- `vivqax_results.json` - Detailed results per question
- `vivqax_summary.txt` - Summary statistics
- `pipeline_images/` - Visualization images (if enabled)

---

## 🔄 **Development**

### Adding New Components
1. Extend `ResponderAgent` class in `Top-Down/core/agents.py`
2. Add initialization in `__init__()` method
3. Integrate in `generate_initial_response()` pipeline
4. Update configuration schema

### Debugging Pipeline
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Pipeline step-by-step
responder = ResponderAgent(enable_dam=True)
result = responder.generate_initial_response(question, image_path)
```

---

## 📝 **Change Log**

See [`REFACTOR_LOG.md`](REFACTOR_LOG.md) for detailed refactoring history.

### Major Changes
- ✅ **Sequential Flow**: Image → VLM → GroundingDINO → DAM → Seeker
- ✅ **Docker Integration**: GroundingDINO in container
- ✅ **Bug Fixes**: DAM API, dependency issues
- ✅ **Auto Setup**: One-command environment setup
- ✅ **Production Ready**: Error handling, fallbacks, monitoring

---

## 🤝 **Contributing**

1. Follow the sequential pipeline pattern
2. Add comprehensive error handling
3. Update documentation and tests
4. Maintain Docker compatibility

---

## 📞 **Support**

- **Pipeline Issues**: Check `REFACTOR_LOG.md` 
- **Component Issues**: Test individual components first
- **Performance**: Monitor logs and resource usage
- **Setup Issues**: Use auto setup script

---

**Status**: 🎉 **Production Ready** - Complete pipeline implementation! 