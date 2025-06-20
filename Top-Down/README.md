# SIRI VQA Framework - Enhanced Local Pipeline

**SIRI (Seeker, Integrator, Responder)** is an advanced Visual Question Answering framework implementing a visualize-then-analyze pipeline with 100% local inference capabilities.

## 🚀 Key Features

- **100% Local Processing**: No external API dependencies
- **Enhanced Visual Pipeline**: GroundingDINO → DAM → vLLM analysis
- **Multi-Agent Architecture**: Responder, Seeker, and Integrator agents
- **Explainable AI**: Full reasoning traces with visual annotations
- **Flexible Backends**: Support for both vLLM (default) and OpenAI API
- **Advanced Object Detection**: Grounding-based visual understanding

## 🏗️ Architecture Overview

### Pipeline Flow
```
Question → Detection Keywords → GroundingDINO → Annotated Image → DAM Analysis → vLLM Reasoning → Final Answer
```

### Agent System
1. **ResponderAgent**: Enhanced with visual object detection and analysis
2. **SeekerAgent**: Generates sub-questions and hypotheses  
3. **IntegratorAgent**: Weighted voting and decision integration

## 📦 Installation

### Prerequisites
- CUDA-compatible GPU
- Python 3.8+
- Conda package manager

### Environment Setup
```bash
# Create and activate environment
conda env create -f Top-Down/VQA_env.yaml
conda activate VQA_env

# Verify GroundingDINO weights are downloaded (693MB)
ls GroundingDINO/weights/groundingdino_swint_ogc.pth
```

### vLLM Server Setup
Start the local vLLM server for inference:
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 9100 \
    --api-key dummy-key \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192
```

## ⚙️ Configuration

### Main Configuration
Edit `Top-Down/config.yaml` for vLLM settings:
```yaml
vllm:
  base_url: "http://localhost:9100/v1"
  model: "Qwen/Qwen2.5-VL-7B-Instruct"
  api_key: "dummy-key"
  max_tokens: 1000
  temperature: 0.7
```

### Dataset Configuration
Configure your dataset in `Top-Down/configs/vivqa_config.yaml`:
```yaml
inference:
  dataset_split: "val"
  num_questions: 5  # Set to -1 for full dataset

dataset_paths:
  questions_file: "/path/to/your/questions.json"
  images_dir: "/path/to/your/images/"
```

## 🚀 Usage

### Backend Options (Simple)

**Option 1: vLLM (Default)**
```bash
# Start vLLM server first (see Installation)
python Top-Down/main.py --backend vllm
```

**Option 2: OpenAI API (gpt-4o-mini)**
```bash
# Create API key file
cp Top-Down/openai_key.txt.template Top-Down/openai_key.txt
# Edit openai_key.txt with your real API key

# Run with OpenAI
python Top-Down/main.py --backend openai
```

### Test Your Setup
```bash
# Test both backends
python Top-Down/test_backends.py
```

### Quick Start Examples
```bash
# Test with 5 questions using vLLM (default)
python Top-Down/main.py --config Top-Down/configs/vllm_test_config.yaml

# Full dataset with vLLM
python Top-Down/main.py --backend vllm

# Quick test with OpenAI
python Top-Down/main.py --backend openai --config Top-Down/configs/vllm_test_config.yaml
```

### Testing vLLM Connection
```bash
curl -X POST "http://localhost:9100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

## 📊 Output

Results are saved to `Top-Down/output/`:
- **`siri_pipeline_results.json`**: Detailed results with visual annotations and reasoning traces
- **`siri_summary.txt`**: Summary statistics and accuracy metrics

### Sample Output Structure
```json
{
  "question_id": 12345,
  "question": "What color is the car?",
  "ground_truth_answer": "red", 
  "final_answer": "red",
  "is_correct": true,
  "explainability_trace": {
    "visual_detection": {
      "detected_objects": ["car"],
      "annotated_image_path": "output/annotated_12345.png"
    },
    "dam_analysis": "Detailed analysis of the red car...",
    "multi_view_knowledge_base": [...],
    "reasoning_steps": [...]
  }
}
```

## 🔧 Advanced Features

### Visual Object Detection
The pipeline automatically:
1. Extracts detection keywords from questions
2. Uses GroundingDINO for precise object localization
3. Generates annotated images with bounding boxes
4. Provides visual context for enhanced reasoning

### Fallback System
- **Component-level**: GroundingDINO/DAM failures fallback to direct VLM
- **Backend-level**: vLLM failures can fallback to OpenAI API
- **Graceful degradation**: System continues with reduced capabilities

## 📁 Project Structure

```
Top-Down/
├── README.md                 # This file
├── main.py                   # Entry point
├── config.yaml              # vLLM configuration  
├── config_loader.py          # Configuration management
├── vllm_client.py           # vLLM client wrapper
├── VQA_env.yaml             # Conda environment
├── core/
│   ├── agents.py            # SIRI agents implementation
│   └── pipeline.py          # Main pipeline orchestration
├── configs/
│   ├── vivqa_config.yaml    # Dataset configuration
│   └── vllm_test_config.yaml # Test configuration
├── docs/                    # Documentation
└── output/                  # Results and logs
```

## 🛠️ Troubleshooting

### Common Issues

**vLLM Server Not Starting**
```bash
# Check GPU memory
nvidia-smi
# Reduce max-model-len if needed
```

**GroundingDINO Compilation Errors**
```bash
# Reinstall with CUDA support
cd GroundingDINO
pip install -e .
```

**DAM Model Warnings**
- Torchvision warnings are normal and don't affect functionality
- Model path warnings can be ignored if inference works

### Performance Optimization
- Use `--max-num-batched-tokens` to control memory usage
- Adjust `num_questions` for testing vs. full evaluation
- Monitor GPU memory during processing

## 📄 Citation

If you use this framework, please cite the original SIRI paper:

```bibtex
@misc{wang2025topdownreasoningexplainablemultiagent,
      title={Towards Top-Down Reasoning: An Explainable Multi-Agent Approach for Visual Question Answering}, 
      author={Zeqing Wang and Wentao Wan and Qiqing Lao and Runmeng Chen and Minjie Lang and Xiao Wang and Keze Wang and Liang Lin},
      year={2025},
      eprint={2311.17331},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.17331}, 
}
```

## 📝 License

This project maintains the original licensing terms. See individual component licenses for details.

## 🤝 Contributing

See `docs/CONTRIBUTING.md` for contribution guidelines and development setup.