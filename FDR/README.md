# Faithful Visual Question Answering with Decomposed Reasoning Multi-Agent System

**Paper- Faithful Visual Question Answering with decomposed reasoning Multi-Agent system**

## 🏗️ Architecture

FDR is a modular VQA pipeline with explainable reasoning capabilities using a multi-agent approach:

- **VerifierAgent**: VLM + GroundingDINO + DAM for visual verification and initial analysis
- **StrategistAgent**: LLM for MVKB construction and hypothesis generation  
- **SynthesizerAgent**: Algorithm 2 weighted voting mechanism for answer selection
- **ExplanationAgent**: Natural language explanation generation with causal reasoning
- **EvalModule**: Comprehensive evaluation metrics for both accuracy and explanation quality

## 📁 Directory Structure

```
FDR/
├── main.py                       # 🎯 Central Control Point (Entry Point)
├── src/                          # Core source code
│   ├── __init__.py              # Package exports and version info
│   ├── pipeline.py              # Main pipeline orchestration logic
│   ├── agents/                   # Modular agent implementations
│   │   ├── __init__.py          # Agent module exports
│   │   ├── base.py              # BaseAgent class with common functionality
│   │   ├── verifier.py          # VerifierAgent (VLM + GroundingDINO + DAM)
│   │   ├── strategist.py        # StrategistAgent (MVKB construction)
│   │   ├── synthesizer.py       # SynthesizerAgent (Algorithm 2 voting)
│   │   ├── explanation.py       # ExplanationAgent (explanation generation)
│   │   └── prompts/             # Agent-specific prompt templates
│   ├── eval/                    # Evaluation modules
│   │   ├── __init__.py          # Evaluation module exports
│   │   └── eval_module.py       # EvalModule class with comprehensive metrics
│   ├── utils/                   # Utility modules
│   │   └── backend_manager.py   # OpenAI/vLLM backend management
│   └── g_evaluator.py           # GEvaluator for automatic scoring
├── config.yaml                 # 📋 Unified Configuration File
├── output/                      # Default output directory
├── utils/                       # Additional utility modules
├── docs/                        # Documentation
├── openai_key.txt              # OpenAI API key file
└── VQA_env.yaml                # Conda environment file
```

## 🚀 Quick Start

### **Primary Usage (Recommended)**

```bash
# 🎯 Main entry point - Uses unified config.yaml automatically
python3 main.py

# Use OpenAI backend instead of vLLM
python3 main.py --backend openai

# Enable comprehensive evaluation
python3 main.py --evaluate

# Test mode with default samples (2)
python3 main.py --test

# Test mode with 10 samples
python3 main.py --test 10

# Combine multiple options
python3 main.py --backend openai --evaluate --test 5
```

### **Research API Usage**

```bash
# Direct pipeline import for research
python3 -c "from src.pipeline import run_fdr_pipeline; run_fdr_pipeline()"

# With custom parameters for experiments
python3 -c "from src.pipeline import run_fdr_pipeline; run_fdr_pipeline(use_vllm=False, enable_evaluation=True)"

# For research scripting
python3 -c "
from src.pipeline import run_fdr_pipeline
results = run_fdr_pipeline(enable_evaluation=True, override_samples=10)
print(f'Processed {len(results)} samples for research')
"
```

## 📋 Configuration

All configuration is centralized in `config.yaml` with support for:
- **Multiple datasets**: VQA-X (English), ViVQA-X (Vietnamese), and custom datasets
- **Agent settings**: Temperature, tokens, model parameters for all 4 agents
- **Backend choice**: vLLM vs OpenAI with automatic fallback
- **Processing options**: Batch size, sampling, error handling, caching
- **Evaluation metrics**: VQA accuracy, explanation quality, multi-language support

### Dataset Switching
Simply change the `active_dataset` value in `config.yaml`:
```yaml
active_dataset: "vqax"    # English VQA-X
active_dataset: "vivqax"  # Vietnamese ViVQA-X  
active_dataset: "custom"  # Your custom dataset
```

## 🎯 Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  --backend BACKEND   Choose 'vllm' or 'openai' (default: vllm)
  --test             Test mode with limited samples
  --evaluate         Enable comprehensive evaluation
  --samples N        Override number of samples to process

Note: Configuration is automatically loaded from config.yaml
```

## 🔧 Development

### Core Classes and Functions

#### Agent Classes (`src/agents/`)

**BaseAgent** (`base.py`):
```python
class BaseAgent:
    def __init__(self, use_vllm: bool = True, model_name: str = "gpt-4o-mini")
    def _initialize_backend(self, backend_type: str = None)
    # Common functionality for all agents
```

**VerifierAgent** (`verifier.py`):
```python
class VerifierAgent(BaseAgent):
    def generate_initial_response(self, question: str, image_path: str) -> dict
    def analyze_with_dam_and_boxes(self, image_path: str, question: str, boxes_xyxy) -> dict
    def detect_and_visualize_with_groundingdino(self, image_path: str, detection_prompt: str) -> tuple
    def answer_contextual_question(self, question: str, image_path: str, context_prompt: str) -> str
```

**StrategistAgent** (`strategist.py`):
```python
class StrategistAgent(BaseAgent):
    def generate_mvkb(self, question: str, answer_candidates: list, caption: str) -> list
    def _create_relevant_issues(self, question: str, answer_candidates: list, caption: str) -> list
    def _formulate_hypotheses_and_confidence(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> dict
```

**SynthesizerAgent** (`synthesizer.py`):
```python
class SynthesizerAgent:
    def conduct_weighted_voting(self, original_question: str, image_path: str, answer_candidates: list, mvkb: list) -> dict
```

**ExplanationAgent** (`explanation.py`):
```python
class ExplanationAgent(BaseAgent):
    def generate_explanation(self, question: str, final_answer: str, image_caption: str, key_hypothesis: str, confidence_score: float) -> str
    def _map_confidence_level(self, confidence_score: float) -> str
```

#### Evaluation Classes (`src/eval/`)

**EvalModule** (`eval_module.py`):
```python
class EvalModule:
    def evaluate_accuracy(self, predictions: list, ground_truths: list) -> dict
    def evaluate_explanations(self, explanations: list, criteria: str = "clarity") -> dict
    def comprehensive_evaluation(self, results: list) -> dict
```

**GEvaluator** (`g_evaluator.py`):
```python
class GEvaluator:
    def evaluate_single(self, prediction: str, ground_truth: str, explanation: str = None) -> dict
    def evaluate_batch(self, evaluation_data: list, criteria: str = "overall") -> dict
```

#### Pipeline Functions (`src/pipeline.py`)

```python
def run_fdr_pipeline(config_path: str, use_vllm: bool = True, enable_evaluation: bool = False, override_samples: int = None) -> list
def run_fdr_pipeline(*args, **kwargs)  # Legacy compatibility alias
def load_config(config_path: str) -> dict
def setup_agents(config: dict, use_vllm: bool) -> tuple
```

### Extension Guide

1. **New Agents**: 
   ```python
   from src.agents.base import BaseAgent
   
   class CustomAgent(BaseAgent):
       def __init__(self, use_vllm=True):
           super().__init__(use_vllm)
           
       def process(self, input_data):
           # Your custom logic
           return result
   ```

2. **Custom Evaluation Metrics**: 
   - Add methods to `EvalModule` class in `src/eval/eval_module.py`
   - Follow existing patterns for metric calculation

3. **Pipeline Modifications**: 
   - Edit `run_fdr_pipeline()` function in `src/pipeline.py`
   - Maintain agent interaction patterns

4. **New CLI Options**: 
   - Modify argument parser in `main.py`
   - Update configuration handling

## 📊 Output Format

Results include:
- Final answers with confidence scores
- Natural language explanations  
- Complete MVKB traces for debugging
- Evaluation metrics and accuracy scores

## 🛠️ Dependencies

### Core Requirements
- **Python**: 3.8+ (3.10 recommended)
- **PyTorch**: For DAM and GroundingDINO GPU acceleration
- **CUDA**: 12.x compatible for GPU support
- **OpenAI API**: For VLM backend (gpt-4o-mini)

### Key Libraries
- **transformers**: Model loading and inference
- **openai**: OpenAI API client
- **PyYAML**: Configuration file parsing
- **tqdm**: Progress bar visualization
- **retrying**: Robust API call handling
- **Pillow (PIL)**: Image processing
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations

### Optional Dependencies
- **vLLM**: Local LLM inference (alternative to OpenAI)
- **tensorboard**: Training visualization (if applicable)
- **jupyter**: Notebook support for development

### Environment Setup
```bash
# Create from environment file
conda env create -f VQA_env.yaml
conda activate VQA

# Or install manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers openai pyyaml tqdm retrying pillow opencv-python numpy
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended for GroundingDINO)
- **CPU**: Multi-core processor for parallel processing
- **RAM**: 8GB+ system memory
- **Storage**: 10GB+ for models and data

## 📄 Legacy Compatibility

The pipeline maintains backward compatibility:
- `run_fdr_pipeline()` → `run_fdr_pipeline()`
- Old config formats are automatically migrated
- All functions accessible via `src.pipeline` imports