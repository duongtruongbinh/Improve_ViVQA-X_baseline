# Project Overview: Top-Down Reasoning VQA Pipeline

## 1. Project Structure & Data Flow

### High-Level Layout
- **Top-Down/**: Main project root.
  - **app/**: Streamlit-based web app for VQA, captioning, search, etc.
  - **lavis/**: Core pipeline, models, datasets, tasks, runners, configs, and utilities (forked/extended from LAVIS).
  - **api_tools/**: LLM API integration toolkit.
  - **assets/**, **dataset_card/**, **prompts/**: Documentation, assets, and prompt templates.
  - **run_scripts/**: Shell scripts for training/evaluation.
  - **auto_metric_vqa_rad_rights_alloction.py**: Custom evaluation/post-processing for VQA results.
  - **README.md**: Project introduction and setup.

### Core Pipeline Flow
1. **Data Loading**: Dataset builders (in `lavis/datasets/builders/`) construct datasets using config files. Datasets inherit from `BaseDataset` and are processed by modular processors.
2. **Processing**: Processors (in `lavis/processors/`) handle image/text transformations, loaded dynamically via registry.
3. **Modeling**: Models (in `lavis/models/`) inherit from `BaseModel`, supporting checkpoint loading, optimizer param grouping, and distributed training.
4. **Task Logic**: Tasks (in `lavis/tasks/`) encapsulate high-level logic (e.g., VQA, captioning), including evaluation and result formatting.
5. **Runner**: Orchestration via runners (in `lavis/runners/`), supporting distributed training, evaluation, and checkpointing.
6. **Evaluation**: VQA evaluation logic in `lavis/common/vqa_tools/` and custom scripts (e.g., `auto_metric_vqa_rad_rights_alloction.py`).
7. **Web App**: Streamlit app (in `app/`) provides a user interface for demoing models.

## 2. Design Motifs & Coding Patterns
- **Registry Pattern**: Dynamic registration of models, tasks, processors, runners for extensibility.
- **Config-Driven**: Extensive use of YAML configs for datasets, models, and tasks.
- **Processor Abstraction**: Modular, composable data transformations.
- **Task Abstraction**: Each task encapsulates its own evaluation and data handling logic.
- **Distributed Training**: Built-in support for DDP and multi-GPU workflows.
- **Separation of Concerns**: Clear separation between data, model, task, and runner logic.
- **Post-Processing Hooks**: Custom scripts for advanced evaluation and result analysis.

## 3. Naming Conventions
- Classes: `CamelCase` (e.g., `BaseModel`, `VQATask`)
- Functions/Methods: `snake_case`
- Configs: YAML, named by dataset/model/task
- Processors: Named by model/task (e.g., `BlipImageEvalProcessor`)
- Registry keys: Lowercase, underscores (e.g., `blip_vqa`)

## 4. Detected Inconsistencies & Anti-Patterns
- Some scripts (e.g., `auto_metric_vqa_rad_rights_alloction.py`) use ad-hoc logic and lack modularization.
- Occasional code duplication (e.g., repeated accuracy calculation logic).
- Mixed use of relative and absolute paths; some hardcoded paths in scripts.
- Inconsistent error handling (asserts vs. exceptions).
- Some files lack docstrings or have minimal inline documentation.
- Legacy or experimental scripts are present in the root directory.

## 5. Implicit Dependencies
- Heavy reliance on LAVIS and its conventions, but with local modifications.
- Some scripts expect precomputed results or specific directory structures.
- Streamlit app expects certain model/task names and processor availability.

## 6. Recommendations
- Refactor custom scripts into modular pipeline components.
- Centralize path and config management.
- Increase code documentation and type hinting.
- Standardize error handling and logging.
- Consider unifying evaluation logic under the main pipeline. 