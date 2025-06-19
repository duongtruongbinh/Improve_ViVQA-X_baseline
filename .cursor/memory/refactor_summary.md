# FDR-MA Refactor Summary

## Included Components
- **core/data_ingestion.py**: Minimal data loading utility
- **core/postprocessing.py**: Minimal result saving utility
- **core/model_openai.py**: OpenAI model integration (inference only)
- **core/model_vllm.py**: vLLM model integration (inference only)
- **configs/config.yaml**: Minimal YAML config for model/data
- **configs/settings.py**: (Optional) Python config
- **scripts/run_inference.py**: Main entry point for running pipeline
- **scripts/analyze_results.py**: Minimal result analysis script
- **scripts/prepare_data.py**: Minimal data cleaning/prep script
- **requirements.txt**: Only openai, vllm, tqdm, numpy, pyyaml
- **.env.example**: Example env vars for API keys/URLs
- **README.md**: Minimal usage and structure doc

## Excluded Components
- All UI/web/demo code (e.g., Streamlit app/, Gradio, Flask, etc.)
- All support for other LLM providers (HuggingFace, Anthropic, Gemini, etc.)
- All legacy, deprecated, or notebook files (e.g., old/, archive/, notebooks/)
- All monitoring, logging, or deployment-specific code (e.g., .github/, Docker/K8s configs not essential for local research)
- All non-essential helpers/utilities not directly supporting vLLM/OpenAI research pipeline
- All model code not related to vLLM/OpenAI (e.g., custom model classes, unrelated scripts)
- All dataset builders, runners, and task abstractions from LAVIS
- All configs unrelated to vLLM/OpenAI

## Rationale
- **Minimalism**: Only essential code for research with vLLM/OpenAI is retained.
- **Clarity**: Directory structure is flat and self-explanatory.
- **Focus**: All code directly supports data processing, model inference, or result analysis for vLLM/OpenAI only.

---
For any future extension, add new components only if they directly serve the research workflow and comply with the minimal/focused principles above. 