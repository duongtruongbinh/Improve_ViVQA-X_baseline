# FDR Framework Documentation

Welcome to the FDR VQA Framework documentation! This directory contains comprehensive guides to help you understand, setup, and use the framework.

## 📚 Documentation Index

### 🚀 Getting Started
- **[SETUP.md](SETUP.md)** - Quick setup guide to get running in minutes
- **[README.md](../README.md)** - Main project overview and introduction

### 🏗️ Understanding the System
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed technical architecture and component design
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

### 🛠️ Development
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and development setup

## 🎯 Quick Navigation

### New Users
1. Start with the main [README.md](../README.md) for overview
2. Follow [SETUP.md](SETUP.md) for installation
3. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues

### Developers
1. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
3. Follow coding standards and testing procedures

### Researchers
1. Study the technical architecture in [ARCHITECTURE.md](ARCHITECTURE.md)
2. Understand the visualize-then-analyze pipeline
3. Explore the multi-agent reasoning system

## 🔧 Framework Overview

The FDR framework implements:
- **100% Local Processing**: No external API dependencies
- **Visual Object Detection**: GroundingDINO integration
- **Multi-Agent Reasoning**: Responder, Seeker, Integrator
- **Explainable AI**: Complete reasoning traces
- **Flexible Backends**: vLLM (default) and OpenAI support

## 📖 Additional Resources

### Configuration Files
- `configs/vivqa_config.yaml` - Dataset configuration
- `configs/vllm_test_config.yaml` - Test configuration
- `config.yaml` - vLLM server settings

### Example Usage
```bash
# Quick test
python main.py --config configs/vllm_test_config.yaml

# Full dataset
python main.py --backend vllm

# With OpenAI fallback
python main.py --backend openai
```

### Key Components
- `core/agents.py` - FDR agent implementations
- `core/pipeline.py` - Main pipeline orchestration
- `utils/vllm_client.py` - vLLM integration
- `utils/config_loader.py` - Configuration management

## 🚨 Important Notes

1. **GPU Requirements**: 16GB+ VRAM recommended for optimal performance
2. **Model Downloads**: GroundingDINO weights (~693MB) required
3. **vLLM Server**: Must be running before starting pipeline
4. **Dataset Paths**: Use absolute paths in configuration files

## 📞 Support

- **Issues**: Check troubleshooting guide first
- **Feature Requests**: See contributing guidelines
- **Questions**: Review architecture documentation
- **Setup Problems**: Follow setup guide step-by-step

## 🎉 Happy VQA!

The FDR framework enables sophisticated visual question answering with explainable AI. Explore the documentation to unlock its full potential! 