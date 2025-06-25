# Contributing to FDR VQA Framework

Thank you for your interest in contributing to the FDR VQA Framework! This guide will help you get started with development and contributions.

## 🛠️ Development Setup

### Environment Preparation
```bash
# Clone and setup the repository
git clone <repository-url>
cd VQA

# Create development environment
conda env create -f Top-Down/VQA_env.yaml
conda activate VQA_env

# Install additional development dependencies
pip install pytest black flake8 pre-commit
```

### Model Dependencies
Ensure all required models are available:
```bash
# GroundingDINO weights (693MB)
ls GroundingDINO/weights/groundingdino_swint_ogc.pth

# DAM model (should be accessible)
ls DAM/
```

## 🏗️ Architecture Overview

### Core Components

1. **ResponderAgent** (`core/agents.py`)
   - Handles visual object detection with GroundingDINO
   - Integrates DAM for image analysis
   - Supports both vLLM and OpenAI backends

2. **SeekerAgent** (`core/agents.py`) 
   - Generates relevant sub-questions
   - Creates hypotheses and confidence scores
   - Builds Multi-View Knowledge Base (MVKB)

3. **IntegratorAgent** (`core/agents.py`)
   - Performs weighted voting
   - Aggregates results from multiple hypotheses

4. **Pipeline** (`core/pipeline.py`)
   - Orchestrates the entire FDR workflow
   - Manages data loading and result saving

### Configuration System
- `config.yaml`: vLLM server configuration
- `config_loader.py`: Configuration management
- `configs/*.yaml`: Dataset-specific configurations

## 🧪 Testing

### Running Tests
```bash
# Quick test with 5 questions
python Top-Down/main.py --config Top-Down/configs/vllm_test_config.yaml

# Test specific components
python -m pytest tests/ -v

# Test with different backends
python Top-Down/main.py --backend vllm
python Top-Down/main.py --backend openai
```

### Adding New Tests
Create test files in the `tests/` directory:
```python
def test_responder_agent():
    """Test ResponderAgent functionality"""
    # Your test code here
    pass
```

## 📝 Code Style

### Python Standards
- Follow PEP 8 style guidelines
- Use type hints where applicable
- Document functions with docstrings
- Maximum line length: 88 characters

### Code Formatting
```bash
# Format code with black
black Top-Down/

# Check style with flake8
flake8 Top-Down/

# Pre-commit hooks (recommended)
pre-commit install
```

### Example Function Documentation
```python
def process_question(
    self, 
    image_path: str, 
    question: str,
    use_visual_detection: bool = True
) -> Dict[str, Any]:
    """
    Process a VQA question with enhanced visual pipeline.
    
    Args:
        image_path: Path to the input image
        question: Question text to answer
        use_visual_detection: Whether to use GroundingDINO detection
        
    Returns:
        Dictionary containing answer and reasoning trace
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ModelError: If inference fails
    """
    # Implementation here
```

## 🔧 Adding New Features

### Adding New Agents
1. Create agent class inheriting from base agent pattern
2. Implement required methods: `process()`, `_format_prompt()`
3. Add configuration support in `config_loader.py`
4. Update pipeline to integrate new agent

### Adding New Models
1. Create model wrapper in appropriate directory
2. Add model configuration to config files
3. Implement fallback mechanisms
4. Update documentation

### Adding New Datasets
1. Create dataset loader in `data/` directory
2. Add dataset configuration template
3. Update `configs/*.yaml` examples
4. Document dataset format requirements

## 🐛 Bug Reports

### Before Reporting
- [ ] Check existing issues
- [ ] Test with latest code
- [ ] Verify environment setup
- [ ] Try with different configurations

### Issue Template
```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- CUDA: [e.g., 11.8]
- GPU: [e.g., RTX 3090]

**Configuration:**
- Backend: [vllm/openai]
- Dataset: [e.g., ViVQA]
- Model: [e.g., Qwen2.5-VL-7B]

**Bug Description:**
Clear description of the issue

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Logs/Error Messages:**
```
Include relevant error messages
```

**Additional Context:**
Any other relevant information
```

## 🚀 Feature Requests

### Request Template
- **Feature Description**: Clear description of proposed feature
- **Use Case**: Why this feature would be useful
- **Implementation Ideas**: Potential approach (optional)
- **Alternatives**: Other solutions considered

## 📋 Pull Request Process

### Before Submitting
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Run tests and style checks
7. Commit with clear messages

### PR Requirements
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changes are backward compatible (if applicable)
- [ ] Performance impact considered
- [ ] Security implications reviewed

### PR Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing performed

## Documentation
- [ ] Code documentation updated
- [ ] README updated (if needed)
- [ ] Configuration examples updated

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Breaking changes documented
```

### Review Process
1. Maintainer review
2. Automated testing
3. Integration testing
4. Merge approval

## 🏷️ Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Steps
1. Update version numbers
2. Update CHANGELOG.md
3. Create release tag
4. Build and test release
5. Publish release

## 📞 Communication

### Getting Help
- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security-related issues

### Community Guidelines
- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow code of conduct

## 🏆 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the FDR VQA Framework! 🎉 