# FDR Prompt Engineering Guide

## 📋 Tổng quan

Hướng dẫn toàn diện cho kỹ sư về cách viết, quản lý và tối ưu hóa prompts trong hệ thống FDR (Faithful Decomposed Reasoning).

## 🏗️ Kiến trúc Prompt System

### Cấu trúc thư mục
```
FDR/src/prompts/
├── prompt_manager.py          # Core engine
├── __init__.py               # Exports chính
├── agents/                   # Prompts theo agent
│   ├── verifier/            # VerifierAgent prompts
│   ├── strategist/          # StrategistAgent prompts  
│   ├── synthesizer/         # SynthesizerAgent prompts
│   └── explanation/         # ExplanationAgent prompts
├── commons/                 # Prompts dùng chung
├── experiments/             # Prompts cho thí nghiệm
├── configs/                # Cấu hình templates
└── versions/               # Version control cho prompts
```

### Các loại prompt files

1. **`.jinja` files** - Jinja2 templates với tính năng nâng cao
2. **`.prompts` files** - Raw prompts đơn giản (legacy)
3. **Config files** - Metadata và cấu hình

## 🚀 Quick Start

### 1. Import và khởi tạo
```python
from prompts import PromptManager

# Khởi tạo với hot-reload
prompt_manager = PromptManager(enable_hot_reload=True)
```

### 2. Render một template
```python
# Basic usage
result = prompt_manager.render(
    'agents/verifier/fdr_verifier_detection.jinja',
    question="What color is the car?",
    context="Image shows a red vehicle"
)

# Với metadata tracking
result = prompt_manager.render(
    'agents/strategist/hypothesis_formulation.jinja',
    question=question,
    answer_candidate=candidate,
    image_description=description,
    experiment="vqa_accuracy_v2"
)
```

## ✏️ Viết Templates

### Template structure chuẩn

```jinja
{# Template metadata #}
{# version: 2.1.0 #}
{# experiment: vqa_baseline #}
{# author: Research Team #}
{# paper: Faithful Decomposed Reasoning for VQA #}
{# performance: 89.5% accuracy on VQA-v2 #}

{# Main prompt content #}
You are an expert visual question answering system.

**Task:** {{ task_description }}

**Image Context:** 
{{ image_description }}

**Question:** {{ question }}

{% if answer_candidates %}
**Answer Candidates:**
{% for candidate in answer_candidates %}
{{ loop.index }}. {{ candidate }}
{% endfor %}
{% endif %}

**Instructions:**
- Analyze the image carefully
- Consider all visual details
- Provide reasoning for your analysis

{% if experiment %}
{# Academic tracking #}
**Experiment:** {{ experiment }}
**Template Version:** {{ _template_version }}
**Timestamp:** {{ _timestamp }}
{% endif %}

**Response Format:**
```json
{
    "analysis": "your detailed analysis",
    "answer": "final answer",
    "confidence": 0.95
}
```
```

### Best Practices

#### 1. **Template Naming Convention**
```
[component]_[agent]_[purpose].jinja

Examples:
- fdr_verifier_detection.jinja
- fdr_strategist_hypothesis.jinja
- fdr_synthesizer_final_answer.jinja
```

#### 2. **Metadata Headers** (Luôn bao gồm)
```jinja
{# version: x.y.z #}
{# experiment: experiment_name #}
{# author: Team/Person Name #}
{# paper: Research Paper Title #}
{# performance: Metrics if available #}
```

#### 3. **Variable Validation**
```python
# Validate template trước khi sử dụng
validation = prompt_manager.validate_template('agents/verifier/detection.jinja')
if not validation['valid']:
    logging.error(f"Template invalid: {validation['error']}")
```

#### 4. **Conditional Logic**
```jinja
{% if use_chain_of_thought %}
Think step by step:
1. First analyze {{ primary_focus }}
2. Then consider {{ secondary_aspects }}
3. Finally conclude your reasoning
{% endif %}

{% if detection_boxes %}
**Detected Objects:**
{% for box in detection_boxes %}
- {{ box.label }}: confidence {{ box.confidence }}
{% endfor %}
{% endif %}
```

## 🔧 Agent-Specific Patterns

### VerifierAgent Templates
```jinja
{# Pattern cho object detection #}
Transform this visual question into detection keywords for GroundingDINO.

**Question:** {{ question }}
**Output Format:** 
```json
{
    "detection_keywords": "object1 . object2 . object3"
}
```
```

### StrategistAgent Templates  
```jinja
{# Pattern cho hypothesis formulation #}
Given the visual evidence and question, formulate a hypothesis.

**Evidence:** {{ evidence }}
**Question:** {{ question }}
**Candidate Answer:** {{ answer_candidate }}

**Analysis Framework:**
1. Visual consistency check
2. Logical reasoning validation  
3. Confidence assessment

**Output:**
```json
{
    "hypothesis": "if-then statement",
    "reasoning": "step by step logic",
    "confidence": 0.0-1.0
}
```
```

### SynthesizerAgent Templates
```jinja
{# Pattern cho final synthesis #}
Synthesize findings from multiple analysis steps.

**Analysis Results:**
{% for result in analysis_results %}
- {{ result.source }}: {{ result.finding }}
{% endfor %}

**Final Decision Criteria:**
- Consistency across methods
- Confidence scores
- Evidence strength

**Output:**
```json
{
    "final_answer": "answer",
    "confidence": 0.95,
    "reasoning_path": ["step1", "step2", "step3"]
}
```
```

## 📊 Analytics & Monitoring

### 1. Template Performance Tracking
```python
# Xem analytics của template
analytics = prompt_manager.get_template_analytics('agents/verifier/detection.jinja')
print(f"Usage count: {analytics['usage_count']}")
print(f"Avg render time: {analytics['avg_render_time_ms']}ms")
```

### 2. Experiment Tracking  
```python
# Tạo experiment config
prompt_manager.create_experiment_config(
    'vqa_accuracy_v3',
    {
        'model': 'gpt-4o-mini',
        'temperature': 0.7,
        'focus': 'detection_accuracy'
    }
)
```

### 3. Hot-reload Development
```python
# Enable hot-reload cho development
prompt_manager = PromptManager(enable_hot_reload=True)

# Template sẽ tự động reload khi file thay đổi
# Không cần restart service
```

## 🎯 Common Use Cases

### 1. Thêm prompt mới cho agent
```bash
# 1. Tạo file template
touch FDR/src/prompts/agents/verifier/new_capability.jinja

# 2. Viết template với metadata
# 3. Test validation
# 4. Integrate vào agent code
```

### 2. A/B testing prompts
```python
# Version A
result_a = prompt_manager.render(
    'agents/strategist/hypothesis_v1.jinja',
    experiment="hypothesis_comparison_a",
    **params
)

# Version B  
result_b = prompt_manager.render(
    'agents/strategist/hypothesis_v2.jinja', 
    experiment="hypothesis_comparison_b",
    **params
)
```

### 3. Template inheritance
```jinja
{# base_vqa.jinja #}
You are a visual question answering expert.

**Instructions:**
{% block instructions %}
Default instructions here
{% endblock %}

**Output Format:**
{% block output_format %}
```json
{"answer": "response"}
```
{% endblock %}

{# specific_agent.jinja #}
{% extends "commons/base_vqa.jinja" %}

{% block instructions %}
Specific instructions for this agent
{{ super() }}
{% endblock %}
```

## 🛠️ Development Workflow

### 1. **Tạo prompt mới**
```bash
# Step 1: Tạo file
mkdir -p FDR/src/prompts/agents/new_agent/
touch FDR/src/prompts/agents/new_agent/capability.jinja

# Step 2: Viết template với metadata header
# Step 3: Validate syntax
# Step 4: Test với sample data
# Step 5: Integrate vào agent
```

### 2. **Testing templates**
```python
# Unit test cho template
def test_verifier_detection_template():
    result = prompt_manager.render(
        'agents/verifier/detection.jinja',
        question="What is the color of the car?"
    )
    
    assert "detection_keywords" in result
    assert result != ""
    
    # Validate JSON output
    import json
    parsed = json.loads(result)
    assert "detection_keywords" in parsed
```

### 3. **Version control**
```bash
# Backup prompt trước khi thay đổi major
cp agents/verifier/detection.jinja versions/detection_v1.2.0.jinja

# Update version trong metadata
# {# version: 1.3.0 #}
# {# changes: Added contextual detection #}
```

## 🚨 Troubleshooting

### Common Issues

1. **Template not found**
```python
# Check available templates
templates = prompt_manager.list_templates()
print("Available:", templates)

# Check specific pattern
detection_templates = prompt_manager.list_templates("detection")
```

2. **Render errors**
```python
# Validate before render
validation = prompt_manager.validate_template('template.jinja')
if not validation['valid']:
    print(f"Error: {validation['error']}")
    print(f"Line: {validation.get('line_number', 'unknown')}")
```

3. **Variable missing**
```python
# Check required variables
validation = prompt_manager.validate_template('template.jinja')
print("Required variables:", validation['variables'])
```

### Performance Issues

1. **Slow rendering**
```python
# Check analytics
analytics = prompt_manager.get_template_analytics('slow_template.jinja')
if analytics['avg_render_time_ms'] > 100:
    print("Template is slow, consider optimization")
```

2. **Memory usage**
```python
# Clear template cache nếu cần
prompt_manager.template_cache.clear()
```

## 📈 Advanced Features

### 1. Custom Filters
```python
# Thêm custom filter
@prompt_manager.env.filter('vqa_format')
def vqa_format_filter(text):
    """Format text cho VQA task"""
    return text.strip().lower().replace('?', '')
```

### 2. Template Preprocessing
```python
# Hook để preprocess template
def preprocess_template(template_content, variables):
    # Custom logic trước khi render
    return modified_content

prompt_manager.preprocess_hook = preprocess_template
```

### 3. Experiment Configs
```yaml
# experiments/vqa_accuracy_v3.yaml
name: "VQA Accuracy V3"
description: "Testing improved detection prompts"
templates:
  - "agents/verifier/detection_v3.jinja"
  - "agents/strategist/hypothesis_v3.jinja"
parameters:
  temperature: 0.7
  max_tokens: 1000
metrics:
  - accuracy
  - response_time
```

## 📚 Resources

### Documentation
- [Jinja2 Template Guide](https://jinja.palletsprojects.com/)
- [FDR Architecture Overview](./ARCHITECTURE.md)
- [Agent Development Guide](./AGENT_DEVELOPMENT.md)

### Tools
- **Template Validator**: `python -m prompts.validate template.jinja`
- **Analytics Dashboard**: `python -m prompts.analytics`
- **Migration Helper**: `python -m prompts.migrate old_prompt.txt new_template.jinja`

---

## 💡 Tips cho Kỹ sư

1. **Luôn include metadata header** cho tracking
2. **Test templates trước khi commit** với validate_template()
3. **Sử dụng hot-reload** trong development
4. **Monitor performance** qua analytics
5. **Version control** các changes quan trọng
6. **Document experiment configs** rõ ràng
7. **Use inheritance** để tránh duplicate code
8. **Validate JSON outputs** trong templates

---

*Guide version: 1.0 | Last updated: {{ datetime.now().strftime('%Y-%m-%d') }}* 