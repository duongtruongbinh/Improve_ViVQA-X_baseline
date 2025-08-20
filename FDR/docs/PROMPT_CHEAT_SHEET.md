# 🚀 FDR Prompt Engineering Cheat Sheet

## Quick Commands

```python
# Import
from prompts import PromptManager
pm = PromptManager(enable_hot_reload=True)

# Render template
result = pm.render('agents/verifier/detection.jinja', question="What color?")

# Validate template  
valid = pm.validate_template('template.jinja')

# List templates
templates = pm.list_templates()  # All
detection = pm.list_templates("detection")  # Filtered

# Analytics
stats = pm.get_template_analytics('template.jinja')
```

## Template Structure

```jinja
{# Metadata (Required) #}
{# version: 1.0.0 #}
{# experiment: test_name #} 
{# author: Your Name #}

You are {{ role }}.

**Task:** {{ task }}

{% if optional_param %}
**Optional:** {{ optional_param }}
{% endif %}

**Output:**
```json
{"answer": "{{ expected_format }}"}
```
```

## Naming Convention

```
[component]_[agent]_[purpose].jinja

✅ fdr_verifier_detection.jinja
✅ fdr_strategist_hypothesis.jinja  
❌ template1.jinja
❌ prompt.jinja
```

## Agent Patterns

### VerifierAgent
```jinja
Transform question to detection keywords.
**Question:** {{ question }}
**Output:** ```json
{"detection_keywords": "obj1 . obj2"}
```
```

### StrategistAgent
```jinja
Formulate hypothesis.
**Evidence:** {{ evidence }}
**Output:** ```json
{"hypothesis": "if X then Y", "confidence": 0.9}
```
```

### SynthesizerAgent
```jinja
Synthesize final answer.
**Results:** {{ results }}
**Output:** ```json
{"final_answer": "answer", "reasoning": ["step1"]}
```
```

## Common Variables

- `question` - VQA question
- `image_description` - Image context
- `answer_candidates` - Possible answers
- `evidence` - Analysis results
- `experiment` - Experiment name

## Troubleshooting

```python
# Template not found
pm.list_templates()

# Render error  
validation = pm.validate_template('template.jinja')
print(validation['error'])

# Missing variables
print(validation['variables'])

# Performance issue
stats = pm.get_template_analytics('slow_template.jinja')
print(f"Avg time: {stats['avg_render_time_ms']}ms")
```

## Best Practices

1. ✅ Always include metadata headers
2. ✅ Use descriptive template names  
3. ✅ Validate before deploy
4. ✅ Test with sample data
5. ✅ Monitor performance
6. ❌ Don't hardcode in agent code
7. ❌ Don't skip version control
8. ❌ Don't ignore validation errors

## File Locations

```
FDR/src/prompts/
├── agents/[agent_name]/     # Agent-specific prompts
├── commons/                 # Shared prompts  
├── experiments/             # Experiment prompts
└── versions/               # Backup versions
``` 