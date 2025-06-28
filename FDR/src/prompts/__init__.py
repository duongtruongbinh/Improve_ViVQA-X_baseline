"""
FDR Prompt Management System
Academic Research-Ready Jinja2 Template Engine

Supports:
- Template inheritance and composition
- Experiment configuration management  
- Version control and reproducibility
- A/B testing and analytics
- Hot-reload for development
"""

from .prompt_manager import PromptManager, PromptRegistry
from .template_validator import TemplateValidator
from .academic_metadata import AcademicMetadataManager

__version__ = "1.0.0"
__all__ = ["PromptManager", "PromptRegistry", "TemplateValidator", "AcademicMetadataManager"] 