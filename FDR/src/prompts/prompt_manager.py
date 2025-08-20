"""
Core Prompt Management System for FDR
Academic research-ready template engine with intelligent features
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader, Template, meta
from jinja2.exceptions import TemplateError, TemplateSyntaxError


@dataclass
class TemplateMetadata:
    """Metadata for academic research tracking"""
    name: str
    version: str = "1.0.0"
    experiment: Optional[str] = None
    author: str = "FDR Research Team"
    paper_title: Optional[str] = None
    citation: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.now)
    performance_notes: Optional[str] = None
    usage_count: int = 0
    avg_render_time: float = 0.0


class PromptRegistry:
    """Central registry for template management and analytics"""
    
    def __init__(self):
        self.templates: Dict[str, TemplateMetadata] = {}
        self.usage_stats: Dict[str, List[float]] = {}  # Render times
        self.experiment_configs: Dict[str, Dict] = {}
        
    def register_template(self, metadata: TemplateMetadata):
        """Register template with metadata"""
        self.templates[metadata.name] = metadata
        self.usage_stats[metadata.name] = []
        logging.info(f"📋 Registered template: {metadata.name} v{metadata.version}")
    
    def record_usage(self, template_name: str, render_time: float):
        """Record template usage for analytics"""
        if template_name in self.templates:
            self.templates[template_name].usage_count += 1
            self.usage_stats[template_name].append(render_time)
            
            # Update average render time
            stats = self.usage_stats[template_name]
            self.templates[template_name].avg_render_time = sum(stats) / len(stats)
    
    def get_analytics(self, template_name: str) -> Dict[str, Any]:
        """Get analytics for academic reporting"""
        if template_name not in self.templates:
            return {}
            
        metadata = self.templates[template_name]
        stats = self.usage_stats[template_name]
        
        return {
            "template_name": template_name,
            "version": metadata.version,
            "usage_count": metadata.usage_count,
            "avg_render_time_ms": metadata.avg_render_time * 1000,
            "total_render_time_ms": sum(stats) * 1000,
            "experiment": metadata.experiment,
            "performance_notes": metadata.performance_notes
        }


class PromptManager:
    """
    Academic Research-Ready Prompt Management System
    
    Features:
    - Jinja2 template engine with inheritance
    - Academic metadata tracking
    - Hot-reload for development  
    - Template validation
    - Performance analytics
    - Experiment configuration support
    """
    
    def __init__(self, templates_dir: Optional[Path] = None, enable_hot_reload: bool = True):
        # Setup paths
        if templates_dir is None:
            templates_dir = Path(__file__).parent
        self.templates_dir = Path(templates_dir)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters for academic use
        self.env.filters['academic_format'] = self._academic_format_filter
        self.env.filters['citation_format'] = self._citation_format_filter
        
        # Management components
        self.registry = PromptRegistry()
        self.hot_reload = enable_hot_reload
        self.template_cache: Dict[str, Template] = {}
        self.file_mtimes: Dict[str, float] = {}
        
        # Load existing templates
        self._discover_templates()
        
        logging.info(f"🚀 PromptManager initialized with {len(self.registry.templates)} templates")
    
    def _discover_templates(self):
        """Discover and register all available templates"""
        template_files = list(self.templates_dir.rglob("*.jinja"))
        
        for template_file in template_files:
            rel_path = template_file.relative_to(self.templates_dir)
            template_name = str(rel_path)
            
            # Extract metadata from template if available
            metadata = self._extract_template_metadata(template_file)
            self.registry.register_template(metadata)
            
            # Cache file modification time
            self.file_mtimes[template_name] = template_file.stat().st_mtime
    
    def _extract_template_metadata(self, template_file: Path) -> TemplateMetadata:
        """Extract metadata from template file comments"""
        try:
            content = template_file.read_text(encoding='utf-8')
            
            # Default metadata
            metadata = TemplateMetadata(
                name=str(template_file.relative_to(self.templates_dir)),
                last_modified=datetime.fromtimestamp(template_file.stat().st_mtime)
            )
            
            # Parse metadata from comments (first 20 lines)
            lines = content.split('\n')[:20]
            for line in lines:
                line = line.strip()
                if line.startswith('{#') and ':' in line:
                    # Extract key-value pairs from Jinja comments
                    content_match = line.replace('{#', '').replace('#}', '').strip()
                    if ':' in content_match:
                        key, value = content_match.split(':', 1)
                        key, value = key.strip().lower(), value.strip()
                        
                        if key == 'version':
                            metadata.version = value
                        elif key == 'experiment':
                            metadata.experiment = value
                        elif key == 'author':
                            metadata.author = value
                        elif key == 'paper':
                            metadata.paper_title = value
                        elif key == 'citation':
                            metadata.citation = value
                        elif key == 'performance':
                            metadata.performance_notes = value
            
            return metadata
            
        except Exception as e:
            logging.warning(f"Could not extract metadata from {template_file}: {e}")
            return TemplateMetadata(
                name=str(template_file.relative_to(self.templates_dir))
            )
    
    def _check_template_updates(self, template_name: str) -> bool:
        """Check if template file has been updated (for hot-reload)"""
        if not self.hot_reload:
            return False
            
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            return False
            
        current_mtime = template_path.stat().st_mtime
        cached_mtime = self.file_mtimes.get(template_name, 0)
        
        if current_mtime > cached_mtime:
            self.file_mtimes[template_name] = current_mtime
            # Clear template cache to force reload
            if template_name in self.template_cache:
                del self.template_cache[template_name]
            logging.info(f"🔄 Hot-reload: {template_name} updated")
            return True
            
        return False
    
    def render(self, template_name: str, **kwargs) -> str:
        """
        Render template with variables and analytics tracking
        
        Args:
            template_name: Template file name (e.g., 'agents/strategist/fdr_strategist_hypothesis.jinja')
            **kwargs: Template variables
            
        Returns:
            Rendered template string
        """
        start_time = time.time()
        
        try:
            # Check for hot-reload updates
            self._check_template_updates(template_name)
            
            # Get or load template
            template = self._get_template(template_name)
            
            # Add academic metadata to template context
            if template_name in self.registry.templates:
                metadata = self.registry.templates[template_name]
                kwargs.update({
                    '_template_version': metadata.version,
                    '_experiment': metadata.experiment,
                    '_timestamp': datetime.now().isoformat()
                })
            
            # Render template
            result = template.render(**kwargs)
            
            # Record analytics
            render_time = time.time() - start_time
            self.registry.record_usage(template_name, render_time)
            
            logging.debug(f"✅ Rendered {template_name} in {render_time*1000:.2f}ms")
            return result
            
        except TemplateError as e:
            logging.error(f"❌ Template rendering failed for {template_name}: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error rendering {template_name}: {e}")
            raise
    
    def _get_template(self, template_name: str) -> Template:
        """Get template with caching"""
        if template_name not in self.template_cache:
            try:
                self.template_cache[template_name] = self.env.get_template(template_name)
            except TemplateError as e:
                logging.error(f"❌ Failed to load template {template_name}: {e}")
                raise
                
        return self.template_cache[template_name]
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate template syntax and variables"""
        try:
            template_path = self.templates_dir / template_name
            if not template_path.exists():
                return {"valid": False, "error": "Template file not found"}
            
            content = template_path.read_text(encoding='utf-8')
            
            # Parse template to check syntax
            ast = self.env.parse(content)
            
            # Extract variable names
            variables = meta.find_undeclared_variables(ast)
            
            return {
                "valid": True,
                "variables": list(variables),
                "template_name": template_name,
                "file_size": len(content),
                "line_count": len(content.split('\n'))
            }
            
        except TemplateSyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.message}",
                "line_number": e.lineno
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def list_templates(self, pattern: Optional[str] = None) -> List[str]:
        """List available templates, optionally filtered by pattern"""
        templates = list(self.registry.templates.keys())
        
        if pattern:
            templates = [t for t in templates if pattern in t]
            
        return sorted(templates)
    
    def get_template_analytics(self, template_name: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """Get analytics for academic reporting"""
        if template_name:
            return self.registry.get_analytics(template_name)
        else:
            return [self.registry.get_analytics(name) for name in self.registry.templates.keys()]
    
    def create_experiment_config(self, experiment_name: str, config: Dict[str, Any]):
        """Create configuration for experiment tracking"""
        self.registry.experiment_configs[experiment_name] = {
            "name": experiment_name,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "templates": []
        }
        logging.info(f"🧪 Created experiment config: {experiment_name}")
    
    def _academic_format_filter(self, text: str) -> str:
        """Custom Jinja2 filter for academic formatting"""
        # Add academic formatting logic (e.g., proper citations, formatting)
        return text.strip()
    
    def _citation_format_filter(self, citation: str) -> str:
        """Custom Jinja2 filter for citation formatting"""
        # Add citation formatting logic
        return f"[{citation}]" if citation else ""


# Global instance for easy import
prompt_manager = PromptManager() 