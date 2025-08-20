#!/usr/bin/env python3
"""
Template Validation Tool for FDR Prompt System

Usage:
    python validate_templates.py                    # Validate all templates
    python validate_templates.py template.jinja     # Validate specific template
    python validate_templates.py --agent verifier   # Validate agent templates
    python validate_templates.py --check-performance # Check performance issues
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from prompt_manager import PromptManager

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

class TemplateValidator:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.errors = []
        self.warnings = []
        self.successes = []
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate a single template"""
        print_info(f"Validating: {template_name}")
        
        try:
            # Basic validation
            validation = self.prompt_manager.validate_template(template_name)
            
            if not validation['valid']:
                error_msg = f"{template_name}: {validation['error']}"
                self.errors.append(error_msg)
                print_error(error_msg)
                return validation
            
            # Check metadata
            self._check_metadata(template_name)
            
            # Check naming convention
            self._check_naming_convention(template_name)
            
            # Test render with sample data
            self._test_render(template_name, validation['variables'])
            
            # Performance check
            self._check_performance(template_name)
            
            success_msg = f"{template_name}: Valid"
            self.successes.append(success_msg)
            print_success(success_msg)
            
            return validation
            
        except Exception as e:
            error_msg = f"{template_name}: Unexpected error - {str(e)}"
            self.errors.append(error_msg)
            print_error(error_msg)
            return {"valid": False, "error": str(e)}
    
    def _check_metadata(self, template_name: str):
        """Check if template has proper metadata"""
        template_path = self.prompt_manager.templates_dir / template_name
        
        try:
            content = template_path.read_text(encoding='utf-8')
            lines = content.split('\n')[:10]  # Check first 10 lines
            
            required_metadata = ['version', 'author']
            found_metadata = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('{#') and ':' in line:
                    for meta in required_metadata:
                        if meta in line.lower():
                            found_metadata.append(meta)
            
            missing_metadata = set(required_metadata) - set(found_metadata)
            if missing_metadata:
                warning_msg = f"{template_name}: Missing metadata: {', '.join(missing_metadata)}"
                self.warnings.append(warning_msg)
                print_warning(warning_msg)
                
        except Exception as e:
            warning_msg = f"{template_name}: Could not check metadata - {str(e)}"
            self.warnings.append(warning_msg)
            print_warning(warning_msg)
    
    def _check_naming_convention(self, template_name: str):
        """Check if template follows naming convention"""
        if not template_name.endswith('.jinja'):
            warning_msg = f"{template_name}: Should use .jinja extension"
            self.warnings.append(warning_msg)
            print_warning(warning_msg)
        
        # Check for descriptive names
        if len(template_name.split('_')) < 3:
            warning_msg = f"{template_name}: Consider more descriptive name (component_agent_purpose.jinja)"
            self.warnings.append(warning_msg)
            print_warning(warning_msg)
    
    def _test_render(self, template_name: str, variables: List[str]):
        """Test template rendering with sample data"""
        try:
            # Create sample data for common variables
            sample_data = {
                'question': 'What color is the car?',
                'image_description': 'A red car on the street',
                'answer_candidates': ['red', 'blue', 'green'],
                'evidence': 'Visual analysis shows red coloring',
                'experiment': 'test_validation',
                'task': 'Visual question answering',
                'role': 'expert visual analyst'
            }
            
            # Add any missing variables with default values
            for var in variables:
                if var not in sample_data:
                    sample_data[var] = f"sample_{var}"
            
            # Try to render
            result = self.prompt_manager.render(template_name, **sample_data)
            
            if not result.strip():
                warning_msg = f"{template_name}: Renders to empty string"
                self.warnings.append(warning_msg)
                print_warning(warning_msg)
            
            # Check if output looks like JSON when expected
            if '```json' in result or '"' in result:
                try:
                    # Try to extract and parse JSON
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = result[json_start:json_end]
                        json.loads(json_str)
                        print_info(f"{template_name}: JSON output validated")
                except json.JSONDecodeError:
                    warning_msg = f"{template_name}: Invalid JSON in output"
                    self.warnings.append(warning_msg)
                    print_warning(warning_msg)
                    
        except Exception as e:
            warning_msg = f"{template_name}: Render test failed - {str(e)}"
            self.warnings.append(warning_msg)
            print_warning(warning_msg)
    
    def _check_performance(self, template_name: str):
        """Check template performance"""
        try:
            analytics = self.prompt_manager.get_template_analytics(template_name)
            
            if analytics.get('avg_render_time_ms', 0) > 100:
                warning_msg = f"{template_name}: Slow rendering ({analytics['avg_render_time_ms']:.2f}ms avg)"
                self.warnings.append(warning_msg)
                print_warning(warning_msg)
                
        except Exception:
            # No analytics available yet
            pass
    
    def validate_all_templates(self) -> Dict[str, Any]:
        """Validate all available templates"""
        print_header("Validating All Templates")
        
        templates = self.prompt_manager.list_templates()
        
        if not templates:
            print_error("No templates found!")
            return {"valid": False, "message": "No templates found"}
        
        print_info(f"Found {len(templates)} templates to validate")
        
        for template in templates:
            self.validate_template(template)
        
        return self._generate_report()
    
    def validate_agent_templates(self, agent_name: str) -> Dict[str, Any]:
        """Validate templates for specific agent"""
        print_header(f"Validating {agent_name.capitalize()} Agent Templates")
        
        templates = self.prompt_manager.list_templates(f"agents/{agent_name}")
        
        if not templates:
            print_error(f"No templates found for agent: {agent_name}")
            return {"valid": False, "message": f"No templates found for {agent_name}"}
        
        print_info(f"Found {len(templates)} templates for {agent_name}")
        
        for template in templates:
            self.validate_template(template)
        
        return self._generate_report()
    
    def check_performance_issues(self) -> Dict[str, Any]:
        """Check for performance issues across all templates"""
        print_header("Performance Analysis")
        
        templates = self.prompt_manager.list_templates()
        slow_templates = []
        
        for template in templates:
            try:
                analytics = self.prompt_manager.get_template_analytics(template)
                avg_time = analytics.get('avg_render_time_ms', 0)
                usage_count = analytics.get('usage_count', 0)
                
                if usage_count > 0:
                    if avg_time > 100:
                        slow_templates.append({
                            'template': template,
                            'avg_time_ms': avg_time,
                            'usage_count': usage_count
                        })
                        print_warning(f"{template}: {avg_time:.2f}ms avg ({usage_count} uses)")
                    else:
                        print_success(f"{template}: {avg_time:.2f}ms avg ({usage_count} uses)")
                else:
                    print_info(f"{template}: No usage data")
                    
            except Exception as e:
                print_warning(f"{template}: Could not get analytics - {str(e)}")
        
        if slow_templates:
            print_header("Slow Templates Summary")
            for template_info in slow_templates:
                print(f"- {template_info['template']}: {template_info['avg_time_ms']:.2f}ms")
        
        return {
            "slow_templates": slow_templates,
            "total_templates": len(templates)
        }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        print_header("Validation Report")
        
        total = len(self.successes) + len(self.errors)
        
        print(f"Total Templates: {total}")
        print_success(f"Valid: {len(self.successes)}")
        print_error(f"Errors: {len(self.errors)}")
        print_warning(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print_header("Errors")
            for error in self.errors:
                print(f"- {error}")
        
        if self.warnings:
            print_header("Warnings")
            for warning in self.warnings:
                print(f"- {warning}")
        
        success_rate = (len(self.successes) / total * 100) if total > 0 else 0
        print_header(f"Success Rate: {success_rate:.1f}%")
        
        return {
            "valid": len(self.errors) == 0,
            "total": total,
            "successes": len(self.successes),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "success_rate": success_rate
        }

def main():
    parser = argparse.ArgumentParser(description="Validate FDR Templates")
    parser.add_argument('template', nargs='?', help='Specific template to validate')
    parser.add_argument('--agent', help='Validate templates for specific agent')
    parser.add_argument('--check-performance', action='store_true', help='Check performance issues')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    validator = TemplateValidator()
    
    try:
        if args.check_performance:
            result = validator.check_performance_issues()
        elif args.agent:
            result = validator.validate_agent_templates(args.agent)
        elif args.template:
            result = validator.validate_template(args.template)
        else:
            result = validator.validate_all_templates()
        
        if args.json:
            print(json.dumps(result, indent=2))
        
        # Exit with error code if validation failed
        if not result.get('valid', True):
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_error("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 