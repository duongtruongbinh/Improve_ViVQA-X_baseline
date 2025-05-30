import yaml
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

CONFIG_FILE_PATH = "../config.yaml"

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration structure and required fields"""
    required_sections = ["vlm_details", "inference_settings", "datasets"]
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")
    
    # Validate vlm_details
    vlm_details = config["vlm_details"]
    if "api_provider" not in vlm_details:
        raise ConfigValidationError("Missing api_provider in vlm_details")
    
    # Validate dataset paths if dataset_name is specified
    if "dataset_name" in config["datasets"]:
        dataset_name = config["datasets"]["dataset_name"]
        if dataset_name == "vqa-v2":
            required_paths = [
                "vqa_v2_val_images_dir",
                "vqa_v2_rest_val_questions_file",
                "vqa_v2_rest_val_annotations_file"
            ]
            for path_key in required_paths:
                if not config["datasets"].get(path_key):
                    raise ConfigValidationError(f"Missing required path for VQA-v2: {path_key}")

def resolve_env_vars(value: Any) -> Any:
    """Recursively resolve environment variables in config values"""
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value

def find_config_file(config_path: str) -> Path:
    """Find the configuration file using multiple search strategies"""
    # Try as absolute path
    if os.path.isabs(config_path):
        path = Path(config_path)
        if path.exists():
            return path
    
    # Try relative to script
    script_dir = Path(__file__).parent
    path = script_dir / config_path
    if path.exists():
        return path
    
    # Try relative to current working directory
    path = Path.cwd() / config_path
    if path.exists():
        return path
    
    # Try relative to workspace root
    workspace_root = Path(os.getenv("WORKSPACE_ROOT", os.getcwd()))
    path = workspace_root / config_path
    if path.exists():
        return path
    
    raise FileNotFoundError(
        f"Configuration file '{config_path}' not found in any of the following locations:\n"
        f"- Absolute path: {config_path}\n"
        f"- Relative to script: {script_dir / config_path}\n"
        f"- Relative to CWD: {Path.cwd() / config_path}\n"
        f"- Relative to workspace: {workspace_root / config_path}"
    )

def load_app_config(config_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    """Loads and validates the application configuration from a YAML file."""
    try:
        # Find the config file
        config_file = find_config_file(config_path)
        print(f"Loading configuration from: {config_file}")
        
        # Load and parse YAML
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            print("Warning: Configuration file is empty. Returning empty config.")
            return {}
            
        # Resolve environment variables
        config = resolve_env_vars(config)
        
        # Validate configuration
        validate_config(config)
        
        print("Configuration loaded and validated successfully.")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except ConfigValidationError as e:
        raise ValueError(f"Configuration validation error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading configuration: {e}")

# Load the configuration globally when this module is imported
try:
    app_config = load_app_config()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load app_config: {e}")
    print("Please ensure 'config.yaml' exists and is correctly formatted.")
    app_config = {}  # Initialize to empty dict to prevent downstream import errors

if not app_config:
    print("Warning: app_config is empty after loading attempt. Application might not function correctly.")
