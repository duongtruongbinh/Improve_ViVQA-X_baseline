import yaml
import os
import sys

CONFIG_FILE_PATH = "config.yaml" # Assumes config.yaml is in the project root or same directory

def load_app_config(config_path=CONFIG_FILE_PATH):
    """Loads the application configuration from a YAML file."""
    # Determine the absolute path to the config file
    if not os.path.isabs(config_path):
        # Try relative to the script first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path_script = os.path.join(script_dir, config_path)
        # Then try relative to current working directory
        potential_path_cwd = os.path.join(os.getcwd(), config_path)

        if os.path.exists(potential_path_script):
            config_path = potential_path_script
        elif os.path.exists(potential_path_cwd):
            config_path = potential_path_cwd
        elif os.path.exists(config_path): # Check if relative path works directly (e.g., if run from root)
             pass # Use the provided relative path
        else:
            raise FileNotFoundError(
                f"Configuration file '{CONFIG_FILE_PATH}' not found at potential locations "
                f"(relative to script: '{potential_path_script}', relative to CWD: '{potential_path_cwd}', "
                f"or as provided: '{config_path}'). Please create config.yaml or ensure the path is correct."
            )
            
    abs_config_path = os.path.abspath(config_path)
    print(f"Attempting to load configuration from: {abs_config_path}")
    
    if not os.path.exists(abs_config_path):
         raise FileNotFoundError(f"Confirmed configuration file path does not exist: {abs_config_path}")

    with open(abs_config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            if config is None:
                print(f"Warning: Configuration file '{abs_config_path}' is empty. Returning empty config.")
                return {}
            print("Configuration loaded successfully.")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{abs_config_path}': {e}")
        except Exception as e_gen:
            raise ValueError(f"An unexpected error occurred while reading '{abs_config_path}': {e_gen}")

# Load the configuration globally when this module is imported.
try:
    app_config = load_app_config()
    if not isinstance(app_config, dict):
         print(f"Warning: Loaded configuration is not a dictionary (type: {type(app_config)}). Resetting to empty dict.")
         app_config = {}
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load app_config from '{CONFIG_FILE_PATH}'.")
    print(f"Error details: {e}")
    print("Please ensure 'config.yaml' exists, is correctly formatted, and paths within it are valid.")
    app_config = {} # Initialize to empty dict to prevent downstream import errors

if not app_config:
    print("Warning: app_config is empty after loading attempt. Application might not function correctly.")