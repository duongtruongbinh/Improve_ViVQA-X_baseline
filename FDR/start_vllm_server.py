#!/usr/bin/env python3
"""
vLLM Server Launcher for FDR Multi-Agent Pipeline
Hosts Qwen2.5-VL-7B-Instruct model via vLLM for VQA tasks
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_model_exists(model_path):
    """Check if model path exists"""
    if not os.path.exists(model_path):
        logging.error(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Check for essential model files
    essential_files = ["config.json", "tokenizer.json"]
    for file in essential_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logging.warning(f"⚠️ Essential file missing: {file}")
    
    logging.info(f"✅ Model found at: {model_path}")
    return True

def start_vllm_server(config, args):
    """Start vLLM server with specified configuration"""
    vllm_config = config["backend_config"]["vllm_settings"]
    
    model_path = vllm_config["model_path"]
    if not check_model_exists(model_path):
        return False
    
    # Build vLLM command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", vllm_config.get("host", "0.0.0.0"),
        "--port", str(vllm_config.get("port", 9100)),
        "--gpu-memory-utilization", str(vllm_config.get("gpu_memory_utilization", 0.85)),
        "--max-model-len", str(vllm_config.get("max_model_len", 4096)),
        "--tensor-parallel-size", str(vllm_config.get("tensor_parallel_size", 1)),
        "--trust-remote-code",
        "--dtype", "auto",
        "--api-key", vllm_config.get("api_key", "dummy-key")
    ]
    
    # Add optional parameters
    if args.disable_log_stats:
        cmd.append("--disable-log-stats")
    
    if args.disable_log_requests:
        cmd.append("--disable-log-requests")
    
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    
    logging.info("🚀 Starting vLLM server...")
    logging.info(f"📍 Model: {model_path}")
    logging.info(f"🌐 Host: {vllm_config.get('host', '0.0.0.0')}:{vllm_config.get('port', 9100)}")
    logging.info(f"🔧 GPU Memory: {vllm_config.get('gpu_memory_utilization', 0.85)}")
    logging.info(f"📏 Max Length: {vllm_config.get('max_model_len', 4096)}")
    
    print("\n" + "="*60)
    print("🔥 FDR vLLM Server Configuration")
    print("="*60)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"URL: http://{vllm_config.get('host', '0.0.0.0')}:{vllm_config.get('port', 9100)}/v1")
    print(f"API Key: {vllm_config.get('api_key', 'dummy-key')}")
    print("="*60)
    print("Server is starting... (this may take a few minutes)")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Start the server
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to start vLLM server: {e}")
        return False
    except KeyboardInterrupt:
        logging.info("🛑 Server stopped by user")
        return True

def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Start vLLM server for FDR Multi-Agent Pipeline"
    )
    
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable logging of stats"
    )
    
    parser.add_argument(
        "--disable-log-requests",
        action="store_true",
        help="Disable logging of requests"
    )
    
    parser.add_argument(
        "--served-model-name",
        type=str,
        help="Custom model name for the API"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Start server
        success = start_vllm_server(config, args)
        
        if success:
            logging.info("✅ vLLM server completed successfully")
        else:
            logging.error("❌ Failed to start vLLM server")
            sys.exit(1)
            
    except FileNotFoundError as e:
        logging.error(f"❌ Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
