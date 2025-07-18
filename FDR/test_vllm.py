#!/usr/bin/env python3
"""
Test vLLM Server Connection
Simple script to test if the vLLM server is running and responding correctly
"""

import requests
import json
import time
import yaml
from pathlib import Path

def load_config():
    """Load vLLM configuration"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["backend_config"]["vllm_settings"]

def test_health_check(base_url):
    """Test server health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_models_endpoint(base_url, api_key):
    """Test models listing endpoint"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("📋 Available models:")
            for model in models.get("data", []):
                print(f"  - {model.get('id', 'Unknown')}")
            return True
        return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_chat_completion(base_url, api_key, model_name):
    """Test chat completion with a simple VQA task"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test message
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": "What is the capital of Vietnam?"
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        print("🧪 Testing chat completion...")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Response: {content.strip()}")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 vLLM Server Connection Test")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        host = config.get("host", "localhost")
        port = config.get("port", 9100)
        api_key = config.get("api_key", "dummy-key")
        model_path = config.get("model_path", "")
        
        base_url = f"http://{host}:{port}/v1"
        model_name = Path(model_path).name if model_path else "Qwen2.5-VL-7B-Instruct"
        
        print(f"🌐 Server URL: {base_url}")
        print(f"🔑 API Key: {api_key}")
        print(f"🤖 Model: {model_name}")
        print()
        
        # Test 1: Health check
        print("1. Testing server health...")
        if test_health_check(base_url):
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
            print("💡 Make sure the vLLM server is running:")
            print(f"   ./launch_vllm.sh")
            return
        
        print()
        
        # Test 2: Models endpoint
        print("2. Testing models endpoint...")
        if test_models_endpoint(base_url, api_key):
            print("✅ Models endpoint working")
        else:
            print("❌ Models endpoint failed")
        
        print()
        
        # Test 3: Chat completion
        print("3. Testing chat completion...")
        if test_chat_completion(base_url, api_key, model_name):
            print("✅ Chat completion working")
        else:
            print("❌ Chat completion failed")
        
        print()
        print("🎉 vLLM server test completed!")
        
    except FileNotFoundError:
        print("❌ Config file not found")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
