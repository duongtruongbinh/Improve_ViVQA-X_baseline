"""
Simple Backend Manager
Supports 2 options:
1. vLLM (local server)
2. OpenAI API (gpt-4o-mini)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI package not installed. Install with: pip install openai")

try:
    from utils.config_loader import app_config
except ImportError:
    app_config = None
    logging.warning("Config loader not available")

class BackendManager:
    """Simple backend manager for vLLM or OpenAI API"""
    
    def __init__(self, backend_type: str = "vllm"):
        """
        Initialize backend manager
        
        Args:
            backend_type: "vllm" or "openai"
        """
        self.backend_type = backend_type.lower()
        self.client = None
        self.model = None
        
        if self.backend_type == "vllm":
            self._setup_vllm()
        elif self.backend_type == "openai":
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}. Use 'vllm' or 'openai'")
    
    def _setup_vllm(self):
        """Setup vLLM backend"""
        try:
            if app_config and "vllm_details" in app_config:
                vllm_config = app_config["vllm_details"]
                self.client = OpenAI(
                    api_key=vllm_config["api_key"],
                    base_url=vllm_config["vlm_url"]
                )
                self.model = vllm_config["vlm_model_name"]
                logging.info(f"✅ Using vLLM: {self.model} at {vllm_config['vlm_url']}")
            else:
                # Fallback to default vLLM settings
                self.client = OpenAI(
                    api_key="dummy-key",
                    base_url="http://localhost:9100/v1"
                )
                self.model = "Qwen/Qwen2.5-VL-7B-Instruct"
                logging.info(f"✅ Using vLLM with default settings: {self.model}")
                
        except Exception as e:
            logging.error(f"❌ Failed to setup vLLM: {e}")
            raise
    
    def _setup_openai(self):
        """Setup OpenAI API backend"""
        # Try to load API key from multiple sources
        api_key = self._load_openai_key()
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Create Top-Down/openai_key.txt or set OPENAI_API_KEY environment variable")
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
            
            # Test the connection
            self.client.models.list()
            logging.info(f"✅ Using OpenAI API: {self.model}")
            
        except Exception as e:
            logging.error(f"❌ Failed to setup OpenAI API: {e}")
            raise
    
    def _load_openai_key(self) -> Optional[str]:
        """Load OpenAI API key from file or environment"""
        # Try loading from file first
        key_file = Path("Top-Down/openai_key.txt")
        if key_file.exists():
            try:
                with open(key_file, 'r') as f:
                    key = f.read().strip()
                    if key and key.startswith('sk-'):
                        logging.info("🔑 Loaded OpenAI API key from file")
                        return key
            except Exception as e:
                logging.warning(f"Failed to read API key file: {e}")
        
        # Try environment variable
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            logging.info("🔑 Loaded OpenAI API key from environment")
            return env_key
        
        logging.warning("⚠️ No OpenAI API key found")
        return None
    
    def is_available(self) -> bool:
        """Check if backend is available"""
        return self.client is not None
    
    def get_info(self) -> Dict[str, str]:
        """Get backend information"""
        return {
            "backend": self.backend_type,
            "model": self.model,
            "available": self.is_available()
        }
    
    def chat_completion(self, 
                       messages: List[Dict[str, Any]], 
                       max_tokens: int = 1000,
                       temperature: float = 0.7,
                       **kwargs) -> Optional[str]:
        """
        Make a chat completion request
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text or None if failed
        """
        if not self.is_available():
            logging.error(f"❌ {self.backend_type} backend not available")
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            # Log usage for OpenAI (vLLM doesn't have usage info)
            if self.backend_type == "openai" and hasattr(response, 'usage'):
                usage = response.usage
                logging.info(f"📊 OpenAI usage - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")
            
            return content
            
        except Exception as e:
            logging.error(f"❌ {self.backend_type} request failed: {e}")
            return None

# Global instance
_backend_manager = None

def get_backend_manager(backend_type: str = "vllm") -> BackendManager:
    """Get or create backend manager"""
    global _backend_manager
    if _backend_manager is None or _backend_manager.backend_type != backend_type:
        _backend_manager = BackendManager(backend_type)
    return _backend_manager

def switch_backend(backend_type: str):
    """Switch to a different backend"""
    global _backend_manager
    _backend_manager = BackendManager(backend_type)
    logging.info(f"🔄 Switched to {backend_type} backend")

def make_request(messages: List[Dict[str, Any]], backend_type: str = "vllm", **kwargs) -> Optional[str]:
    """Make a request with specified backend"""
    manager = get_backend_manager(backend_type)
    return manager.chat_completion(messages, **kwargs) 