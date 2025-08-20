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
    # logging.warning("Config loader not available")

try:
    from utils.vllm_api import VllmAPI
except ImportError:
    VllmAPI = None
    # logging.warning("vLLM API not available")

class BackendManager:
    """Enhanced backend manager supporting dual-model architecture for VQA-X"""

    def __init__(self, backend_type: str = "vllm", model_preference: str = "auto"):
        """
        Initialize backend manager with dual-model support

        Args:
            backend_type: "vllm" or "openai"
            model_preference: "vlm", "llm", or "auto" (smart routing)
        """
        self.backend_type = backend_type.lower()
        self.model_preference = model_preference
        self.client = None
        self.model = None
        self.vlm_client = None  # Vision-Language Model client
        self.llm_client = None  # Language Model client

        if self.backend_type == "vllm":
            self._setup_vllm()
        elif self.backend_type == "openai":
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}. Use 'vllm' or 'openai'")
    
    def _setup_vllm(self):
        """Setup vLLM backend with dual-model support"""
        try:
            # Setup VLM client (Vision-Language Model) - Port 9100
            self.vlm_client = OpenAI(
                api_key="dummy-key",
                base_url="http://localhost:9100/v1"
            )

            # Setup LLM client (Language Model) - Port 9200 (updated for current setup)
            self.llm_client = OpenAI(
                api_key="dummy-key",
                base_url="http://localhost:9200/v1"
            )

            # Set default client based on preference
            if self.model_preference == "vlm":
                self.client = self.vlm_client
                self.model = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
                logging.info(f"✅ Using VLM-only mode: {self.model}")
            elif self.model_preference == "llm":
                self.client = self.llm_client
                self.model = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct"
                logging.info(f"✅ Using LLM-only mode: {self.model}")
            else:  # auto mode
                self.client = self.vlm_client  # Default to VLM
                self.model = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
                logging.info(f"✅ Using dual-model mode with smart routing")

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
        # Try config-specified path first
        if app_config and "openai_settings" in app_config and "api_key_file" in app_config["openai_settings"]:
            config_path = Path(app_config["openai_settings"]["api_key_file"])
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        key = f.read().strip()
                        if key and key.startswith('sk-'):
                            logging.info(f"🔑 Loaded OpenAI API key from config path: {config_path}")
                            return key
                except Exception as e:
                    logging.warning(f"Failed to read API key file {config_path}: {e}")
        
        # Try loading from file - check multiple possible paths
        possible_paths = [
            Path("/home/huynq/VQA/FDR/openai_key.txt"),  # User specified path
            Path("openai_key.txt"),                      # Current directory
            Path("FDR/openai_key.txt"),            # From parent directory
            Path(__file__).parent.parent / "openai_key.txt"  # Relative to this file
        ]
        
        for key_file in possible_paths:
            if key_file.exists():
                try:
                    with open(key_file, 'r') as f:
                        key = f.read().strip()
                        if key and key.startswith('sk-'):
                            logging.info(f"🔑 Loaded OpenAI API key from {key_file}")
                            return key
                except Exception as e:
                    logging.warning(f"Failed to read API key file {key_file}: {e}")
        
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

    def _is_vision_task(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Determine if the task requires vision capabilities.
        Improved logic for better dual-model routing in VQA-X pipeline.
        """
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                # Check for image content in message
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        return True
            elif isinstance(content, str):
                content_lower = content.lower()

                # Strong indicators for text-only tasks (should go to LLM)
                text_only_indicators = [
                    'reasoning plan', 'hypothesis', 'mvkb', 'evidence_set', 'hypothesis_set',
                    'explanation generation', 'synthesis', 'logical reasoning', 'confidence',
                    'json output', 'structured reasoning', 'causal trace', 'conflict resolution'
                ]

                if any(indicator in content_lower for indicator in text_only_indicators):
                    return False

                # Strong indicators for vision tasks (should go to VLM)
                vision_indicators = [
                    'analyze the image', 'describe the image', 'what do you see',
                    'visual question answering', 'image analysis', 'caption generation',
                    'object detection', 'visual grounding', 'image_url', 'base64'
                ]

                if any(indicator in content_lower for indicator in vision_indicators):
                    return True

                # Weak vision keywords (only count if no text indicators present)
                weak_vision_keywords = ['image', 'visual', 'picture', 'photo']
                if any(keyword in content_lower for keyword in weak_vision_keywords):
                    # Check if it's in a reasoning context (should go to LLM)
                    reasoning_context = [
                        'given', 'based on', 'considering', 'analyze', 'reasoning',
                        'hypothesis', 'evidence', 'conclusion', 'therefore'
                    ]
                    if any(ctx in content_lower for ctx in reasoning_context):
                        return False
                    return True

        return False

    def get_optimal_client(self, messages: List[Dict[str, Any]]):
        """Get the optimal client based on task type with improved routing"""
        if self.model_preference != "auto":
            return self.client, self.model

        # Smart routing for dual-model architecture
        is_vision = self._is_vision_task(messages)

        # Log routing decision for debugging
        content_preview = str(messages[0].get('content', ''))[:100] if messages else 'No content'
        logging.debug(f"Task routing: {'VLM' if is_vision else 'LLM'} - Content: {content_preview}...")

        if is_vision:
            return self.vlm_client, "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
        else:
            return self.llm_client, "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct"
    
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

        # Get optimal client for this task (dual-model routing)
        optimal_client, optimal_model = self.get_optimal_client(messages)

        try:
            response = optimal_client.chat.completions.create(
                model=optimal_model,
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

def get_backend_manager(backend_type: str = "vllm", model_preference: str = "auto") -> BackendManager:
    """Get or create backend manager with model preference"""
    global _backend_manager
    if (_backend_manager is None or
        _backend_manager.backend_type != backend_type or
        _backend_manager.model_preference != model_preference):
        _backend_manager = BackendManager(backend_type, model_preference)
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