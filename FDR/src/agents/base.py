"""
Base module for MVKB-X agents
Contains common imports, helper functions, and base classes.
"""

import base64
import logging
import os
import json
import asyncio
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from retrying import retry

# Add GroundingDINO to Python path
groundingdino_path = Path(__file__).parent.parent.parent.parent / "GroundingDINO"
if groundingdino_path.exists():
    sys.path.insert(0, str(groundingdino_path))

def encode_image_to_base64(image_path: str) -> str | None:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found at {image_path}")
        return None

class BaseAgent:
    """Base class for all MVKB-X agents with common functionality."""
    
    def __init__(self, use_vllm: bool = True, model_name: str = None):
        self.use_vllm = use_vllm
        self.model_name = model_name or "gpt-4o-mini"
        self.client = None
        self.model = None
        self.temperature = 0.7
        self.max_tokens = 1000
        
    def _initialize_backend(self, backend_type: str = None):
        """Initialize backend manager"""
        if backend_type is None:
            backend_type = "vllm" if self.use_vllm else "openai"
            
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.backend_manager import get_backend_manager
            
            self.backend_manager = get_backend_manager(backend_type)
            self.client = self.backend_manager.client
            self.model = self.backend_manager.model
            
            logging.info(f"✅ {self.__class__.__name__} initialized with {backend_type} backend")
            
        except Exception as e:
            logging.error(f"Failed to initialize backend: {e}, using fallback")
            self.client = None
            self.model = self.model_name
            logging.warning("⚠️ No working backend available. Agent may not function properly.") 