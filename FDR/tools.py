"""
LangChain Tools for Vietnamese VQA Architecture
Wrap GroundingDINO và DAM thành LangChain Tools
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .schemas import TextPrompts, ImageWithBoxes, DetailedDescription

# Add GroundingDINO to path
groundingdino_path = Path(__file__).parent.parent / "GroundingDINO"
if groundingdino_path.exists():
    sys.path.insert(0, str(groundingdino_path))


class GroundingDINOInput(BaseModel):
    """Input schema cho GroundingDINO tool"""
    image_path: str = Field(description="Đường dẫn đến ảnh")
    text_prompts: List[str] = Field(description="Danh sách text prompts để phát hiện đối tượng")
    box_threshold: float = Field(default=0.3, description="Ngưỡng confidence cho bounding box")
    text_threshold: float = Field(default=0.25, description="Ngưỡng confidence cho text")


class DAMInput(BaseModel):
    """Input schema cho DAM tool"""
    image_path: str = Field(description="Đường dẫn đến ảnh")
    question: str = Field(description="Câu hỏi VQA")
    boxes_xyxy: Optional[List[List[float]]] = Field(default=None, description="Bounding boxes để focus")
    temperature: float = Field(default=0.2, description="Temperature cho generation")
    max_tokens: int = Field(default=256, description="Số token tối đa")


class GroundingDINOTool(BaseTool):
    """LangChain Tool cho GroundingDINO object detection"""
    
    name: str = "grounding_dino"
    description: str = """
    Phát hiện và định vị đối tượng trong ảnh dựa trên mô tả văn bản.
    Input: đường dẫn ảnh và danh sách text prompts
    Output: ảnh có annotation và tọa độ bounding boxes
    """
    args_schema = GroundingDINOInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.enabled = False
        self.docker_mode = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GroundingDINO model"""
        try:
            # Try native installation first
            import supervision as sv
            from groundingdino.util.inference import load_model, load_image, predict, annotate
            
            model_config_path = Path(__file__).parent.parent / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            model_checkpoint_path = Path(__file__).parent.parent / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
            
            if model_config_path.exists() and model_checkpoint_path.exists():
                with torch.cuda.device(0):
                    self.model = load_model(str(model_config_path), str(model_checkpoint_path), device="cuda:0")
                    self.enabled = True
                    self.docker_mode = False
                    logging.info("✅ GroundingDINO native model loaded")
                    return
                    
        except Exception as e:
            logging.warning(f"Native GroundingDINO failed: {e}")
        
        # Fallback to Docker
        try:
            import subprocess
            result = subprocess.run(["docker", "images", "-q", "groundingdino:latest"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                self.enabled = True
                self.docker_mode = True
                logging.info("✅ GroundingDINO Docker available")
                return
        except Exception as e:
            logging.warning(f"GroundingDINO Docker check failed: {e}")
        
        logging.error("❌ GroundingDINO not available")
        self.enabled = False
    
    def _run(self, image_path: str, text_prompts: List[str], box_threshold: float = 0.3, text_threshold: float = 0.25) -> ImageWithBoxes:
        """Run GroundingDINO detection"""
        if not self.enabled:
            raise ValueError("GroundingDINO tool not available")
        
        # Join prompts with periods
        detection_prompt = " . ".join(text_prompts) + " ."
        
        if self.docker_mode:
            return self._run_docker(image_path, detection_prompt, box_threshold, text_threshold)
        else:
            return self._run_native(image_path, detection_prompt, box_threshold, text_threshold)
    
    def _run_native(self, image_path: str, detection_prompt: str, box_threshold: float, text_threshold: float) -> ImageWithBoxes:
        """Run native GroundingDINO"""
        try:
            from groundingdino.util.inference import load_image, predict, annotate
            from torchvision.ops import box_convert
            import cv2
            
            with torch.cuda.device(0):
                # Load image
                image_source, image = load_image(image_path)
                
                # Run detection
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image,
                    caption=detection_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device="cuda:0"
                )
                
                # Convert boxes to absolute coordinates
                boxes_xyxy = None
                if len(boxes) > 0:
                    h, w = image_source.shape[:2]
                    boxes_abs = boxes * torch.tensor([w, h, w, h])
                    boxes_xyxy = box_convert(boxes_abs, in_fmt='cxcywh', out_fmt='xyxy').cpu().tolist()
                
                # Annotate image
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                
                # Save annotated image
                output_dir = os.path.dirname(image_path)
                output_filename = f"groundingdino_{os.path.basename(image_path)}"
                annotated_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(annotated_path, annotated_frame)
                
                return ImageWithBoxes(
                    image_path=image_path,
                    annotated_image_path=annotated_path,
                    boxes_xyxy=boxes_xyxy
                )
                
        except Exception as e:
            logging.error(f"GroundingDINO native error: {e}")
            raise
    
    def _run_docker(self, image_path: str, detection_prompt: str, box_threshold: float, text_threshold: float) -> ImageWithBoxes:
        """Run GroundingDINO via Docker"""
        # Implementation for Docker mode (similar to existing code)
        # Return ImageWithBoxes with at least annotated image path
        return ImageWithBoxes(
            image_path=image_path,
            annotated_image_path=None,  # Docker implementation would populate this
            boxes_xyxy=None
        )


class DAMTool(BaseTool):
    """LangChain Tool cho DAM (Describe Anything Model)"""
    
    name: str = "dam"
    description: str = """
    Tạo mô tả chi tiết cho các vùng cụ thể trong ảnh.
    Input: đường dẫn ảnh, câu hỏi, và optional bounding boxes
    Output: mô tả chi tiết và phân tích vùng
    """
    args_schema = DAMInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dam = None
        self.dam_device = None
        self.enabled = False
        self._initialize_dam()
    
    def _initialize_dam(self):
        """Initialize DAM model với multiple strategies"""
        import torch
        from transformers import AutoModel
        
        strategies = [
            {
                "name": "GPU_SHARED",
                "device": "cuda:0",
                "dtype": torch.float16,
                "dtype_str": "torch.float16"
            },
            {
                "name": "CPU_OPTIMIZED", 
                "device": "cpu",
                "dtype": torch.float32,
                "dtype_str": "torch.float32"
            }
        ]
        
        for strategy in strategies:
            try:
                logging.info(f"🔥 Trying DAM strategy: {strategy['name']}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                device = torch.device(strategy["device"])
                
                model = AutoModel.from_pretrained(
                    'nvidia/DAM-3B-Self-Contained',
                    trust_remote_code=True,
                    torch_dtype=strategy["dtype_str"]
                )
                
                if strategy["device"] == "cpu":
                    model = model.float()
                
                model = model.to(device)
                dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
                
                # Test inference
                if self._test_dam_inference(dam, device):
                    self.dam = dam
                    self.dam_device = device
                    self.enabled = True
                    logging.info(f"✅ DAM initialized with {strategy['name']}")
                    return
                    
            except Exception as e:
                logging.warning(f"❌ DAM strategy {strategy['name']} failed: {e}")
                continue
        
        logging.error("❌ All DAM strategies failed")
        self.enabled = False
    
    def _test_dam_inference(self, dam, device):
        """Test DAM inference"""
        try:
            from PIL import Image
            test_image = Image.new('RGB', (224, 224), color='red')
            test_mask = Image.new('L', (224, 224), 255)
            
            tokens = []
            for i, token in enumerate(dam.get_description(
                test_image, test_mask, '<image>Test',
                streaming=True, temperature=0.1, max_new_tokens=5
            )):
                tokens.append(token)
                if i >= 3:
                    break
            
            return len(''.join(tokens).strip()) > 0
        except Exception:
            return False
    
    def _run(self, image_path: str, question: str, boxes_xyxy: Optional[List[List[float]]] = None, 
             temperature: float = 0.2, max_tokens: int = 256) -> DetailedDescription:
        """Run DAM analysis"""
        if not self.enabled:
            raise ValueError("DAM tool not available")
        
        try:
            from PIL import Image, ImageDraw
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if boxes_xyxy and len(boxes_xyxy) > 0:
                # Use first detected box for focused analysis
                box = [int(x) for x in boxes_xyxy[0]]  # [x1, y1, x2, y2]
                
                # Create mask from bounding box
                mask = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle(box, fill=255)
                
                logging.info(f"🎯 DAM analyzing focused region: {box}")
            else:
                # Full image analysis
                mask = Image.new('L', image.size, 255)
                logging.info("🔄 DAM analyzing full image")
            
            # Generate description
            with torch.cuda.device(self.dam_device) if self.dam_device.type == 'cuda' else torch.no_grad():
                caption_prompt = '<image>\nDescribe this region in detail. Focus on objects, colors, actions, and spatial relationships.'
                
                tokens = []
                for token in self.dam.get_description(
                    image, mask, caption_prompt,
                    streaming=True, temperature=temperature, max_new_tokens=max_tokens
                ):
                    tokens.append(token)
                
                description = ''.join(tokens).strip()
                
                return DetailedDescription(
                    description=description,
                    region_analysis={"focused_region": bool(boxes_xyxy), "box_coordinates": boxes_xyxy}
                )
                
        except Exception as e:
            logging.error(f"DAM analysis failed: {e}")
            raise


# Export tools for use in LangGraph
def get_available_tools() -> List[BaseTool]:
    """Get list of available tools"""
    tools = []
    
    try:
        grounding_dino = GroundingDINOTool()
        if grounding_dino.enabled:
            tools.append(grounding_dino)
    except Exception as e:
        logging.warning(f"GroundingDINO tool init failed: {e}")
    
    try:
        dam = DAMTool()
        if dam.enabled:
            tools.append(dam)
    except Exception as e:
        logging.warning(f"DAM tool init failed: {e}")
    
    return tools 