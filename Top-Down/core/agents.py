# Top-Down/core/agents.py
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
groundingdino_path = Path(__file__).parent.parent.parent / "GroundingDINO"
if groundingdino_path.exists():
    sys.path.insert(0, str(groundingdino_path))

# --- Helper Functions ---

def encode_image_to_base64(image_path: str) -> str | None:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found at {image_path}")
        return None

# --- Agent Definitions ---

class ResponderAgent:
    """
    The Responder Agent, based on a VLM.
    Its main goal is to generate initial answer candidates and image captions.
    Enhanced with 100% local GroundingDINO and DAM integration for visualization pipeline.
    Now supports both OpenAI and local vLLM servers.
    """
    def __init__(self, client: OpenAI = None, model_name: str = None, temperature: float = None, max_tokens: int = None, use_vllm: bool = True, enable_dam: bool = True, groundingdino_docker: bool = False):
        # Initialize backend - simple choice between vLLM or OpenAI
        backend_type = "vllm" if use_vllm else "openai"
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.backend_manager import get_backend_manager
            
            self.backend_manager = get_backend_manager(backend_type)
            self.client = self.backend_manager.client
            self.model = self.backend_manager.model
            self.temperature = temperature or 0.7
            self.max_tokens = max_tokens or 1000
            
            logging.info(f"✅ ResponderAgent initialized with {backend_type} backend")
            
        except Exception as e:
            logging.error(f"Failed to initialize backend: {e}, using fallback")
            # Create a dummy client that will fail gracefully
            self.client = None
            self.model = model_name or "gpt-4o-mini"
            self.temperature = temperature or 0.7
            self.max_tokens = max_tokens or 1000
            logging.warning("⚠️ No working backend available. Agents will not function properly.")
        
        # Local components configuration
        self.groundingdino_enabled = False
        self.groundingdino_model = None
        self.image_storage_dir = None
        self.enable_dam = enable_dam
        self.dam = None
        self._force_groundingdino_docker = groundingdino_docker  # Config-driven Docker mode
        
        # Initialize local components
        self._initialize_groundingdino()
        self._initialize_dam()
    
    def _initialize_groundingdino(self):
        """Initialize GroundingDINO - Force Docker for production stability"""
        
        # Check if config forces Docker mode
        if hasattr(self, '_force_groundingdino_docker') and self._force_groundingdino_docker:
            logging.info("🐳 Config forces GroundingDINO Docker mode")
            self._setup_groundingdino_docker()
            return
        
        try:
            # Try native GroundingDINO installation first
            import supervision as sv
            sys.path.append(str(Path(__file__).parent.parent.parent / "GroundingDINO"))
            from groundingdino.util.inference import load_model, load_image, predict, annotate
            
            # Try loading model
            model_config_path = Path(__file__).parent.parent.parent / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            model_checkpoint_path = Path(__file__).parent.parent.parent / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
            
            if model_config_path.exists() and model_checkpoint_path.exists():
                # Use GPU 0 for GroundingDINO
                import torch
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(0)
                
                self.groundingdino_model = load_model(str(model_config_path), str(model_checkpoint_path), device="cuda:0")
                self.groundingdino_enabled = True
                self.groundingdino_docker = False
                logging.info("✅ GroundingDINO native installation loaded on GPU 0")
                
                # Restore original device
                torch.cuda.set_device(original_device)
                return
                
        except Exception as e:
            logging.warning(f"Native GroundingDINO failed ({str(e)[:50]}...), switching to Docker")
        
        # Fallback to Docker service (production mode)
        self._setup_groundingdino_docker()
    
    def _setup_groundingdino_docker(self):
        """Setup GroundingDINO Docker service"""
        try:
            import subprocess
            
            # Check if Docker is available and GroundingDINO image exists
            result = subprocess.run(["docker", "images", "-q", "groundingdino:latest"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                self.groundingdino_enabled = True
                self.groundingdino_docker = True
                self.groundingdino_model = None  # Docker service doesn't need model object
                logging.info("✅ GroundingDINO Docker service available")
                return
            else:
                logging.warning("GroundingDINO Docker image not found. Run: docker build -t groundingdino:latest GroundingDINO/")
                
        except Exception as e:
            logging.warning(f"GroundingDINO Docker check failed: {e}")
        
        # Both failed
        logging.warning("⚠️ GroundingDINO not available (neither native nor Docker)")
        self.groundingdino_enabled = False
        self.groundingdino_docker = False
        self.groundingdino_model = None
    
    def _initialize_dam(self):
        """Robust DAM initialization with multiple fallback strategies"""
        if not self.enable_dam:
            logging.info("🔄 DAM disabled")
            self.dam = None
            return
            
        import torch
        from transformers import AutoModel
        
        # Multiple strategies: GPU shared → CPU optimized
        strategies = [
            {
                "name": "GPU_SHARED",
                "device": "cuda:0",  # Share with GroundingDINO
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
                logging.info(f"🔥 Trying DAM strategy: {strategy['name']} on {strategy['device']}")
                
                # Clear GPU cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    available_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    logging.info(f"Available GPU memory: {available_memory / 1024**2:.1f} MB")
                
                # Load model with exact official pattern
                
                device = torch.device(strategy["device"])
                
                logging.info(f"Loading DAM model with dtype: {strategy['dtype_str']}")
                model = AutoModel.from_pretrained(
                    'nvidia/DAM-3B-Self-Contained',
                    trust_remote_code=True,
                    torch_dtype=strategy["dtype_str"]  # Use string format like official
                )
                
                # Force dtype consistency for CPU
                if strategy["device"] == "cpu":
                    logging.info("Converting all model weights to float32 for CPU compatibility")
                    model = model.float()  # Ensure all weights are float32
                
                # Move to device
                model = model.to(device)
                
                # Initialize DAM
                dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
                
                # Test inference to verify compatibility
                logging.info(f"Testing DAM inference on {strategy['device']}...")
                test_success = self._test_dam_inference(dam, device)
                
                if test_success:
                    self.dam = dam
                    self.dam_device = device
                    logging.info(f"✅ DAM successfully initialized with {strategy['name']} strategy")
                    return
                else:
                    logging.warning(f"❌ DAM test inference failed with {strategy['name']}")
                    
            except Exception as e:
                logging.warning(f"❌ DAM strategy {strategy['name']} failed: {str(e)[:100]}...")
                # Clean up failed attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # All strategies failed
        logging.error("❌ All DAM initialization strategies failed")
        self.dam = None
    
    def _test_dam_inference(self, dam, device):
        """Test DAM inference to verify functionality"""
        try:
            from PIL import Image
            import torch
            
            # Create minimal test inputs
            test_image = Image.new('RGB', (224, 224), color='red')
            test_mask = Image.new('L', (224, 224), 255)
            test_prompt = '<image>What color?'
            
            # Try inference with timeout
            with torch.cuda.device(device) if device.type == 'cuda' else torch.no_grad():
                tokens = []
                for i, token in enumerate(dam.get_description(
                    test_image, 
                    test_mask, 
                    test_prompt,
                    streaming=True, 
                    temperature=0.1, 
                    max_new_tokens=5
                )):
                    tokens.append(token)
                    if i >= 3:  # Limit test tokens
                        break
                
                result = ''.join(tokens).strip()
                logging.info(f"DAM test result: '{result[:20]}...'")
                return len(result) > 0  # Success if got any output
                
        except Exception as e:
            logging.warning(f"DAM test inference failed: {e}")
            return False
    
    def transform_question_to_detection_prompt(self, question: str) -> str:
        """Transform VQA question into object detection keywords for GroundingDINO"""
        prompt = f"""Transform this visual question into object detection keywords for GroundingDINO.
Extract only the nouns/objects to detect, separated by periods.
Avoid adjectives and focus on concrete objects that can be visually detected.
Use simple, common object names.

Question: "{question}"

Detection keywords (format: object1 . object2 . object3):"""
        
        if not self.client:
            logging.error("No working backend available for detection prompt")
            return "object . item ."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        detection_prompt = response.choices[0].message.content.strip()
        
        # Ensure proper formatting
        if not detection_prompt.endswith('.'):
            detection_prompt += ' .'
        
        logging.debug(f"Transformed '{question}' to detection prompt: '{detection_prompt}'")
        return detection_prompt
    
    def detect_and_visualize_with_groundingdino(self, image_path: str, detection_prompt: str) -> Optional[tuple]:
        """Run GroundingDINO detection - supports both native and Docker - returns (annotated_path, boxes_xyxy)"""
        if not self.groundingdino_enabled:
            logging.warning("GroundingDINO not available")
            return None
        
        try:
            if self.groundingdino_docker:
                # Use Docker service (only returns annotated image, no boxes)
                annotated_path = self._run_groundingdino_docker(image_path, detection_prompt)
                return (annotated_path, None) if annotated_path else None
            else:
                # Use native installation (returns both image and boxes)
                return self._run_groundingdino_native(image_path, detection_prompt)
                
        except Exception as e:
            logging.error(f"GroundingDINO detection failed: {e}")
            return None
    
    def _run_groundingdino_docker(self, image_path: str, detection_prompt: str) -> Optional[str]:
        """Run GroundingDINO using Docker service"""
        try:
            import subprocess
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy input image to temp directory
                temp_input = os.path.join(temp_dir, "input.jpg")
                temp_output = os.path.join(temp_dir, "output.jpg")
                shutil.copy2(image_path, temp_input)
                
                # Prepare Docker command
                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{temp_dir}:/workspace",
                    "--gpus", "all",
                    "groundingdino:latest",
                    "python", "-c", f"""
import sys
sys.path.append('/opt/program/GroundingDINO')
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

try:
    # Load model
    model = load_model('/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
                       '/opt/program/weights/groundingdino_swint_ogc.pth')
    
    # Load image
    image_source, image = load_image('/workspace/input.jpg')
    
    # Run detection
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption='{detection_prompt}',
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    # Annotate image
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
    # Save result
    cv2.imwrite('/workspace/output.jpg', annotated_frame)
    print('Detection completed successfully')
    
except Exception as e:
    print(f'Error: {{e}}')
"""
                ]
                
                # Run Docker command with fallback to CPU if GPU fails
                try:
                    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired:
                    # Try without GPU
                    docker_cmd[docker_cmd.index("--gpus")] = "--cpus"
                    docker_cmd[docker_cmd.index("all")] = "2"
                    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(temp_output):
                    # Copy result back to permanent location
                    output_dir = os.path.dirname(image_path)
                    output_filename = f"groundingdino_{os.path.basename(image_path)}"
                    final_output = os.path.join(output_dir, output_filename)
                    shutil.copy2(temp_output, final_output)
                    logging.info(f"✅ GroundingDINO Docker: {final_output}")
                    return final_output
                else:
                    logging.error(f"GroundingDINO Docker failed: {result.stderr}")
                    return None
                    
        except Exception as e:
            logging.error(f"GroundingDINO Docker service error: {e}")
            return None
    
    def _run_groundingdino_native(self, image_path: str, detection_prompt: str) -> Optional[tuple]:
        """Run GroundingDINO using native installation - returns (annotated_path, boxes_xyxy)"""
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent / "GroundingDINO"))
            from groundingdino.util.inference import load_image, predict, annotate
            from torchvision.ops import box_convert
            import cv2
            import torch
            
            # Ensure we're using GPU 0 for GroundingDINO
            with torch.cuda.device(0):
                # Load image
                image_source, image = load_image(image_path)
                
                # Run detection
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image,
                    caption=detection_prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                    device="cuda:0"
                )
                
                # Convert boxes to DAM format (absolute xyxy coordinates)
                boxes_xyxy = None
                if len(boxes) > 0:
                    h, w = image_source.shape[:2]
                    # Convert from normalized cxcywh to absolute xyxy for DAM
                    boxes_abs = boxes * torch.tensor([w, h, w, h])
                    boxes_xyxy = box_convert(boxes_abs, in_fmt='cxcywh', out_fmt='xyxy')
                    logging.debug(f"GroundingDINO detected {len(boxes_xyxy)} objects")
            
            # Annotate image
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            
            # Save annotated image
            output_dir = os.path.dirname(image_path)
            output_filename = f"groundingdino_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated_frame)
            
            logging.info(f"✅ GroundingDINO native: {output_path}")
            return (output_path, boxes_xyxy)
            
        except Exception as e:
            logging.error(f"GroundingDINO native error: {e}")
            return None
    
    def analyze_with_dam_and_boxes(self, image_path: str, question: str, boxes_xyxy=None) -> Dict[str, Any]:
        """Analyze image with DAM using detected bounding boxes for focused analysis"""
        if not self.dam:
            return {
                "answer_candidates": ["Error: DAM model not initialized"],
                "caption": "Error: Could not analyze image"
            }
            
        try:
            from PIL import Image
            import torch
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if boxes_xyxy is not None and len(boxes_xyxy) > 0:
                # Move boxes to CPU for processing
                if hasattr(boxes_xyxy, 'cpu'):
                    boxes_xyxy = boxes_xyxy.cpu()
                
                # Use first detected box for focused analysis
                box = boxes_xyxy[0].int().tolist()  # [x1, y1, x2, y2]
                
                # Create mask from bounding box (like DAM examples)
                mask = Image.new('L', image.size, 0)  # Black background
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.rectangle(box, fill=255)  # White rectangle for detected region
                
                logging.info(f"🎯 DAM analyzing focused region: {box}")
            else:
                # Fallback to full image analysis
                mask = Image.new('L', image.size, 255)  # White mask = full image
                logging.info("🔄 DAM analyzing full image (no boxes detected)")
            
            return self._analyze_with_nvidia_dam_focused(image, mask, question)
                
        except Exception as e:
            logging.error(f"DAM with boxes analysis failed: {e}")
            return {
                "answer_candidates": ["Error: DAM analysis failed"],
                "caption": f"Error: {str(e)}"
            }
    
    def _analyze_with_nvidia_dam_focused(self, image, mask, question: str) -> Dict[str, Any]:
        """Analyze with DAM using focused mask from GroundingDINO"""
        try:
            import torch
            
            # Ensure inputs are on the correct device and dtype
            if hasattr(self, 'dam_device') and hasattr(self, 'dam_dtype'):
                # Move computation to DAM's device context
                with torch.cuda.device(self.dam_device) if self.dam_device.type == 'cuda' else torch.no_grad():
                    
                    # Generate caption for masked region
                    caption_prompt = '<image>\nDescribe the highlighted/masked region in detail. Focus on objects, colors, actions, and spatial relationships.'
                    
                    caption_tokens = []
                    for token in self.dam.get_description(
                        image, 
                        mask,
                        caption_prompt,
                        streaming=True, 
                        temperature=0.2, 
                        top_p=0.5,
                        num_beams=1, 
                        max_new_tokens=256
                    ):
                        caption_tokens.append(token)
                    
                    caption_text = ''.join(caption_tokens).strip()
                    
                    # Generate VQA candidates focusing on detected objects
                    vqa_prompt = f'<image>\nFocus on the highlighted region. Question: {question.strip()}\n\nBased on the detected objects and their properties, provide 3 candidate answers:\nAnswers:'
                    
                    answer_tokens = []
                    for token in self.dam.get_description(
                        image, 
                        mask,
                        vqa_prompt,
                        streaming=True, 
                        temperature=0.3, 
                        top_p=0.6,
                        num_beams=1, 
                        max_new_tokens=128
                    ):
                        answer_tokens.append(token)
                    
                    answer_text = ''.join(answer_tokens).strip()
            else:
                # Fallback - original implementation
                # Generate caption for masked region
                caption_prompt = '<image>\nDescribe the highlighted/masked region in detail. Focus on objects, colors, actions, and spatial relationships.'
                
                caption_tokens = []
                for token in self.dam.get_description(
                    image, 
                    mask,
                    caption_prompt,
                    streaming=True, 
                    temperature=0.2, 
                    top_p=0.5,
                    num_beams=1, 
                    max_new_tokens=256
                ):
                    caption_tokens.append(token)
                
                caption_text = ''.join(caption_tokens).strip()
                
                # Generate VQA candidates focusing on detected objects
                vqa_prompt = f'<image>\nFocus on the highlighted region. Question: {question.strip()}\n\nBased on the detected objects and their properties, provide 3 candidate answers:\nAnswers:'
                
                answer_tokens = []
                for token in self.dam.get_description(
                    image, 
                    mask,
                    vqa_prompt,
                    streaming=True, 
                    temperature=0.3, 
                    top_p=0.6,
                    num_beams=1, 
                    max_new_tokens=128
                ):
                    answer_tokens.append(token)
                
                answer_text = ''.join(answer_tokens).strip()
            
            # Parse answers
            if ',' in answer_text:
                answer_candidates = [ans.strip() for ans in answer_text.split(',')[:3]]
            else:
                answer_candidates = [answer_text.strip()]
            
            # Ensure 3 candidates
            while len(answer_candidates) < 3:
                answer_candidates.append("uncertain")
            
            return {
                "answer_candidates": answer_candidates[:3],
                "caption": caption_text
            }
            
        except Exception as e:
            logging.error(f"DAM focused analysis failed: {e}")
            return {
                "answer_candidates": ["Error: DAM focused analysis failed"],
                "caption": f"Error: {str(e)}"
            }

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def generate_initial_response(self, question: str, image_path: str) -> dict:
        """
        REFACTORED: Image → VLM → GroundingDINO → DAM → Seeker
        Simple, focused workflow following user specification
        """
        logging.info(f"🔄 REFACTORED PIPELINE: {image_path}")
        
        try:
            # Step 1: VLM.process() - Get initial description
            logging.info("📝 Step 1: VLM.process()")
            vlm_description = self._get_vlm_description(image_path, question)
            
            # Step 2: GroundingDINO.generate(BBox) from VLM description
            logging.info("🎯 Step 2: GroundingDINO.generate(BBox)")
            annotated_image_path, boxes_xyxy = self._groundingdino_from_description(image_path, vlm_description, question)
            
            # Step 3: DAM.process() - Analyze with DAM using detected boxes
            if self.dam:
                logging.info("🔍 Step 3: DAM.process()")
                # Use original image for DAM analysis, but with bounding boxes for focus
                dam_results = self.analyze_with_dam_and_boxes(image_path, question, boxes_xyxy)
                
                if dam_results and dam_results.get("answer_candidates"):
                    logging.info("✅ Step 4: Seeker.receive() - Ready for MVKB")
                    return dam_results
            
            # Fallback: VLM-only analysis
            logging.info("🔄 DAM not available, using VLM fallback")
            return self._analyze_with_vlm_baseline(image_path, question)
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            return self._analyze_with_vlm_baseline(image_path, question)
    
    def _get_vlm_description(self, image_path: str, question: str) -> str:
        """Step 1: Use VLM to get initial description of the image"""
        base64_image = encode_image_to_base64(image_path)
        if not base64_image or not self.client:
            return "Error: Could not process image for description"
        
        description_prompt = f"""
Analyze this image and provide a detailed description that will help with object detection.

Focus on:
- Objects and their locations
- Colors, shapes, and sizes  
- Spatial relationships
- Any elements relevant to this question: "{question}"

Provide a clear, factual description in 2-3 sentences:
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": description_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=256,
        )
        
        description = response.choices[0].message.content.strip()
        logging.debug(f"VLM Description: {description}")
        return description
    
    def _groundingdino_from_description(self, image_path: str, description: str, question: str) -> tuple:
        """Step 2: Use description to guide GroundingDINO object detection - returns (annotated_path, boxes_xyxy)"""
        if not self.groundingdino_enabled:
            logging.warning("GroundingDINO not available for Step 2")
            return (None, None)
            
        # Create detection prompt from VLM description
        detection_keywords = self._extract_detection_keywords_from_description(description, question)
        
        # Use GroundingDINO with the extracted keywords
        result = self.detect_and_visualize_with_groundingdino(image_path, detection_keywords)
        
        if result:
            annotated_path, boxes_xyxy = result
            if annotated_path:
                logging.info(f"✅ GroundingDINO created annotated image: {annotated_path}")
            if boxes_xyxy is not None:
                logging.info(f"✅ GroundingDINO detected {len(boxes_xyxy)} bounding boxes")
            return (annotated_path, boxes_xyxy)
        else:
            logging.warning("GroundingDINO failed to create annotated image")
            return (None, None)
    
    def _extract_detection_keywords_from_description(self, description: str, question: str) -> str:
        """Extract object detection keywords from VLM description"""
        if not self.client:
            # Fallback: simple keyword extraction
            return self.transform_question_to_detection_prompt(question)
        
        prompt = f"""
Extract object detection keywords from this description and question for GroundingDINO.

Description: "{description}"
Question: "{question}"

Extract 3-5 key objects, colors, or attributes that should be detected and highlighted.
Return them as a single phrase separated by periods.

Example: "red car . traffic light . street sign . person"

Keywords:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        
        keywords = response.choices[0].message.content.strip()
        logging.debug(f"Detection keywords from description: {keywords}")
        return keywords
    
    def _analyze_with_vlm_baseline(self, image_path: str, question: str) -> Dict[str, Any]:
        """Baseline VLM analysis - works with both original and annotated images"""
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return {
                "answer_candidates": ["Error: Image not found."],
                "caption": "Error: Could not read the image file.",
            }

        # Enhanced prompt that works well with annotated images from GroundingDINO
        prompt = f"""
You are a Visual Question Answering assistant. Analyze the provided image carefully.

The image may contain visual annotations (bounding boxes, highlighted objects) that indicate important regions related to the question.

Your task:
1. Provide a detailed caption describing what you see in the image
2. Based on the image and any visual annotations, generate 3 most likely answers to the question

Question: "{question}"

Format your response as JSON:
{{
  "caption": "Detailed description of the image and any highlighted regions",
  "answer_candidates": ["answer1", "answer2", "answer3"]
}}
"""
        
        if not self.client:
            logging.error("No working backend available for VQA")
            return {
                "answer_candidates": ["Error: No backend available"],
                "caption": "Error: Cannot process image without working backend"
            }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        
        try:
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Log if we're using an annotated image
            if "annotated" in image_path.lower():
                logging.info("🎯 Using GroundingDINO-enhanced visual analysis")
            
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Responder: Failed to parse JSON response. Error: {e}. Response: {content}")
            return {
                "answer_candidates": ["Error: Invalid model response."],
                "caption": "Error: Could not parse the model's response.",
            }

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def answer_contextual_question(self, question: str, image_path: str, context_prompt: str) -> str:
        """
        Answers a question given an image and a rich context prompt (used by the Integrator).
        """
        logging.debug(f"Responder: Answering contextual question for image {image_path}")
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Error: Image not found."

        # The context_prompt is expected to contain the hypothesis and the original question.
        full_prompt = f"{context_prompt}\n\nBased on the image and the hypothesis, what is the final answer to the original question?"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                 {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0, # Low temperature for factual, direct answers
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

class SeekerAgent:
    """
    The Seeker Agent, based on an LLM.
    Its main goal is to generate a Multi-View Knowledge Base (MVKB)
    by creating relevant issues, forming hypotheses, and assigning confidence.
    Now supports both OpenAI and local vLLM servers.
    """
    def __init__(self, client: OpenAI = None, model_name: str = None, responder: ResponderAgent = None, use_vllm: bool = True):
        # Initialize backend - simple choice between vLLM or OpenAI
        backend_type = "vllm" if use_vllm else "openai"
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.backend_manager import get_backend_manager
            
            self.backend_manager = get_backend_manager(backend_type)
            self.client = self.backend_manager.client
            self.model = self.backend_manager.model
            
            logging.info(f"✅ SeekerAgent initialized with {backend_type} backend")
            
        except Exception as e:
            logging.error(f"Failed to initialize backend: {e}, using fallback")
            self.client = None
            self.model = model_name or "gpt-4o-mini"
        
        self.responder = responder

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _create_relevant_issues(self, question: str, answer_candidates: list, caption: str) -> list:
        """Generates clarifying questions (relevant issues) to differentiate answers."""
        prompt = f"""
You are a reasoning expert for a Visual Question Answering system.
Your goal is to generate a few clarifying questions (called "relevant issues") that, when answered, will help determine which of the candidate answers is correct.

Original Question: "{question}"
Image Caption: "{caption}"
Candidate Answers: {answer_candidates}

Generate a JSON list of 2-3 relevant issue questions. The questions should be specific and designed to be answered by looking at the image.
Example: If the answers are "sunny" or "cloudy", a good relevant issue is "What does the sky look like in the image?".

Output ONLY the JSON list of strings.
Example:
["Is there a clock visible in the image?", "What is the person's facial expression?"]
"""
        if not self.client:
            logging.error("No working backend available for Seeker")
            return []
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            # The model should return a json object with a key that contains the list
            data = json.loads(response.choices[0].message.content)
            # Find the list in the returned dict
            for key, value in data.items():
                if isinstance(value, list):
                    return value
            return [] # Return empty if no list found
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Seeker: Failed to parse relevant issues: {e}")
            return []

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _formulate_hypotheses_and_confidence(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> dict:
        """Forms a logical hypothesis and assigns a confidence score."""
        prompt = f"""
You are a logical reasoning module. Your task is to create a single logical hypothesis and assign a confidence score.

Context:
- Original Question: "{question}"
- A possible answer to the Original Question: "{answer_candidate}"
- A related sub-question (Relevant Issue): "{relevant_issue}"
- The answer to the Relevant Issue (based on the image): "{issue_answer}"

Task:
1.  Formulate a single, clear "IF-THEN" hypothesis that connects the sub-question's answer to the main answer.
2.  Based on common-sense reasoning, assign a confidence score from 0.0 (not confident) to 1.0 (very confident) that this hypothesis is logically sound.
3.  Convert the confidence score to a confidence word: <0.4 is "Unlikely", 0.4-0.7 is "Possible", >0.7 is "Likely".

Provide your response as a single JSON object with three keys: "hypothesis", "confidence_score" (a float), and "confidence_word" (a string).

Example:
{{
  "hypothesis": "IF the sky in the image is full of dark clouds, THEN the weather is likely rainy.",
  "confidence_score": 0.9,
  "confidence_word": "Likely"
}}
"""
        if not self.client:
            logging.error("No working backend available for hypothesis generation")
            return {}
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Seeker: Failed to parse hypothesis response: {e}")
            return {}

    def build_mvkv(self, question: str, image_path: str, answer_candidates: list, caption: str) -> list:
        """
        Builds the complete Multi-View Knowledge Base by orchestrating the Seeker's logic.
        """
        logging.info(f"Seeker: Building MVKB for question '{question}'")
        mvkv = []

        relevant_issues = self._create_relevant_issues(question, answer_candidates, caption)
        logging.debug(f"Seeker: Generated relevant issues: {relevant_issues}")

        for issue in relevant_issues:
            # Use the responder to get the answer for the sub-question
            issue_response = self.responder.generate_initial_response(issue, image_path)
            issue_answer = issue_response['answer_candidates'][0] # Take the top answer for the issue
            logging.debug(f"Seeker: Answer for issue '{issue}' is '{issue_answer}'")
            
            for candidate in answer_candidates:
                hypothesis_data = self._formulate_hypotheses_and_confidence(question, candidate, issue, issue_answer)
                if hypothesis_data:
                    mvkv_entry = {
                        "original_question": question,
                        "answer_candidate": candidate,
                        "relevant_issue": issue,
                        "issue_answer": issue_answer,
                        **hypothesis_data  # Unpack hypothesis, score, and word
                    }
                    mvkv.append(mvkv_entry)
        
        logging.info(f"Seeker: MVKB built with {len(mvkv)} entries.")
        return mvkv 

class IntegratorAgent:
    """
    The Integrator Agent, a logical mechanism.
    Its main goal is to conduct a weighted vote over the initial answer
    candidates based on the evidence from the MVKB.
    """
    def __init__(self, responder: ResponderAgent):
        self.responder = responder

    def conduct_weighted_voting(self, original_question: str, image_path: str, answer_candidates: list, mvkv: list) -> str:
        """
        Conducts the final weighted voting to determine the best answer.
        """
        logging.info("Integrator: Conducting weighted voting.")
        
        if not mvkv:
            logging.warning("Integrator: MVKB is empty. Returning the first answer candidate as a fallback.")
            return answer_candidates[0] if answer_candidates else "No answer could be determined."

        vote_scores = {candidate: 0.0 for candidate in answer_candidates}
        
        # This is the re-evaluation step from the paper
        for entry in mvkv:
            confidence_score = entry.get("confidence_score", 0.0)
            
            # Create the context-rich prompt for the Responder
            context_prompt = f"""
Hypothesis (Confidence: {entry.get('confidence_word', 'N/A')}): {entry.get('hypothesis', 'No hypothesis.')}
"""
            # Ask the responder to answer the original question again, but with this new context
            final_vote = self.responder.answer_contextual_question(
                question=original_question,
                image_path=image_path,
                context_prompt=context_prompt
            )
            logging.debug(f"Integrator: Vote received for '{final_vote}' with score {confidence_score}")

            # Add the confidence score to the answer that was voted for.
            # We check which of the original candidates the vote is closest to.
            for candidate in answer_candidates:
                if candidate.lower() in final_vote.lower():
                    vote_scores[candidate] += confidence_score
                    break # Stop after finding the first match

        logging.debug(f"Integrator: Final vote scores: {vote_scores}")

        # Find the candidate with the highest total score
        if not any(vote_scores.values()):
             logging.warning("Integrator: No votes were cast. Returning first candidate.")
             return answer_candidates[0]

        final_answer = max(vote_scores, key=vote_scores.get)
        logging.info(f"Integrator: Final answer chosen is '{final_answer}'")
        
        return final_answer 