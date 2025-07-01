"""
VerifierAgent for FDR Pipeline
The Verifier Agent (formerly ResponderAgent), based on a VLM.
Multi-role agent handling visual verification, object detection, and image analysis.
Enhanced with GroundingDINO and DAM integration for comprehensive visual understanding.
"""

import logging
import os
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from retrying import retry

from .base import BaseAgent, encode_image_to_base64

# Import the new prompt management system
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "prompts"))
    from prompt_manager import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
    logging.info("✅ PromptManager imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ PromptManager not available: {e}")
    PROMPT_MANAGER_AVAILABLE = False


class VerifierAgent(BaseAgent):
    """
    The Verifier Agent (formerly ResponderAgent), based on a VLM.
    Multi-role agent handling visual verification, object detection, and image analysis.
    Enhanced with GroundingDINO and DAM integration for comprehensive visual understanding.
    """
    
    def __init__(self, client: OpenAI = None, model_name: str = None, temperature: float = None, 
                 max_tokens: int = None, use_vllm: bool = True, enable_dam: bool = True, 
                 groundingdino_docker: bool = False):
        super().__init__(use_vllm, model_name)
        
        # Override with specific parameters
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 1000
        
        # Initialize backend
        self._initialize_backend()
        
        # Local components configuration
        self.groundingdino_enabled = False
        self.groundingdino_model = None
        self.image_storage_dir = None
        self.enable_dam = enable_dam
        self.dam = None
        self._force_groundingdino_docker = groundingdino_docker
        
        # Initialize local components
        self._initialize_groundingdino()
        self._initialize_dam()
        
        # Initialize prompt manager
        if PROMPT_MANAGER_AVAILABLE:
            try:
                prompts_dir = Path(__file__).parent.parent / "prompts"
                self.prompt_manager = PromptManager(prompts_dir, enable_hot_reload=True)
                self.use_templates = True
                logging.info("✅ Verifier: Template system initialized")
            except Exception as e:
                logging.warning(f"⚠️ Verifier: Template system fallback: {e}")
                self.prompt_manager = None
                self.use_templates = False
        else:
            self.prompt_manager = None
            self.use_templates = False
    
    def _initialize_groundingdino(self):
        """Initialize GroundingDINO with graceful fallback for import conflicts"""
        
        # Check if config forces Docker mode
        if hasattr(self, '_force_groundingdino_docker') and self._force_groundingdino_docker:
            logging.info("🐳 Config forces GroundingDINO Docker mode")
            self._setup_groundingdino_docker()
            return
        
        try:
            # Try native GroundingDINO installation first
            logging.info("🔍 Attempting to load GroundingDINO native installation...")
            
            # Test basic imports first to catch conflicts early
            import torch
            import torchvision
            from PIL import Image
            
            # Then try GroundingDINO specific imports
            import supervision as sv
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "GroundingDINO"))
            from groundingdino.util.inference import load_model, load_image, predict, annotate
            
            # Try loading model
            model_config_path = Path(__file__).parent.parent.parent.parent / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            model_checkpoint_path = Path(__file__).parent.parent.parent.parent / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
            
            if model_config_path.exists() and model_checkpoint_path.exists():
                # Use GPU 2 for GroundingDINO
                logging.info("Attempting to load GroundingDINO on GPU 2")
                device_str = "cuda:2"
                
                self.groundingdino_model = load_model(str(model_config_path), str(model_checkpoint_path), device=device_str)
                self.groundingdino_enabled = True
                self.groundingdino_docker = False
                logging.info(f"✅ GroundingDINO native installation loaded on {device_str}")
                
                return
            else:
                logging.warning("GroundingDINO model files not found. Checking paths...")
                logging.warning(f"Config path exists: {model_config_path.exists()}")
                logging.warning(f"Checkpoint path exists: {model_checkpoint_path.exists()}")
                
        except ImportError as e:
            logging.warning(f"GroundingDINO import failed due to dependency conflicts: {str(e)[:100]}...")
            logging.warning("This is often caused by torchvision/PIL version mismatches")
        except Exception as e:
            logging.warning(f"GroundingDINO native initialization failed: {str(e)[:100]}...")
        
        # Try Docker fallback
        logging.info("🔄 Trying GroundingDINO Docker fallback...")
        try:
            self._setup_groundingdino_docker()
            if self.groundingdino_enabled:
                return
        except Exception as e:
            logging.warning(f"GroundingDINO Docker fallback failed: {e}")
        
        # Complete fallback - disable GroundingDINO
        logging.warning("⚠️ GroundingDINO completely disabled - using VLM-only mode")
        self.groundingdino_enabled = False
        self.groundingdino_docker = False
        self.groundingdino_model = None
    
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
        
        # Define specific strategy for GPU 2 to avoid ambiguity
        strategy = {
            "name": "GPU_2_ONLY",
            "device": "cuda:2",
            "dtype": torch.float16,
            "dtype_str": "torch.float16"
        }
        
        try:
            logging.info(f"🔥 Forcing DAM strategy: {strategy['name']} on {strategy['device']}")
            
            # Clear target GPU cache before loading
            device = torch.device(strategy["device"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model with exact official pattern
            logging.info(f"Loading DAM model with dtype: {strategy['dtype_str']}")
            model = AutoModel.from_pretrained(
                'nvidia/DAM-3B-Self-Contained',
                trust_remote_code=True,
                torch_dtype=strategy["dtype_str"]
            )
            
            # Move to the specified device
            model = model.to(device)
            
            # Initialize DAM
            dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
            
            # Test inference to verify compatibility
            logging.info(f"Testing DAM inference on {strategy['device']}...")
            test_success = self._test_dam_inference(dam, device)
            
            if test_success:
                self.dam = dam
                self.dam_device = device
                self.dam_dtype = strategy["dtype"]
                logging.info(f"✅ DAM successfully initialized with {strategy['name']} strategy on {strategy['device']}")
                return
            else:
                logging.error(f"❌ DAM test inference failed with {strategy['name']}. DAM will be disabled.")
                self.dam = None
                
        except Exception as e:
            logging.error(f"❌ DAM strategy {strategy['name']} failed catastrophically: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        """Transform VQA question into object detection keywords for GroundingDINO using templates"""
        if not self.client:
            logging.error("No working backend available for detection prompt")
            return "object . item ."
        
        try:
            # Try template-based approach first
            if self.use_templates and self.prompt_manager:
                prompt = self.prompt_manager.render(
                    'agents/verifier/fdr_verifier_detection.jinja',
                    question=question
                )
                logging.debug("✅ Using template-based prompt for detection keywords")
            else:
                # Fallback to hardcoded prompt
                prompt = f"""Transform this visual question into object detection keywords for GroundingDINO.
Extract only the nouns/objects to detect, separated by periods.
Avoid adjectives and focus on concrete objects that can be visually detected.
Use simple, common object names.

Question: "{question}"

Detection keywords (format: object1 . object2 . object3):"""
                logging.debug("⚠️ Using fallback hardcoded prompt for detection keywords")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response if using templates
            if self.use_templates and self.prompt_manager and content.startswith('{'):
                import json
                try:
                    parsed = json.loads(content)
                    detection_prompt = parsed.get('detection_keywords', 'object . item .')
                    logging.debug("✅ Successfully parsed JSON response from detection template")
                except json.JSONDecodeError:
                    logging.warning("⚠️ Failed to parse JSON, using raw content")
                    detection_prompt = content
            else:
                # Use raw content for fallback mode
                detection_prompt = content
            
            # Ensure proper formatting
            if not detection_prompt.endswith('.'):
                detection_prompt += ' .'
            
            logging.debug(f"Transformed '{question}' to detection prompt: '{detection_prompt}'")
            return detection_prompt
            
        except Exception as e:
            logging.error(f"Detection prompt generation failed: {e}")
            return "object . item ."
    
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
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "GroundingDINO"))
            from groundingdino.util.inference import load_image, predict, annotate
            from torchvision.ops import box_convert
            import cv2
            import torch
            
            # Ensure we're using GPU 2 for GroundingDINO
            with torch.cuda.device(2):
                # Load image
                image_source, image = load_image(image_path)
                
                # Run detection
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image,
                    caption=detection_prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                    device="cuda:2"
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
            
            # Use DAM for detailed description + VLM for answer candidates (parallel approach)
            detailed_caption = self._get_dam_detailed_description(image, mask, question)
            
            # VLM generates short answer candidates in parallel
            answer_candidates = self._get_vlm_short_answers(image_path, question)
            
            return {
                "answer_candidates": answer_candidates,
                "caption": detailed_caption
            }
                
        except Exception as e:
            logging.error(f"DAM with boxes analysis failed: {e}")
            return {
                "answer_candidates": ["Error: DAM analysis failed"],
                "caption": f"Error: {str(e)}"
            }

    def _get_dam_detailed_description(self, image, mask, question: str) -> str:
        """Use DAM to generate detailed description for MVKB construction"""
        try:
            import torch
            
            if hasattr(self, 'dam_device') and hasattr(self, 'dam_dtype'):
                with torch.cuda.device(self.dam_device) if self.dam_device.type == 'cuda' else torch.no_grad():
                    
                    # Detailed description prompt for MVKB construction
                    description_prompt = f'<image>\nProvide a comprehensive description of this image, focusing on details that would help answer: "{question}". Include:\n1. All visible objects and their properties\n2. Spatial relationships and positioning\n3. Colors, textures, and specific details\n4. Any relevant context or setting information\n\nDescription:'
                    
                    description_tokens = []
                    for token in self.dam.get_description(
                        image, 
                        mask,
                        description_prompt,
                        streaming=True, 
                        temperature=0.2, 
                        top_p=0.5,
                        num_beams=1, 
                        max_new_tokens=400  # Allow longer for detailed description
                    ):
                        description_tokens.append(token)
                        if len(description_tokens) >= 400:
                            break
                    
                    detailed_description = ''.join(description_tokens).strip()
                    
                    # Clean up description
                    if 'Description:' in detailed_description:
                        detailed_description = detailed_description.split('Description:')[-1].strip()
                    
                    logging.info(f"📝 DAM detailed description (first 100 chars): {detailed_description[:100]}...")
                    return detailed_description
                    
        except Exception as e:
            logging.error(f"DAM detailed description failed: {e}")
            return f"Error generating detailed description: {str(e)}"
        
        return "Error: DAM description unavailable"

    def _get_vlm_short_answers(self, image_path: str, question: str) -> list:
        """Use VLM to generate short answer candidates"""
        if not self.client:
            logging.warning("VLM not available for short answers, using fallback")
            return ["unknown"]
        
        try:
            image_b64 = encode_image_to_base64(image_path)
            if not image_b64:
                return ["error"]
            
            # Create appropriate prompt based on question type
            vqa_prompt = self._create_short_answer_prompt(question)
            
            # Generate short answer candidates using VLM
            candidates = self._generate_short_answer_candidates(image_b64, vqa_prompt, question)
            
            logging.info(f"🎯 VLM short answer candidates: {candidates}")
            return candidates
            
        except Exception as e:
            logging.error(f"VLM short answer generation failed: {e}")
            return ["error"]
    
    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def generate_initial_response(self, question: str, image_path: str) -> dict:
        """
        Generate initial response with answer candidates and image caption.
        Enhanced with GroundingDINO + DAM pipeline for focused analysis.
        """
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return {
                "answer_candidates": ["Error: Image not found"],
                "caption": "Error: Image file not accessible"
            }
        
        logging.info(f"🔍 VerifierAgent analyzing: {question}")
        
        try:
            # Step 1: Get VLM description and detection prompt
            description = self._get_vlm_description(image_path, question)
            
            # Step 2: Run GroundingDINO detection based on description
            groundingdino_result = self._groundingdino_from_description(image_path, description, question)
            
            # Step 3: Enhanced analysis with DAM using detected boxes
            if groundingdino_result and groundingdino_result[1] is not None:
                # GroundingDINO + DAM pipeline (best performance)
                annotated_path, boxes_xyxy = groundingdino_result
                result = self.analyze_with_dam_and_boxes(image_path, question, boxes_xyxy)
                logging.info("✅ GroundingDINO + DAM pipeline completed")
            else:
                # Fallback to VLM baseline
                logging.info("🔄 Falling back to VLM baseline analysis")
                result = self._analyze_with_vlm_baseline(image_path, question)
            
            # Add description context to result
            if description and "caption" in result:
                result["caption"] = f"{description}. {result['caption']}"
            
            return result
            
        except Exception as e:
            logging.error(f"VerifierAgent generation failed: {e}")
            return {
                "answer_candidates": ["Error: Analysis failed"],
                "caption": f"Error during image analysis: {str(e)}"
            }
    
    def _get_vlm_description(self, image_path: str, question: str) -> str:
        """Get VLM description for GroundingDINO prompt generation"""
        if not self.client:
            return "Unable to analyze image due to backend unavailability"
        
        try:
            image_b64 = encode_image_to_base64(image_path)
            if not image_b64:
                return "Error: Could not encode image"
            
            prompt = f"""Analyze this image and provide a concise description focusing on:
1. Main objects and their locations
2. Key visual elements relevant to the question: "{question}"
3. Important details that could help answer the question

Keep the description under 100 words and focus on observable facts."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }],
                temperature=0.3,
                max_tokens=200
            )
            
            description = response.choices[0].message.content.strip()
            logging.debug(f"VLM description: {description[:100]}...")
            return description
            
        except Exception as e:
            logging.error(f"VLM description failed: {e}")
            return "Error generating image description"
    
    def _groundingdino_from_description(self, image_path: str, description: str, question: str) -> tuple:
        """Run GroundingDINO based on VLM description"""
        if not self.groundingdino_enabled:
            return None
        
        try:
            # Extract detection keywords from description
            detection_prompt = self._extract_detection_keywords_from_description(description, question)
            
            # Run GroundingDINO with extracted keywords
            return self.detect_and_visualize_with_groundingdino(image_path, detection_prompt)
            
        except Exception as e:
            logging.error(f"GroundingDINO from description failed: {e}")
            return None
    
    def _extract_detection_keywords_from_description(self, description: str, question: str) -> str:
        """Extract object detection keywords from VLM description"""
        if not self.client:
            return "object . item ."
        
        try:
            prompt = f"""Extract object detection keywords from this image description for GroundingDINO.
Focus on concrete, detectable objects mentioned in the description that are relevant to the question.

Description: "{description}"
Question: "{question}"

Extract 2-4 key objects/nouns, separated by periods (format: object1 . object2 . object3):"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            
            keywords = response.choices[0].message.content.strip()
            
            # Ensure proper formatting
            if not keywords.endswith('.'):
                keywords += ' .'
            
            logging.debug(f"Extracted keywords: {keywords}")
            return keywords
            
        except Exception as e:
            logging.error(f"Keyword extraction failed: {e}")
            return "object . item ."
    
    def _analyze_with_vlm_baseline(self, image_path: str, question: str) -> Dict[str, Any]:
        """Fallback VLM baseline analysis when GroundingDINO/DAM unavailable"""
        if not self.client:
            return {
                "answer_candidates": ["Error: No backend available"],
                "caption": "Error: Backend unavailable"
            }
        
        try:
            image_b64 = encode_image_to_base64(image_path)
            if not image_b64:
                return {
                    "answer_candidates": ["Error: Could not process image"],
                    "caption": "Error: Image encoding failed"
                }
            
            # Generate VLM analysis using templates
            if self.use_templates and self.prompt_manager:
                vlm_prompt = self.prompt_manager.render(
                    'agents/verifier/fdr_verifier_vlm_analysis.jinja',
                    question=question,
                    context_prompt=None
                )
                logging.debug("✅ Using template-based VLM analysis prompt")
                
                # Single VLM call with comprehensive template
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vlm_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }],
                    temperature=0.3,
                    max_tokens=400
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response if using templates
                if content.startswith('{'):
                    import json
                    try:
                        parsed = json.loads(content)
                        return {
                            "answer_candidates": parsed.get("answer_candidates", ["Unknown"]),
                            "caption": parsed.get("image_description", "")
                        }
                    except json.JSONDecodeError:
                        logging.warning("⚠️ Failed to parse JSON VLM response, falling back to legacy mode")
                        # Fall through to legacy parsing
                        caption = content[:300]
                        vqa_prompt = self._create_short_answer_prompt(question)
                else:
                    # Non-JSON response from template - use as caption
                    caption = content[:300]
                    vqa_prompt = self._create_short_answer_prompt(question)
            else:
                # Fallback to legacy prompts
                logging.debug("⚠️ Using fallback hardcoded VLM prompts")
                
                # Generate caption
                caption_prompt = "Describe this image in detail, focusing on all visible objects, their properties, and spatial relationships."
                
                caption_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }],
                    temperature=0.3,
                    max_tokens=300
                )
                
                caption = caption_response.choices[0].message.content.strip()
                
                # Generate answer candidates with improved short answer prompt
                vqa_prompt = self._create_short_answer_prompt(question)
            
            # Generate short answer candidates
            candidates = self._generate_short_answer_candidates(image_b64, vqa_prompt, question)
            
            return {
                "answer_candidates": candidates,
                "caption": caption
            }
            
        except Exception as e:
            logging.error(f"VLM baseline analysis failed: {e}")
            return {
                "answer_candidates": ["Error: Analysis failed"],
                "caption": f"Error: {str(e)}"
            }

    def _create_short_answer_prompt(self, question: str) -> str:
        """
        Create a single, robust prompt to generate a short, direct answer for any question type.
        This approach simplifies logic and relies on the LLM's ability to understand context.
        """
        return f"""You are an expert Visual Question Answering system. Your task is to answer the following question about the image with a very short and direct response.

- If the question is a "yes/no" question, answer with only "yes" or "no".
- If the question asks "how many", answer with only a number.
- For all other questions, provide the most direct and concise answer possible (ideally 1-3 words).

Do not provide explanations or full sentences.

Question: {question}

Answer:"""

    def _generate_short_answer_candidates(self, image_b64: str, vqa_prompt: str, question: str) -> list:
        """Generate multiple short answer candidates with different approaches"""
        candidates = []
        
        try:
            # Primary answer with low temperature for consistency
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vqa_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=5  # Force very short responses
            )
            
            primary_answer = response.choices[0].message.content.strip()
            # Clean up answer - remove common prefixes
            primary_answer = self._clean_short_answer(primary_answer)
            if primary_answer:
                candidates.append(primary_answer)
            
            # For yes/no questions, always provide both options
            question_lower = question.lower()
            if any(word in question_lower for word in ['does', 'is', 'are', 'can', 'will', 'would', 'should', 'has', 'have']):
                if primary_answer.lower() == 'yes' and 'no' not in [c.lower() for c in candidates]:
                    candidates.append('no')
                elif primary_answer.lower() == 'no' and 'yes' not in [c.lower() for c in candidates]:
                    candidates.append('yes')
            else:
                # For non-yes/no questions, try to get one alternative with higher temperature
                try:
                    alt_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vqa_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                            ]
                        }],
                        temperature=0.5,
                        max_tokens=10
                    )
                    
                    alt_answer = alt_response.choices[0].message.content.strip()
                    alt_answer = self._clean_short_answer(alt_answer)
                    if alt_answer and alt_answer.lower() != primary_answer.lower():
                        candidates.append(alt_answer)
                        
                except Exception:
                    pass  # Alternative answer is optional
            
            # Ensure we have at least one candidate
            if not candidates:
                candidates = ["unknown"]
                
        except Exception as e:
            logging.error(f"Short answer generation failed: {e}")
            candidates = ["error"]
        
        return candidates[:3]  # Limit to 3 candidates maximum

    def _clean_short_answer(self, answer: str) -> str:
        """Clean and normalize short answers"""
        if not answer:
            return ""
        
        # Remove common prefixes and suffixes
        answer = answer.strip()
        
        # Remove quotes
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        
        # Remove common response prefixes
        prefixes_to_remove = [
            "the answer is", "answer:", "response:", "result:", 
            "i see", "i can see", "looking at", "in the image"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Keep only the first word if still too long
        words = answer.split()
        if len(words) > 1:
            # For numerical answers, allow joining if it seems intentional
            if all(word.isdigit() for word in words):
                 answer = "".join(words)
            else:
                 answer = words[0] # Take only the first word
        
        return answer
    
    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def answer_contextual_question(self, question: str, image_path: str, context_prompt: str) -> str:
        """
        Answer a question with additional context prompt.
        Used by SynthesizerAgent for contextual re-evaluation.
        """
        if not self.client:
            logging.error("No working backend available for contextual question")
            return "Error: Backend unavailable"
        
        try:
            image_b64 = encode_image_to_base64(image_path)
            if not image_b64:
                return "Error: Could not process image"
            
            full_prompt = f"""{context_prompt}

Question: {question}

Answer this question based on the image and the context provided above. Give a concise, direct answer:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=50
            )
            
            answer = response.choices[0].message.content.strip()
            logging.debug(f"Contextual answer: {answer}")
            return answer
            
        except Exception as e:
            logging.error(f"Contextual question answering failed: {e}")
            return f"Error: {str(e)}" 