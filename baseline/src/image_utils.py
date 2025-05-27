# image_utils.py
import base64
import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
# from torchvision.ops import box_convert 

class BaseImageProcessor(ABC):
    """Base class for image processing operations."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
    def validate_image_path(self, image_path: str) -> None:
        """Validate that the image path exists and is accessible."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
    def validate_image_array(self, image: np.ndarray) -> None:
        """Validate image array dimensions and channels."""
        if image.ndim < 2 or image.ndim > 3:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(f"Image has unsupported channel count: {image.shape[2]}")
            
    def process_bbox(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Process bounding box if provided."""
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
        return image
        
    @abstractmethod
    def process(self, image_input: Union[str, np.ndarray], bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Process the image and return the result."""
        pass

class VLMImageProcessor(BaseImageProcessor):
    """Image processor specifically for VLM (Vision Language Model) consumption."""
    
    def process(self, image_input: Union[str, np.ndarray], bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Process an image for VLM consumption via OpenAI-compatible API.
        Returns a base64 encoded data URL string.
        """
        try:
            # Handle input type
            if isinstance(image_input, str):
                self.validate_image_path(image_input)
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"cv2.imread failed for: {image_input}")
            elif isinstance(image_input, np.ndarray):
                image = image_input
            else:
                raise TypeError(f"Input must be a file path (str) or NumPy array (cv2 image), got {type(image_input)}")

            # Validate and convert image
            self.validate_image_array(image)
            
            # Process bbox if provided
            image = self.process_bbox(image, bbox)
            
            # Convert to RGB and encode
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            is_success, buffer = cv2.imencode('.jpg', image_rgb, encode_param)
            
            if not is_success:
                raise RuntimeError("cv2.imencode failed during JPG encoding.")

            # Create base64 data URL
            image_bytes = buffer.tobytes()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"

        except Exception as e:
            raise ValueError(f"Image processing failed: {e}") from e

# For backward compatibility
def process_image_for_vlm_agent(image_path_or_cv2_image: Union[str, np.ndarray], 
                              app_config_ref: Optional[dict] = None, 
                              bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
    """Legacy function that uses VLMImageProcessor internally."""
    processor = VLMImageProcessor(config=app_config_ref)
    return processor.process(image_path_or_cv2_image, bbox)
