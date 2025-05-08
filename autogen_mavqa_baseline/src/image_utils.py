import base64
import cv2
import numpy as np
import os
# from torchvision.ops import box_convert # Keep commented unless bbox logic is added

def process_image_for_vlm_agent(image_path_or_cv2_image, app_config_ref, bbox=None):
    """
    Processes an image file path or a cv2 image (NumPy array) for VLM consumption
    via OpenAI-compatible API. Returns a base64 encoded data URL string.
    Handles basic image loading, color conversion, and encoding.
    BBox processing logic from QueryVLM is currently a placeholder.
    """
    try:
        if isinstance(image_path_or_cv2_image, str):
            if not os.path.exists(image_path_or_cv2_image):
                 raise FileNotFoundError(f"Image file not found: {image_path_or_cv2_image}")
            image = cv2.imread(image_path_or_cv2_image)
            if image is None:
                raise ValueError(f"cv2.imread failed for: {image_path_or_cv2_image}")
        elif isinstance(image_path_or_cv2_image, np.ndarray):
            image = image_path_or_cv2_image # Assume it's a valid cv2 image
        else:
            raise TypeError(f"Input must be a file path (str) or NumPy array (cv2 image), got {type(image_path_or_cv2_image)}")

        # Validate image dimensions and channels
        if image.ndim < 2 or image.ndim > 3:
             raise ValueError(f"Unexpected image dimensions: {image.shape}")
        if image.ndim == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] != 3: # Should be BGR now
            raise ValueError(f"Image has unsupported channel count: {image.shape[2]}")

        # --- Placeholder for BBox processing ---
        if bbox is not None:
            print(f"Note: BBox processing activated but logic is currently a placeholder in image_utils.py")
            # vlm_params = app_config_ref.get("vlm_interaction_params", {})
            # min_bbox_size = vlm_params.get("min_bbox_size", 32)
            # Implement cropping logic here using bbox and min_bbox_size if needed
            # Example:
            # x1, y1, x2, y2 = ... calculate crop coordinates ...
            # image = image[y1:y2, x1:x2]
        # --- End Placeholder ---

        # Convert BGR to RGB before encoding
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode to JPEG format (adjust quality as needed)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        is_success, buffer = cv2.imencode('.jpg', image_rgb, encode_param)
        if not is_success:
            raise RuntimeError("cv2.imencode failed during JPG encoding.")

        # Base64 encode
        image_bytes = buffer.tobytes()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return data URL
        return f"data:image/jpeg;base64,{base64_image}"

    except FileNotFoundError as e:
        print(f"Error in process_image: {e}")
        raise # Re-raise FileNotFoundError to be caught by caller
    except Exception as e:
        print(f"Error during image processing: {e}")
        # Depending on how critical this is, you might return None or raise a specific error
        raise ValueError(f"Image processing failed: {e}") from e