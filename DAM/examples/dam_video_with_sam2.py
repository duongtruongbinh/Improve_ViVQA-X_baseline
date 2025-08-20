# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This script is used to segment objects in a video using SAM2 and then describe the segmented objects using DAM. 
# This script uses SAM (v2.1) and requires localization for the first frame.

import argparse
import ast
import torch
import numpy as np
from PIL import Image
from dam import DescribeAnythingModel, disable_torch_init
import cv2
import glob
import os
import tempfile
from sam2.build_sam import build_sam2_video_predictor

def extract_frames_from_video(video_path):
    """Extract frames from a video file and save them to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        raise ValueError("No frames were extracted from the video.")
    
    return frame_paths, temp_dir

def apply_sam2(image_files, points=None, box=None, normalized_coords=False):
    """Apply SAM2 to video frames using points or box on first frame"""

    # If coordinates are normalized, convert them to absolute coordinates
    if normalized_coords:
        # Read first frame to get dimensions
        first_frame = cv2.imread(image_files[0])
        height, width = first_frame.shape[:2]
        
        if points is not None:
            points = np.array(points, dtype=np.float32)
            points[:, 0] *= width
            points[:, 1] *= height
        elif box is not None:
            box = np.array(box, dtype=np.float32)
            box[0] *= width  # x1
            box[1] *= height # y1
            box[2] *= width  # x2
            box[3] *= height # y2

    # Initialize inference state
    video_dir = os.path.dirname(image_files[0])
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    # Add points or box on first frame
    ann_frame_idx = 0
    ann_obj_id = 1

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if points is not None:
            # Convert points to numpy array and add labels (all positive)
            points = np.array(points, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels
            )
        elif box is not None:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box
            )

        # Propagate through video and collect masks
        masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            masks.append(mask)

    return masks

def print_streaming(text):
    """Helper function to print streaming text with flush"""
    print(text, end="", flush=True)

def add_contour(img, mask, input_points=None, input_boxes=None):
    """Add contours, points, and boxes to the image for visualization."""
    img = img.copy()

    # Ensure mask is 2D and uint8
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask = (mask * 255).astype(np.uint8)
    
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

    # Draw points if provided
    if input_points is not None:
        for points in input_points:  # Handle batch of points
            for x, y in points:
                # Draw a filled circle for each point
                cv2.circle(img, (int(x), int(y)), radius=10, color=(1.0, 0.0, 0.0), thickness=-1)
                # Draw a white border around the circle
                cv2.circle(img, (int(x), int(y)), radius=10, color=(1.0, 1.0, 1.0), thickness=2)

    # Draw boxes if provided
    if input_boxes is not None:
        for box in input_boxes:  # Iterate through boxes
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle with white color
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=4)
            # Draw inner rectangle with red color
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=2)

    return img

if __name__ == '__main__':
    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --points '[[1824, 397]]' --output_image_dir videos/1_visualization
    # Example: python examples/dam_video_with_sam2.py --video_file videos/1.mp4 --points '[[1824, 397]]' --output_image_dir videos/1_visualization

    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --box '[1612, 364, 1920, 430]' --output_image_dir videos/1_visualization
    
    parser = argparse.ArgumentParser(description="Describe Anything script")
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument('--video_dir', type=str, help='Directory containing video frames')
    video_group.add_argument('--video_file', type=str, help='Path to video file (e.g., mp4)')
    parser.add_argument('--points', type=str, default=None, 
                       help='Points for first frame, format: [[x1,y1], [x2,y2], ...]')
    parser.add_argument('--box', type=str, default=None,
                       help='Box for first frame, format: [x1,y1,x2,y2]')
    parser.add_argument('--query', type=str, default='Video: <image><image><image><image><image><image><image><image>\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.', help='Prompt for the model')
    parser.add_argument('--model_path', type=str, default='nvidia/DAM-3B-Video', help='Path to the model checkpoint')
    parser.add_argument('--prompt_mode', type=str, default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str, default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p for sampling')
    parser.add_argument('--output_image_path', type=str, default=None, help='Path to save the output image with contour')
    parser.add_argument('--normalized_coords', action='store_true', 
                       help='Interpret coordinates as normalized (0-1) values')
    parser.add_argument('--no_stream', action='store_true', help='Disable streaming output')
    parser.add_argument('--use_box', action='store_true', help='Use bounding boxes instead of points')
    parser.add_argument('--output_image_dir', type=str, default=None, 
                       help='Directory to save the output images with contours')

    args = parser.parse_args()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Get list of image files and sort them
    if args.video_file:
        image_files, temp_dir = extract_frames_from_video(args.video_file)
    else:
        image_files = sorted(glob.glob(os.path.join(args.video_dir, "*.jpg")))
    
    # Select 8 frames uniformly
    indices = np.linspace(0, len(image_files)-1, 8, dtype=int)
    
    selected_files = [image_files[i] for i in indices]

    # Parse points or box for first frame
    points = ast.literal_eval(args.points) if args.points else None
    box = ast.literal_eval(args.box) if args.box else None

    # Process video with SAM2
    masks = apply_sam2(image_files, points=points, box=box, normalized_coords=args.normalized_coords)
    
    # Select masks for the 8 frames we want
    selected_masks = [masks[i] for i in indices]

    # Convert frames to PIL images
    processed_images = [Image.open(f).convert('RGB') for f in selected_files]
    processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in selected_masks]

    # Initialize DAM model and get description
    disable_torch_init()

    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }
    
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode),
    ).to(device)

    # Get description
    print("Description:")
    if not args.no_stream:
        for token in dam.get_description(processed_images, processed_masks, args.query, streaming=True, temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=512):
            print_streaming(token)
        print()
    else:
        outputs = dam.get_description(processed_images, processed_masks, args.query, temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=512)
        print(f"Description:\n{outputs}")

    # Add visualization code before DAM processing
    if args.output_image_dir:
        os.makedirs(args.output_image_dir, exist_ok=True)
        
        # Prepare visualization inputs
        vis_points = [points] if points is not None else None
        vis_box = [box] if box is not None else None
        
        # Save visualizations for selected frames
        for idx, (img_path, mask) in enumerate(zip(selected_files, selected_masks)):
            # Read image and convert to float
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = img.astype(float) / 255.0
            
            # Add contours and points/box
            img_with_contour_np = add_contour(img_np, mask, 
                                            input_points=vis_points,
                                            input_boxes=vis_box)
            
            # Convert back to uint8 and save
            img_with_contour = Image.fromarray((img_with_contour_np * 255.0).astype(np.uint8))
            output_path = os.path.join(args.output_image_dir, f'frame_{idx:03d}.png')
            img_with_contour.save(output_path)
        
        print(f"Output images with contours saved in {args.output_image_dir}")

    # Clean up temporary directory if we extracted frames from video
    if args.video_file:
        import shutil
        shutil.rmtree(temp_dir)
