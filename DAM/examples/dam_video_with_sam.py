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

# This script is used to segment objects in a video using SAM and then describe the segmented objects using DAM. 
# This script uses SAM (v1) and requires localization for all the frames.

import argparse
import ast
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from dam import DescribeAnythingModel, disable_torch_init
import cv2
import glob
import os

def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
    inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes, 
                         input_labels=input_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np

def add_contour(img, mask, input_points=None, input_boxes=None):
    img = img.copy()

    # Draw contour
    mask = mask.astype(np.uint8) * 255
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
        for box_batch in input_boxes:  # Handle batch of boxes
            for box in box_batch:  # Iterate through boxes in the batch
                x1, y1, x2, y2 = map(int, box)
                # Draw rectangle with white color
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=4)
                # Draw inner rectangle with red color
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=2)

    return img

def denormalize_coordinates(coords, image_size, is_box=False):
    """Convert normalized coordinates (0-1) to pixel coordinates."""
    width, height = image_size
    if is_box:
        # For boxes: [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords
        return [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        ]
    else:
        # For points: [x, y]
        x, y = coords
        return [int(x * width), int(y * height)]

def print_streaming(text):
    """Helper function to print streaming text with flush"""
    print(text, end="", flush=True)

if __name__ == '__main__':
    # Note that when we only provide one set of points or boxes, it will be used for all frames.
    # Example: python examples/dam_video_with_sam.py --video_dir videos/1 --points '[[[1172, 812], [1572, 800]]]'
    parser = argparse.ArgumentParser(description="Describe Anything script")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--points_list', type=str, default=None, 
                       help='List of points for each frame, format: [[[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]], ...]')
    parser.add_argument('--boxes_list', type=str, default=None,
                       help='List of boxes for each frame, format: [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]')
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

    args = parser.parse_args()

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # Get list of image files and sort them
    image_files = sorted(glob.glob(os.path.join(args.video_dir, "*.jpg")))
    
    # Select 8 frames uniformly
    if len(image_files) < 8:
        # Upsample by repeating frames
        indices = np.linspace(0, len(image_files)-1, 8, dtype=int)
    else:
        # Downsample by selecting frames uniformly
        indices = np.linspace(0, len(image_files)-1, 8, dtype=int)
    
    selected_files = [image_files[i] for i in indices]

    # Parse points and boxes for each frame
    if args.points_list:
        points_list = ast.literal_eval(args.points_list)
        if len(points_list) == 1:  # If only one set of points provided, use it for all frames
            points_list = points_list * 8
        elif len(points_list) != 8:
            raise ValueError("Must provide either 1 or 8 sets of points")
    else:
        points_list = None

    if args.boxes_list:
        boxes_list = ast.literal_eval(args.boxes_list)
        if len(boxes_list) == 1:  # If only one box provided, use it for all frames
            boxes_list = boxes_list * 8
        elif len(boxes_list) != 8:
            raise ValueError("Must provide either 1 or 8 boxes")
    else:
        boxes_list = None

    # Process each frame
    processed_images = []
    processed_masks = []

    for idx, image_path in enumerate(selected_files):
        img = Image.open(image_path).convert('RGB')
        image_size = img.size

        # Prepare input_points or input_boxes for this frame
        if args.use_box and boxes_list:
            input_boxes = boxes_list[idx]
            if args.normalized_coords:
                input_boxes = denormalize_coordinates(input_boxes, image_size, is_box=True)
            input_boxes = [[input_boxes]]
            mask_np = apply_sam(img, input_boxes=input_boxes)
        elif not args.use_box and points_list:
            input_points = points_list[idx]
            if args.normalized_coords:
                input_points = [denormalize_coordinates(point, image_size) 
                              for point in input_points]
            input_labels = [1] * len(input_points)
            input_points = [[x, y] for x, y in input_points]
            input_points = [input_points]
            input_labels = [input_labels]
            mask_np = apply_sam(img, input_points=input_points, input_labels=input_labels)
        else:
            raise ValueError("Must provide either points or boxes")

        mask = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        processed_images.append(img)
        processed_masks.append(mask)

        # Save visualization if requested
        if args.output_image_path:
            img_np = np.asarray(img).astype(float) / 255.0
            vis_points = input_points if not args.use_box else None
            vis_boxes = input_boxes if args.use_box else None
            img_with_contour_np = add_contour(img_np, mask_np, 
                                            input_points=vis_points,
                                            input_boxes=vis_boxes)
            img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.0).astype(np.uint8))
            output_path = f"{os.path.splitext(args.output_image_path)[0]}_{idx:02d}.png"
            img_with_contour_pil.save(output_path)
            print(f"Output image with contour saved as {output_path}")

    # Initialize DAM model
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
