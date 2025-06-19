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

# This script offers an example of how to use the DAM OpenAI-compatible server.

import argparse
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
import numpy as np
def main(model, server_url):
    # Process multiple frames from the video
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": "Video: "
    })
    
    # Load and process each frame and its mask
    for frame_idx in np.linspace(0, 11, 8):
        frame_idx = int(frame_idx)
        frame_path = f"videos/1/{frame_idx:05d}.jpg"
        mask_path = f"videos/1_masks/{frame_idx:05d}_mask_1.png"
        
        image = Image.open(frame_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        rgba_image = Image.merge("RGBA", image.split() + (mask,))
        
        buffered = BytesIO()
        rgba_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_base64 = f"data:image/png;base64,{img_str}"
        
        # Add each frame to the content list
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": img_base64,
            }
        })
    
    # Add the text prompt after all frames
    messages[0]["content"].append({
        "type": "text",
        "text": "\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail."
    })
    
    client = OpenAI(api_key="api_key", base_url=server_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.5,
    )
    
    assistant_content = response.choices[0].message.content
    print(assistant_content)

if __name__ == "__main__":
    # Example: python examples/query_dam_server_video.py --model DAM-3B-Video --server_url http://localhost:8000
    parser = argparse.ArgumentParser(description="Specify model.")
    parser.add_argument("--model", type=str, default="DAM-3B-Video", help="Model to use")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    main(args.model, args.server_url)
