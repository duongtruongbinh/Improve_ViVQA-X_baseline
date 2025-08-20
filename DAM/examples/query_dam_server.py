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

def main(model, server_url):
    # One could load an RGBA image (in png format) or could load an RGB image and a mask
    # Here we load an image and the mask separately
    image = Image.open("images/1.jpg").convert("RGB")
    mask = Image.open("images/1_example_mask.png").convert("L")
    rgba_image = Image.merge("RGBA", image.split() + (mask,))
    # You can load this RGBA image instead of the image and mask separately
    # rgba_image.save("images/1_example_masked.png")
    buffered = BytesIO()
    rgba_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_base64 = f"data:image/png;base64,{img_str}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64,
                    }
                },
                {
                    "type": "text",
                    "text": "\nDescribe the masked region in detail.",
                }
            ],
        }
    ]
    
    client = OpenAI(api_key="api_key", base_url=server_url)
    
    if not args.no_stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            top_p=0.5,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                for content_piece in chunk.choices[0].delta.content:
                    if content_piece['type'] == 'text':
                        print(content_piece['text'], end='', flush=True)
        print()  # Final newline
    else:
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
    parser = argparse.ArgumentParser(description="Specify model.")
    parser.add_argument("--model", type=str, default="describe_anything_model")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()
    main(args.model, args.server_url)
