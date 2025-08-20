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

# Adapted from https://github.com/NVlabs/VILA/blob/ec7fb2c264920bf004fd9fa37f1ec36ea0942db5/server.py
# This script offers an OpenAI-compatible server for the Describe Anything Model (DAM).

import argparse
import base64
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Literal, Optional, Union, get_args

import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel
import numpy as np
import traceback
import json
import asyncio
from typing import AsyncGenerator, Generator

from dam import DescribeAnythingModel, DEFAULT_IMAGE_TOKEN, disable_torch_init


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


IMAGE_CONTENT_BASE64_REGEX = re.compile(
    r"^data:image/(png|jpe?g);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1


def load_image(image_url: str) -> Image:
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content))
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64)))
    assert image.mode == "RGBA", f"Image mode is {image.mode}, but it should be RGBA"
    return image


def process_rgba_image(rgba_pil):
    image_pil = PILImage.fromarray(np.asarray(rgba_pil)[..., :3])
    mask_pil = PILImage.fromarray(
        (np.asarray(rgba_pil)[..., 3] > 0).astype(np.uint8) * 255)

    return image_pil, mask_pil


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dam
    disable_torch_init()
    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }
    dam = DescribeAnythingModel(
        model_path=app.args.model_path,
        conv_mode=app.args.conv_mode,
        prompt_mode=prompt_modes[app.args.prompt_mode],
    )
    print(
        f"Model {dam.model_name} loaded successfully.")
    yield


app = FastAPI(debug=True, lifespan=lifespan)


async def convert_generator_to_async(gen: Generator) -> AsyncGenerator:
    for item in gen:
        yield item
        await asyncio.sleep(0)

# Load model upon startup


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global dam

        # Validate the model name (use "describe_anything_model" to skip the model name check)
        if request.model != "describe_anything_model" and request.model != dam.model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {dam.model_name}, "
                f"but the request model is {request.model}"
            )

        messages = request.messages

        images = []
        query = ""

        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    query += message.content
                elif isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            query += content.text
                        elif content.type == "image_url":
                            image = load_image(content.image_url.url)
                            assert image.mode == "RGBA", f"Image mode is {image.mode}, but it should be RGBA"
                            images.append(image)
                        else:
                            raise ValueError("Unsupported content type")
            elif message.role == "assistant":
                pass  # We can ignore assistant messages in the input

        if len(images) == 0:
            raise ValueError("No image with mask found in input messages.")

        # Remove the prefix of the query if it exists. We detect the prefix and add it back on our own.
        query = query.strip()
        query = query.removeprefix("Image:")
        query = query.removeprefix("Video:")
        query = query.strip()
        while query.startswith(DEFAULT_IMAGE_TOKEN):
            query = query.removeprefix(DEFAULT_IMAGE_TOKEN)
        assert DEFAULT_IMAGE_TOKEN not in query, f"{DEFAULT_IMAGE_TOKEN} should not be in other positions than the beginning of the query"
        query = query.strip()

        if app.args.image_video_joint_checkpoint:
            if len(images) == 1:
                query = f"Image: {DEFAULT_IMAGE_TOKEN}\n{query}"
            elif len(images) == 8:
                query = f"Video: {DEFAULT_IMAGE_TOKEN * 8}\n{query}"
            else:
                raise ValueError(
                    f"Only 1 image and video (with 8 frames) are supported, but {len(images)} images are provided")
        else:
            assert len(images) == 1, "Only one image with mask is supported"
            query = f"{DEFAULT_IMAGE_TOKEN}\n{query}"

        # Print the query for debugging
        # print(f"Query: {query}")

        pils = [process_rgba_image(image) for image in images]

        image_pils, mask_pils = zip(*pils)

        if request.stream:
            async def generate_stream():
                try:
                    description_generator = dam.get_description(
                        image_pils,
                        mask_pils,
                        query,
                        streaming=True,
                        temperature=app.args.temperature,
                        top_p=app.args.top_p,
                        num_beams=app.args.num_beams,
                        max_new_tokens=app.args.max_new_tokens,
                    )
                    async for text in convert_generator_to_async(description_generator):
                        chunk = {
                            "id": uuid.uuid4().hex,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "delta": {
                                    "content": [{
                                        "type": "text",
                                        "text": text
                                    }]
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Send the final chunk
                    yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Error in stream: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            outputs = dam.get_description(
                image_pils,
                mask_pils,
                query,
                streaming=False,
                temperature=app.args.temperature,
                top_p=app.args.top_p,
                num_beams=app.args.num_beams,
                max_new_tokens=app.args.max_new_tokens,
            )

            return {
                "id": uuid.uuid4().hex,
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"message": ChatMessage(
                        role="assistant", content=outputs)}
                ],
            }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":
    # Example: python dam_server.py --model-path nvidia/DAM-3B --conv-mode v1 --prompt-mode focal_prompt --temperature 0.2 --top_p 0.9 --num_beams 1 --max_new_tokens 512 --workers 1
    # Example: python dam_server.py --model-path nvidia/DAM-3B-Video --conv-mode v1 --prompt-mode focal_prompt --temperature 0.2 --top_p 0.9 --num_beams 1 --max_new_tokens 512 --workers 1 --image_video_joint_checkpoint
    host = os.getenv("DAM_HOST", "0.0.0.0")
    port = int(os.getenv("DAM_PORT", "8000"))
    model_path = os.getenv("DAM_MODEL_PATH", "nvidia/DAM-3B")
    conv_mode = os.getenv("DAM_CONV_MODE", "v1")
    workers = int(os.getenv("DAM_WORKERS", "1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    parser.add_argument("--prompt-mode", type=str, default="focal_prompt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--image_video_joint_checkpoint", action="store_true",
                        help="The loaded checkpoint is an image-video joint checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    app.args = parser.parse_args()

    if "joint" in app.args.model_path and not app.args.image_video_joint_checkpoint:
        print("Warning: The loaded checkpoint looks like an image-video joint checkpoint, but the --image_video_joint_checkpoint flag is not set. This might lead to incorrect behavior, as joint checkpoints use a different prompt format even for single image inputs.")

    uvicorn.run(app, host=app.args.host, port=app.args.port,
                workers=app.args.workers)
