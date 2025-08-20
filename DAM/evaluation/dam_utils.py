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

import argparse
import numpy as np
import torch
from PIL import Image

from dam import DescribeAnythingModel
from dam import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dam import SeparatorStyle, conv_templates
from dam import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_image, tokenizer_image_token)
from dam import load_pretrained_model
from dam import disable_torch_init

args, model, tokenizer, prompt, conv = None, None, None, None, None
crop_image = DescribeAnythingModel.crop_image

def get_prompt(qs):
    if DEFAULT_IMAGE_TOKEN not in qs:
        raise ValueError("no <image> tag found in input.")

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt, conv

def get_description(image_pil, mask_pil):
    describe_anything_image_mode = args.crop_mode
    
    if "+" not in describe_anything_image_mode:
        # Only one image
        crop_mode, crop_mode2 = describe_anything_image_mode, None
    else:
        crop_mode, crop_mode2 = describe_anything_image_mode.split("+")
        if args.no_concat_images:
            assert crop_mode == "full", "Current prompt only supports first crop as full (non-cropped). If you need other specifications, please update the prompt."
    
    mask_np = (np.asarray(mask_pil) / 255).astype(np.uint8)
    full_mask_np = mask_np
    images_tensor, image_info = process_image(image_pil, model.config, None, pil_preprocess_fn=lambda pil_img: crop_image(image_pil, mask_np=mask_np, crop_mode=crop_mode))
    images_tensor = images_tensor[None].to(model.device, dtype=torch.float16)

    mask_np = image_info["mask_np"]
    mask_pil = Image.fromarray(mask_np * 255)
    
    masks_tensor = process_image(mask_pil, model.config, None)
    masks_tensor = masks_tensor[None].to(model.device, dtype=torch.float16)
    
    images_tensor = torch.cat((images_tensor, masks_tensor[:, :1, ...]), dim=1)

    if crop_mode2 is not None:
        images_tensor2, image_info2 = process_image(image_pil, model.config, None, pil_preprocess_fn=lambda pil_img: crop_image(pil_img, mask_np=full_mask_np, crop_mode=crop_mode2))
        images_tensor2 = images_tensor2[None].to(model.device, dtype=torch.float16)

        mask_np2 = image_info2["mask_np"]
        mask_pil2 = Image.fromarray(mask_np2 * 255)
        
        masks_tensor2 = process_image(mask_pil2, model.config, None)
        masks_tensor2 = masks_tensor2[None].to(model.device, dtype=torch.float16)
        
        images_tensor2 = torch.cat((images_tensor2, masks_tensor2[:, :1, ...]), dim=1)
    else:
        images_tensor2 = None
    
    # `images_tensor`: 4 channels
    info = dict(images_tensor=images_tensor, images_tensor2=images_tensor2)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if images_tensor2 is not None:
        if args.no_concat_images:
            images = [
                images_tensor,
                images_tensor2
            ]
        else:
            images = [
                torch.cat((images_tensor, images_tensor2), dim=1)
            ]
    else:
        images = [images_tensor]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    return outputs, info

def init(model_path, **kwargs):
    global args, model, tokenizer, prompt, conv

    args = argparse.Namespace(
        model_path=model_path,
        model_base=None,
        **kwargs
    )
    
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    model.config.image_processor = image_processor

    qs = args.query

    prompt, conv = get_prompt(qs)
