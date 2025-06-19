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

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import model_cache
import argparse
import random
import torch

query_dict = {
    "default": "<image>\nDescribe the masked region in detail.",
    "joint": "Image: <image>\nGiven the image above, describe the object in the masked region.",
}

def select_ann(img_id, area_min=None, area_max=None):
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

    if area_min is not None:
        ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] >= area_min]

    if area_max is not None:
        ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] <= area_max]
    
    return ann_ids

def get_mask(ann_id):
    anns = coco.loadAnns([ann_id])
    mask = coco.annToMask(anns[0]) * 255

    return mask

if __name__ == "__main__":
    # Example (run in evaluation directory):
    # python get_model_outputs.py --model_type dam --model_path nvidia/DAM-3B
    
    parser = argparse.ArgumentParser(description='Get model outputs')
    parser.add_argument("--model_type", type=str, help="Type of the model")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--crop-mode", type=str, help="Crop mode", default="full+focal_crop")
    parser.add_argument("--conv-mode", type=str, help="Conversation mode", default="v1")
    parser.add_argument("--query", type=str, help="Query for the model", default="<image>\nDescribe the masked region in detail.")
    parser.add_argument("--query-key", type=str, help="Query key for the model. Default: use --query.", default=None, choices=["default", "joint"])
    parser.add_argument("--use-keys", type=str, help="Use a subset of the keys", default=None)
    parser.add_argument("--data-root", type=str, help="Data root", default="DLC-bench")
    parser.add_argument("--no-concat-images", action="store_true", help="Do not concat images in the channel dimension")
    parser.add_argument("--suffix", type=str, help="Suffix to the saved json", default="")
    parser.add_argument("--temperature", type=float, help="Temperature", default=0.2)
    parser.add_argument("--top-p", type=float, help="Top p", default=None)
    parser.add_argument("--cache_base", default=None, type=str, help="Override the cache base")
    parser.add_argument("--cache-name-override", type=str, help="Override the cache name", default=None)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    args = parser.parse_args()

    if args.cache_base is not None:
        model_cache.cache_base = args.cache_base

    # This coco instance is actually an o365 subset. This is for code reuse.
    coco = COCO(os.path.join(args.data_root, 'annotations.json'))

    if args.query_key is not None:
        query = query_dict[args.query_key]
        print(f"Query key: {args.query_key}, overriding query: {args.query} to {query}")
        args.query = query
    else:
        print(f"Query: {args.query}")

    # This is for sanity checks
    if args.conv_mode == "llama_3":
        assert "llama3" in args.model_path or "llama-3" in args.model_path or "llama_3" in args.model_path, "Model path does not contain llama3"
    elif args.conv_mode == "v1":
        assert "llama" not in args.model_path, "Model path contains llama"

    if args.model_type == "dam":
        import dam_utils as eval_utils
    
        init_kwargs = dict(
            sep=", ", 
            conv_mode=args.conv_mode, 
            query=args.query, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            num_beams=1, 
            max_new_tokens=512,
            crop_mode=args.crop_mode,
            no_concat_images=args.no_concat_images,
        )
        eval_utils.init(model_path=args.model_path, **init_kwargs)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    if args.use_keys is not None:
        with open(args.use_keys, 'r') as f:
            use_keys = json.load(f)
    else:
        use_keys = None

    img_ids = list(coco.imgs.keys())

    model_outputs = {}

    cache_name = args.cache_name_override if args.cache_name_override is not None else os.path.basename(args.model_path) + args.suffix
    model_outputs = model_cache.load_cached_model_outputs(cache_name)

    num_anns = len(coco.anns)

    # Set all seeds for reproducibility. If args.seed is set to -1, no seed is set.
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    pbar = tqdm(total=num_anns)
    for img_id in img_ids:
        ann_ids = select_ann(img_id)
        img_info = coco.loadImgs(img_id)[0]

        for i, ann_id in enumerate(ann_ids):
            if use_keys is not None and str(ann_id) not in use_keys:
                # print(f"Skipping ann id {ann_id}")
                pbar.update(1)
                continue
            
            if ann_id in model_outputs.keys():
                pbar.update(1)
                continue

            mask = get_mask(ann_id)
            mask = Image.fromarray(mask)
            
            img_path = os.path.join(args.data_root, "images", img_info['file_name'])
            img = Image.open(img_path)

            outputs, info = eval_utils.get_description(img, mask)
            print(f"img id: {img_id}, ann id: {ann_id}")
            print(outputs)
            model_outputs[ann_id] = outputs
            
            pbar.update(1)
    pbar.close()

    # print(model_outputs)

    model_outputs["query"] = args.query

    model_cache.cache_model_outputs(cache_name, model_outputs, overwrite=True)

    print(f"Cache name: {cache_name}")
