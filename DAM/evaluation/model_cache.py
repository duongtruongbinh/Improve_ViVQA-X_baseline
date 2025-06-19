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

import os
import json

cache_base = "model_outputs_cache"
def cache_model_outputs(cache_name, cache_values, overwrite=False):
    os.makedirs(cache_base, exist_ok=True)
    with open(os.path.join(cache_base, cache_name + ".json"), 'w' if overwrite else 'x') as f:
        json.dump(cache_values, f, indent=4)

def parse_key(k):
    try:
        return int(k)
    except ValueError:
        return k

def load_cached_model_outputs(cache_name):
    cache_path = os.path.join(cache_base, cache_name + ".json")
    if not os.path.exists(cache_path):
        return {}
    print("Loading cache from", cache_path)
    with open(cache_path, 'r') as f:
        model_outputs = json.load(f)
    model_outputs = {parse_key(k): v for k, v in model_outputs.items()}
    return model_outputs
