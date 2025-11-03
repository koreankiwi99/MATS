"""
API utilities for Hyperbolic
"""

import os
import requests
import time


class HyperbolicAPI:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get("HYPERBOLIC_API_KEY")
            if not api_key:
                raise ValueError("HYPERBOLIC_API_KEY not set. Please set environment variable.")

        self.api_key = api_key
        self.base_url = "https://api.hyperbolic.xyz/v1/completions"
        self.total_tokens = 0

    def __call__(self, model_name, prompt, max_tokens=1, temperature=0.0, logprobs=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Detect chat model
        is_chat_model = "Instruct" in model_name or "instruct" in model_name

        if is_chat_model:
            # Use chat/completions endpoint
            url = "https://api.hyperbolic.xyz/v1/chat/completions"
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if logprobs is not None:
                data["logprobs"] = True
                data["top_logprobs"] = logprobs
        else:
            # Use completions endpoint
            url = self.base_url
            data = {
                "prompt": prompt,
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if logprobs is not None:
                data["logprobs"] = logprobs

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise error if request failed

        result = response.json()
        return result

# Model names
BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"
CHAT_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"