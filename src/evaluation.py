"""
Evaluation utilities
"""

import numpy as np
from tqdm import tqdm

from prompt_utils import get_judge_prompt_fewshot
from logprob_utils import get_yes_no_diff_logprobs


def evaluate_on_test(api, model_name, demonstrations, test_data, save_details=True):
    """
    Evaluate model on test set using demonstrations
    return accuracy, detailed_results (if save_details=True)
    """
    correct = 0
    total = 0
    detailed_results = []

    for item in tqdm(test_data, desc="Evaluating"):
        # Create test example
        test_ex = {
            'uid': -1,
            'prompt': item['prompt']
        }

        # Create few-shot prompt with demonstrations
        prompt = get_judge_prompt_fewshot(test_ex, demonstrations)

        # Query API
        response = api(
            model_name=model_name,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=20
        )

        # Extract prediction
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "logprobs" in choice and choice["logprobs"] is not None and "top_logprobs" in choice["logprobs"]:
                top_logprobs = choice["logprobs"]["top_logprobs"][0]
                score = get_yes_no_diff_logprobs(top_logprobs)
                prediction = 1 if score > 0 else 0
            else:
                prediction = 0
        else:
            prediction = 0

        # Check correctness
        if prediction == item['label']:
            correct += 1
        total += 1

        # Save detailed results
        if save_details:
            detailed_results.append({
                'uid': item.get('uid', total-1),
                'prompt': prompt,
                'response': response,
                'prediction': prediction,
                'ground_truth': item['label'],
                'correct': prediction == item['label']
            })

    accuracy = correct / total if total > 0 else 0.0
    if save_details:
        return accuracy, detailed_results
    return accuracy


def zero_shot_baseline(api, model_name, test_data, save_details=True):
    """
    Zero-shot baseline (no demonstrations)
    return accuracy, detailed_results (if save_details=True)
    """
    correct = 0
    total = 0
    detailed_results = []

    for item in tqdm(test_data, desc="Zero-shot baseline"):
        prompt = item['prompt']

        response = api(
            model_name=model_name,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=20
        )

        # Extract prediction
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "logprobs" in choice and choice["logprobs"] is not None and "top_logprobs" in choice["logprobs"]:
                top_logprobs = choice["logprobs"]["top_logprobs"][0]
                score = get_yes_no_diff_logprobs(top_logprobs)
                prediction = 1 if score > 0 else 0
            else:
                prediction = 0
        else:
            prediction = 0

        if prediction == item['label']:
            correct += 1
        total += 1

        # Save detailed results
        if save_details:
            detailed_results.append({
                'uid': item.get('uid', total-1),
                'prompt': prompt,
                'response': response,
                'prediction': prediction,
                'ground_truth': item['label'],
                'correct': prediction == item['label']
            })

    accuracy = correct / total if total > 0 else 0.0
    if save_details:
        return accuracy, detailed_results
    return accuracy
