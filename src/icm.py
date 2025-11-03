"""
Simplified ICM Algorithm
Adapted from ICM.py main() function
"""

import random
import math
import numpy as np
from tqdm import tqdm

from api_utils import HyperbolicAPI
from prompt_utils import get_judge_prompt_fewshot
from logprob_utils import get_yes_no_diff_logprobs


def predict_label(api, model_name, target_example, labeled_examples):
    """
    Predict label using mutual predictability
    Adapted from predict_assignment()
    """
    # Create few-shot prompt
    demos = [ex for ex in labeled_examples if ex['uid'] != target_example['uid']]
    prompt = get_judge_prompt_fewshot(target_example, demos)

    # Query API for logprobs
    response = api(
        model_name=model_name,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=20
    )

    # Extract logprobs
    if "choices" in response and len(response["choices"]) > 0:
        choice = response["choices"][0]
        if "logprobs" in choice and "top_logprobs" in choice["logprobs"]:
            # Get first token's logprobs
            top_logprobs = choice["logprobs"]["top_logprobs"][0]
            score = get_yes_no_diff_logprobs(top_logprobs)
        else:
            score = 0.0
    else:
        score = 0.0

    label = 1 if score > 0 else 0
    return label, score


def calculate_score(examples, alpha=50):
    """
    Calculate U(D) = alpha * P_theta(D)
    No inconsistency term since we ignore ConsistencyFix
    Adapted from get_energy()
    """
    scores = []
    for ex in examples:
        if ex['label'] is not None:
            if ex['label'] == 1:
                scores.append(ex['score'])
            else:
                scores.append(-ex['score'])

    if len(scores) == 0:
        return -1e6

    mean_prob = np.mean(scores)
    return alpha * mean_prob #no reason to use alpha here but keeping for consistency


def get_temperature(iteration, initial_temp=10.0, final_temp=0.01):
    """
    Calculate temperature for simulated annealing (logarithmic schedule)
    Adapted from get_temperature()
    """
    return max(final_temp, initial_temp / (1 + 0.99 * np.log(1 + iteration))) #original code uses 2


def run_icm(api, model_name, train_data, num_seed=8, num_iterations=500, alpha=50):
    """
    Main ICM algorithm
    Adapted from ICM main()
    """
    # Initialize examples
    examples = []
    for item in train_data:
        examples.append({
            'uid': item['uid'],
            'prompt': item['prompt'],
            'question': item['question'],
            'choice': item['choice'],
            'vanilla_label': item['vanilla_label'],
            'label': None,
            'score': 0.0
        })

    print(f"Initializing with {num_seed} example")

    # Randomly label num_seed examples
    labeled_ids = random.sample(range(len(examples)), num_seed)
    for idx in labeled_ids:
        examples[idx]['label'] = random.choice([0, 1])

    # Get initial scores
    labeled_examples = [ex for ex in examples if ex['label'] is not None]
    for ex in labeled_examples:
        _, ex['score'] = predict_label(api, model_name, ex, labeled_examples)

    current_score = calculate_score(examples, alpha)
    flip_count = 0

    print(f"ICM iteration start")

    # Main loop
    for iteration in tqdm(range(num_iterations)):
        # Update temperature
        T = get_temperature(flip_count)

        # Sample example
        example_id = random.choice(range(len(examples)))

        # Predict new label
        labeled_examples = [ex for ex in examples if ex['label'] is not None]
        new_label, new_score = predict_label(api, model_name, examples[example_id], labeled_examples)

        # If label changed
        if examples[example_id]['label'] != new_label:
            # Store old values
            old_label = examples[example_id]['label']
            old_score = examples[example_id]['score']

            # Try new label
            examples[example_id]['label'] = new_label
            examples[example_id]['score'] = new_score

            # Calculate new score
            new_total_score = calculate_score(examples, alpha)

            # Simulated annealing acceptance
            delta = new_total_score - current_score
            accept_prob = min(1.0, math.exp(delta / T)) if T > 0 else (1.0 if delta > 0 else 0.0)

            if random.random() < accept_prob:
                # Accept
                current_score = new_total_score
                flip_count += 1
            else:
                # Reject - revert
                examples[example_id]['label'] = old_label
                examples[example_id]['score'] = old_score

    return examples