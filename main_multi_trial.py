"""
Main script for multi-trial
"""

import os
import sys
import random
import numpy as np
import json

sys.path.insert(0, 'src')

from api_utils import HyperbolicAPI, BASE_MODEL, CHAT_MODEL
from data_utils import load_truthfulqa
from icm import run_icm
from evaluation import evaluate_on_test, zero_shot_baseline
from plotting import plot_results


def save_results(results, filename='results_multi_trial.json'):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def main():
    # Multi-trial configuration
    NUM_TRIALS = 3
    SEEDS = [42, 123, 456]  # Different random seeds for each trial

    # Hyperparameters
    NUM_SEED = 8          # Initial random labels K
    NUM_ITERATIONS = 500  # Iterations
    ALPHA = 50            # Weight for mutual predictability

    # Initialize API
    api = HyperbolicAPI()

    # Load data
    train_data = load_truthfulqa('data', split='train')
    test_data = load_truthfulqa('data', split='test')

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    # Load existing results if available
    all_results = {}
    if os.path.exists('results_multi_trial.json'):
        with open('results_multi_trial.json', 'r') as f:
            all_results = json.load(f)
        print(f"Loaded: {all_results.keys()}")

    # Zero-shot Base
    if 'zero_shot_base' not in all_results:
        print("\n" + "="*60)
        print("Running Zero-shot Base")
        print("="*60)
        random.seed(42)
        np.random.seed(42)
        acc_zero_base, _ = zero_shot_baseline(api, BASE_MODEL, test_data)
        all_results['zero_shot_base'] = acc_zero_base
        print(f"Accuracy: {acc_zero_base * 100:.2f}%")
        save_results(all_results)
    else:
        print("\n Zero-shot Base completed")
        acc_zero_base = all_results['zero_shot_base']

    # Zero-shot Chat
    if 'zero_shot_chat' not in all_results:
        print("\n" + "="*60)
        print("Running Zero-shot Chat")
        print("="*60)
        random.seed(42)
        np.random.seed(42)
        acc_zero_chat, _ = zero_shot_baseline(api, CHAT_MODEL, test_data)
        all_results['zero_shot_chat'] = acc_zero_chat
        print(f"Accuracy: {acc_zero_chat * 100:.2f}%")
        save_results(all_results)
    else:
        print("\n Zero-shot Chat completed")
        acc_zero_chat = all_results['zero_shot_chat']

    # ICM
    if 'icm_trials' not in all_results:
        all_results['icm_trials'] = []

    for trial_idx in range(NUM_TRIALS):
        if len(all_results['icm_trials']) > trial_idx:
            print(f"\n ICM trial {trial_idx + 1}/{NUM_TRIALS} already completed")
            continue

        print("\n" + "="*60)
        print(f"Running ICM Trial {trial_idx + 1}/{NUM_TRIALS} (seed={SEEDS[trial_idx]})")
        print("="*60)

        # Set seed
        random.seed(SEEDS[trial_idx])
        np.random.seed(SEEDS[trial_idx])

        icm_labeled = run_icm(
            api,
            BASE_MODEL,
            train_data,
            num_seed=NUM_SEED,
            num_iterations=NUM_ITERATIONS,
            alpha=ALPHA
        )

        # Evaluate ICM
        acc_icm, _ = evaluate_on_test(api, BASE_MODEL, icm_labeled, test_data)
        print(f"Accuracy: {acc_icm * 100:.2f}%")

        # Save
        all_results['icm_trials'].append({
            'seed': SEEDS[trial_idx],
            'accuracy': acc_icm
        })
        save_results(all_results)

        # Save ICM data
        with open(f'icm_labeled_trial{trial_idx + 1}.json', 'w') as f:
            json.dump(icm_labeled, f, indent=2)

    # Golden labels
    if 'golden_supervision' not in all_results:
        print("\n" + "="*60)
        print("Running Golden Labels")
        print("="*60)
        random.seed(42)
        np.random.seed(42)

        golden_demos = []
        for item in train_data:
            golden_demos.append({
                'uid': item['uid'],
                'prompt': item['prompt'],
                'label': item['vanilla_label']  
            })

        acc_golden, _ = evaluate_on_test(api, BASE_MODEL, golden_demos, test_data)
        all_results['golden_supervision'] = acc_golden
        print(f"Accuracy: {acc_golden * 100:.2f}%")
        save_results(all_results)
    else:
        print("\n Golden Labels completed")
        acc_golden = all_results['golden_supervision']

    # Compute statistics for ICM
    icm_accuracies = [trial['accuracy'] for trial in all_results['icm_trials']]
    icm_mean = np.mean(icm_accuracies)
    icm_std = np.std(icm_accuracies, ddof=1)  # Sample std

    plot_results_multi = {
        'Zero-shot\n(Base)': acc_zero_base,
        'Zero-shot\n(Chat)': acc_zero_chat,
        'Unsupervised\n(Ours)': icm_mean,
        'Golden\nLabels': acc_golden
    }

    error_bars = {
        'Zero-shot\n(Base)': 0,
        'Zero-shot\n(Chat)': 0,
        'Unsupervised\n(Ours)': icm_std,
        'Golden\nLabels': 0
    }

    plot_results(plot_results_multi, save_path='results_multi_trial.png', error_bars=error_bars)

    print("\nDone!")


if __name__ == "__main__":
    main()
