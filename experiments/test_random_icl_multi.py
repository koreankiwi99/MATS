"""
Test ICL with random labels (no ICM) - Multi-trial
Run 3 times with different seeds to match ICM experiment setup
"""

import sys
import random
import numpy as np
import json

sys.path.insert(0, 'src')

from api_utils import HyperbolicAPI, BASE_MODEL
from data_utils import load_truthfulqa
from evaluation import evaluate_on_test

def main():
    # Multi-trial configuration (same as ICM)
    NUM_TRIALS = 3
    SEEDS = [42, 123, 456]

    # Initialize API
    api = HyperbolicAPI()

    # Load data
    train_data = load_truthfulqa('data', split='train')
    test_data = load_truthfulqa('data', split='test')

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    all_results = {'random_icl_trials': []}

    # Run 3 trials with random labels
    for trial_idx in range(NUM_TRIALS):
        print("\n" + "="*60)
        print(f"Running Random ICL Trial {trial_idx + 1}/{NUM_TRIALS} (seed={SEEDS[trial_idx]})")
        print("="*60)

        # Set seed
        random.seed(SEEDS[trial_idx])
        np.random.seed(SEEDS[trial_idx])

        # Create demonstrations with RANDOM labels
        random_demos = []
        random_label_count = {0: 0, 1: 0}
        correct_count = 0

        for item in train_data:
            random_label = random.choice([0, 1])
            random_demos.append({
                'uid': item['uid'],
                'prompt': item['prompt'],
                'label': random_label
            })
            random_label_count[random_label] += 1
            if random_label == item['vanilla_label']:
                correct_count += 1

        random_acc_train = correct_count / len(train_data)
        print(f"Random label distribution: {random_label_count}")
        print(f"Random labels match ground truth: {random_acc_train*100:.2f}% (expected ~50%)")

        # Evaluate with random labels
        acc_random, _ = evaluate_on_test(api, BASE_MODEL, random_demos, test_data)
        print(f"Test Accuracy: {acc_random * 100:.2f}%")

        # Save trial result
        all_results['random_icl_trials'].append({
            'seed': SEEDS[trial_idx],
            'accuracy': acc_random,
            'train_label_accuracy': random_acc_train
        })

    # Compute statistics
    accuracies = [trial['accuracy'] for trial in all_results['random_icl_trials']]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Random ICL (3 trials)")
    print("="*60)
    for i, trial in enumerate(all_results['random_icl_trials']):
        print(f"Trial {i+1} (seed={trial['seed']}): {trial['accuracy']*100:.2f}%")
    print(f"\nMean: {mean_acc*100:.2f}%")
    print(f"Std:  {std_acc*100:.2f}%")
    print(f"Result: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # Save results
    with open('results_random_icl_multi.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results_random_icl_multi.json")

if __name__ == "__main__":
    main()
