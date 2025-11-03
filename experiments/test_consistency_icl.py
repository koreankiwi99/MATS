"""
Test ICL with Logical Consistency Fixing (No ICM iteration)

Approach:
1. Add consistency_key from ground truth (defines logical structure)
2. Start with random labels
3. Detect inconsistencies using consistency_key
4. Fix inconsistencies with LLM
5. Use consistency-fixed labels for ICL
"""

import sys
import random
import numpy as np
import json
from copy import deepcopy

sys.path.insert(0, 'src')

from api_utils import HyperbolicAPI, BASE_MODEL
from data_utils import load_truthfulqa
from evaluation import evaluate_on_test


def add_consistency_keys(data):
    """
    Add consistency_key field based on ground truth labels
    Following original paper's approach for TruthfulQA
    """
    for item in data:
        # Store original label as vanilla_label
        item['vanilla_label'] = item['label']
        # Add consistency_key: 'A' if true, 'B' if false
        item['consistency_key'] = 'A' if item['label'] == 1 else 'B'
    return data


def pick_inconsistent_claims(data):
    """
    Detect logical inconsistencies (adapted from original repo)

    Two types:
    1. Implication: Same consistency_key but different assigned labels
    2. Contradiction: Different consistency_keys but both assigned label=1
    """
    # Group by consistency_id
    consistency_groups = {}
    for item in data:
        cid = item['consistency_id']
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(item)

    inconsistent_pairs = []

    for group in consistency_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # TYPE 1: Implication violation
                # Same consistency_key should have same label
                if (group[i]['consistency_key'] == group[j]['consistency_key']) and \
                   (group[i]['assigned_label'] != group[j]['assigned_label']):
                    inconsistent_pairs.append({
                        'claim_1': group[i],
                        'claim_2': group[j],
                        'type': 'implication',
                        'consistency_id': cid
                    })

                # TYPE 2: Contradiction
                # Different consistency_keys can't both be true
                elif (group[i]['consistency_key'] != group[j]['consistency_key']) and \
                     (group[i]['assigned_label'] == 1 and group[j]['assigned_label'] == 1):
                    inconsistent_pairs.append({
                        'claim_1': group[i],
                        'claim_2': group[j],
                        'type': 'contradiction',
                        'consistency_id': cid
                    })

    return inconsistent_pairs


def simple_consistency_fix(data):
    """
    Simple heuristic-based consistency fixing
    For each inconsistent group:
    - Use majority vote within consistency_key groups
    - Ensure at most one consistency_key group has label=1
    """
    # Group by consistency_id
    consistency_groups = {}
    for item in data:
        cid = item['consistency_id']
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(item)

    for cid, group in consistency_groups.items():
        # Sub-group by consistency_key
        key_groups = {}
        for item in group:
            key = item['consistency_key']
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(item)

        # For each consistency_key, use majority vote
        key_votes = {}
        for key, items in key_groups.items():
            votes = [item['assigned_label'] for item in items]
            majority = 1 if sum(votes) > len(votes) / 2 else 0
            key_votes[key] = majority

        # If multiple keys have label=1, keep only the one with most votes
        keys_with_one = [k for k, v in key_votes.items() if v == 1]
        if len(keys_with_one) > 1:
            # Count total votes for each key
            vote_counts = {}
            for key in keys_with_one:
                vote_counts[key] = sum([item['assigned_label'] for item in key_groups[key]])
            # Keep only the key with most votes
            best_key = max(vote_counts.items(), key=lambda x: x[1])[0]
            for key in keys_with_one:
                if key != best_key:
                    key_votes[key] = 0

        # Apply fixed labels
        for key, fixed_label in key_votes.items():
            for item in key_groups[key]:
                item['assigned_label'] = fixed_label

    return data


def main():
    # Multi-trial configuration
    NUM_TRIALS = 3
    SEEDS = [42, 123, 456]

    # Initialize API
    api = HyperbolicAPI()

    # Load data
    train_data = load_truthfulqa('data', split='train')
    test_data = load_truthfulqa('data', split='test')

    # Add consistency keys from ground truth
    train_data = add_consistency_keys(train_data)

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    all_results = {'consistency_icl_trials': []}

    # Run 3 trials
    for trial_idx in range(NUM_TRIALS):
        print("\n" + "="*60)
        print(f"Running Consistency ICL Trial {trial_idx + 1}/{NUM_TRIALS} (seed={SEEDS[trial_idx]})")
        print("="*60)

        # Set seed
        random.seed(SEEDS[trial_idx])
        np.random.seed(SEEDS[trial_idx])

        # Assign random labels
        train_with_random = deepcopy(train_data)
        for item in train_with_random:
            item['assigned_label'] = random.choice([0, 1])

        # Check initial accuracy and inconsistencies
        initial_acc = np.mean([
            item['assigned_label'] == item['vanilla_label']
            for item in train_with_random
        ])
        initial_inconsistencies = pick_inconsistent_claims(train_with_random)

        print(f"Initial random labels:")
        print(f"  Accuracy: {initial_acc*100:.2f}% (vs ground truth)")
        print(f"  Inconsistencies: {len(initial_inconsistencies)}")

        # Fix inconsistencies
        if len(initial_inconsistencies) > 0:
            train_fixed = simple_consistency_fix(train_with_random)

            # Check after fixing
            fixed_acc = np.mean([
                item['assigned_label'] == item['vanilla_label']
                for item in train_fixed
            ])
            fixed_inconsistencies = pick_inconsistent_claims(train_fixed)

            print(f"After consistency fixing:")
            print(f"  Accuracy: {fixed_acc*100:.2f}% (vs ground truth)")
            print(f"  Inconsistencies: {len(fixed_inconsistencies)}")
        else:
            train_fixed = train_with_random
            fixed_acc = initial_acc

        # Create demonstrations with fixed labels
        consistency_demos = []
        for item in train_fixed:
            consistency_demos.append({
                'uid': item.get('uid', len(consistency_demos)),
                'prompt': item['prompt'],
                'label': item['assigned_label']
            })

        # Evaluate with consistency-fixed labels
        acc_consistency, _ = evaluate_on_test(api, BASE_MODEL, consistency_demos, test_data)
        print(f"Test Accuracy: {acc_consistency * 100:.2f}%")

        # Save trial result
        all_results['consistency_icl_trials'].append({
            'seed': SEEDS[trial_idx],
            'accuracy': acc_consistency,
            'train_label_accuracy_initial': initial_acc,
            'train_label_accuracy_fixed': fixed_acc,
            'inconsistencies_initial': len(initial_inconsistencies),
            'inconsistencies_fixed': len(fixed_inconsistencies) if len(initial_inconsistencies) > 0 else 0
        })

    # Compute statistics
    accuracies = [trial['accuracy'] for trial in all_results['consistency_icl_trials']]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Consistency ICL (3 trials)")
    print("="*60)
    for i, trial in enumerate(all_results['consistency_icl_trials']):
        print(f"Trial {i+1} (seed={trial['seed']}): {trial['accuracy']*100:.2f}%")
        print(f"  Initial inconsistencies: {trial['inconsistencies_initial']}")
        print(f"  After fixing: {trial['inconsistencies_fixed']}")
    print(f"\nMean: {mean_acc*100:.2f}%")
    print(f"Std:  {std_acc*100:.2f}%")
    print(f"Result: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # Save results
    with open('results_consistency_icl.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results_consistency_icl.json")


if __name__ == "__main__":
    main()
