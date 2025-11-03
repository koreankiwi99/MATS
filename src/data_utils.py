"""
Data loading utilities
Adapted from ICM.py
"""

import json
from prompt_utils import create_truthfulqa_prompt


def load_truthfulqa(data_dir, split='train'):
    """
    Load TruthfulQA dataset
    Adapted from ICM.py load_data()
    """
    filepath = f"{data_dir}/truthfulqa_{split}.json"

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Format each example
    for i, item in enumerate(data):
        # Create prompt using template from ICM.py
        item['prompt'] = create_truthfulqa_prompt(item['question'], item['choice'])
        item['uid'] = i
        item['vanilla_label'] = item['label']  # Store ground truth

    return data