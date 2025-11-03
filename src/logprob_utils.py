"""
Logprob extraction utilities
Adapted from Unsupervised-Elicitation/src/model_querying/solution_extraction.py
"""

import math
import logging

logger = logging.getLogger(__name__)


def get_yes_no(x):
    x = x.lower()
    y = "true" in x
    n = "false" in x
    if y == n:  # Both or neither
        return None
    return y


def get_yes_no_diff_logprobs(logprobs):
    """
    Extract log P(True) - log P(False) from logprobs dictionary
    Aggregates probabilities over all tokens containing "true" or "false"
    """
    eps = 1e-5
    prob_sums = {False: eps, True: eps}

    for k, v in logprobs.items():
        o = get_yes_no(k)
        if o is None:
            continue
        prob_sums[o] += math.exp(v)

    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])


def extract_claim_logprobs(response):
    """
    Extract score from API response and add to response dict
    """
    response_dict = response.copy() if isinstance(response, dict) else {}

    try:
        # For OpenAI-compatible API (Hyperbolic)
        if hasattr(response, 'choices'):
            logprobs_data = response.choices[0].logprobs.top_logprobs[0]
            # Convert to dict
            logprobs = {token: logprob for token, logprob in logprobs_data.items()}
        else:
            # Already a dict
            logprobs = response.get("response", {}).get("logprobs", [{}])[0]

        score = get_yes_no_diff_logprobs(logprobs)
        response_dict["score"] = score

    except Exception as e:
        logger.warning(f"Error extracting logprobs: {repr(e)}")
        response_dict["score"] = 0

    return response_dict
