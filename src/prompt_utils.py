"""
Prompt creation utilities
Adapted from model_querying/prompt_creation.py and ICM.py
"""


def get_judge_prompt_fewshot(example, demonstrations):
    """
    Create few-shot prompt for ICL
    from prompt_creation.py/get_judge_prompt_fewshot()
    """
    prompt = ""

    # Add demonstrations
    for demo in demonstrations:
        prompt += demo['prompt']
        prompt += "True" if demo["label"] == 1 else "False"
        prompt += "\n\n"

    # Add target example (without answer)
    prompt += example['prompt']

    return prompt


def create_truthfulqa_prompt(question, choice):
    """
    Create TruthfulQA prompt template
    From ICM.py
    """
    template = """Question: {question}
Claim: {answer}
I think this claim is """

    return template.format(question=question, answer=choice)
