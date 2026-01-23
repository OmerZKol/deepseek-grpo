"""
Shared utilities for GRPO training and evaluation.
"""

import re

# Regex pattern for extracting numbers (handles negatives, decimals, and commas)
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# Pattern for extracting content from <answer> tags
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def extract_last_number(text):
    """
    Extract the last number from a string.
    Handles negative numbers, decimals, and comma-separated thousands.

    Args:
        text: String that may contain numbers

    Returns:
        Float value of the last number found, or None if no number found
    """
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    # Take the last number (final answer is typically at the end)
    last_match = matches[-1]
    # Remove commas for parsing (e.g., "1,000" -> "1000")
    return float(last_match.replace(",", ""))


def extract_answer_from_tags(completion):
    """
    Extract the numerical answer from <answer>...</answer> tags.

    Args:
        completion: Model completion string

    Returns:
        Float value of the answer, or None if not found/parseable
    """
    match = ANSWER_TAG_PATTERN.search(completion)
    if not match:
        return None
    return extract_last_number(match.group(1).strip())


def extract_gold_answer(answer_text):
    """
    Extract gold answer from GSM8K format.
    GSM8K answers are formatted as "... #### <number>"

    Args:
        answer_text: The answer field from GSM8K dataset

    Returns:
        Float value of the gold answer, or None if not found
    """
    gold_str = answer_text.split("####")[-1].strip()
    return extract_last_number(gold_str)


def format_prompt(question):
    """
    Format a question as a chat prompt for the model.

    Args:
        question: The math question text

    Returns:
        Formatted prompt string with system and user messages
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful logic assistant. You must output your final answer "
        "wrapped in <answer> tags. Example: <answer>42</answer>.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_reward_func(completions, **kwargs):
    """
    Reward Structure: Checks if the completion uses the specific XML format.
    Regex: <answer> ... </answer>

    Args:
        completions: List of completion strings

    Returns:
        List of rewards (0.5 for correct format, 0.0 otherwise)
    """
    rewards = []
    for completion in completions:
        match = ANSWER_TAG_PATTERN.search(completion)
        rewards.append(0.5 if match else 0.0)
    return rewards


def accuracy_reward_func(completions, answer, **kwargs):
    """
    Reward Correctness: Extracts the last number from the completion and compares to ground truth.
    This is independent of format - it will find the last number anywhere in the output,
    regardless of whether it's in <answer> tags or not.

    Args:
        completions: List of completion strings
        answer: List of ground truth answers

    Returns:
        List of rewards (1.0 for correct, 0.0 otherwise)
    """
    rewards = []

    for completion, correct_answer in zip(completions, answer):
        try:
            pred_val = extract_last_number(completion)
            gold_val = extract_gold_answer(correct_answer)

            if pred_val is None or gold_val is None:
                rewards.append(0.0)
            elif abs(pred_val - gold_val) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards