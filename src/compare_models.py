"""
Compare base model vs GRPO-trained model on GSM8K test set.

Usage:
    python src/compare_models.py
    python src/compare_models.py --num_samples 100  # Limit samples for quick test
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from utils import (
    ANSWER_TAG_PATTERN,
    extract_answer_from_tags,
    extract_gold_answer,
    format_prompt,
    format_reward_func,
    accuracy_reward_func,
)

# Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_MODEL_PATH = "grpo-math"
MAX_NEW_TOKENS = 768


def load_models(device):
    """Load base model and trained model."""
    print(f"Loading tokenizer from {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {BASE_MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    print(f"Loading trained model from {TRAINED_MODEL_PATH}...")
    # Load base model again for PEFT
    trained_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
    )
    trained_model = PeftModel.from_pretrained(trained_base, TRAINED_MODEL_PATH)
    trained_model.eval()

    return tokenizer, base_model, trained_model


def generate_answer(model, tokenizer, prompt, device):
    """Generate answer from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated


def evaluate_model(model, tokenizer, dataset, device, model_name):
    """Evaluate model on dataset and return metrics including rewards."""
    correct = 0
    format_correct = 0
    total = len(dataset)

    total_format_reward = 0.0
    total_accuracy_reward = 0.0

    results = []
    completions = []
    answers = []

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        prompt = format_prompt(example["question"])
        completion = generate_answer(model, tokenizer, prompt, device)

        # Check format - Use same regex pattern as training reward function
        has_format = ANSWER_TAG_PATTERN.search(completion) is not None
        if has_format:
            format_correct += 1

        # Check accuracy
        pred_answer = extract_answer_from_tags(completion)
        gold_answer = extract_gold_answer(example["answer"])

        is_correct = False
        if pred_answer is not None and gold_answer is not None:
            is_correct = abs(pred_answer - gold_answer) < 1e-5
            if is_correct:
                correct += 1

        results.append({
            "question": example["question"],
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "completion": completion,
            "is_correct": is_correct,
            "has_format": has_format,
        })

        # Store for batch reward calculation
        completions.append(completion)
        answers.append(example["answer"])

    # Calculate rewards using the same functions from training
    format_rewards = format_reward_func(completions)
    accuracy_rewards = accuracy_reward_func(completions, answers)

    total_format_reward = sum(format_rewards)
    total_accuracy_reward = sum(accuracy_rewards)

    return {
        "accuracy": correct / total * 100,
        "format_rate": format_correct / total * 100,
        "correct": correct,
        "format_correct": format_correct,
        "total": total,
        "avg_format_reward": total_format_reward / total,
        "avg_accuracy_reward": total_accuracy_reward / total,
        "total_format_reward": total_format_reward,
        "total_accuracy_reward": total_accuracy_reward,
        "avg_total_reward": (total_format_reward + total_accuracy_reward) / total,
        "results": results,
    }


def save_results(base_metrics, trained_metrics, output_file="comparison_results.json"):
    """Save comparison results to a JSON file."""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "base_model": {
            "model_id": BASE_MODEL_ID,
            "accuracy": base_metrics["accuracy"],
            "format_rate": base_metrics["format_rate"],
            "correct": base_metrics["correct"],
            "format_correct": base_metrics["format_correct"],
            "total": base_metrics["total"],
            "avg_format_reward": base_metrics["avg_format_reward"],
            "avg_accuracy_reward": base_metrics["avg_accuracy_reward"],
            "avg_total_reward": base_metrics["avg_total_reward"],
            "total_format_reward": base_metrics["total_format_reward"],
            "total_accuracy_reward": base_metrics["total_accuracy_reward"],
        },
        "trained_model": {
            "model_id": TRAINED_MODEL_PATH,
            "accuracy": trained_metrics["accuracy"],
            "format_rate": trained_metrics["format_rate"],
            "correct": trained_metrics["correct"],
            "format_correct": trained_metrics["format_correct"],
            "total": trained_metrics["total"],
            "avg_format_reward": trained_metrics["avg_format_reward"],
            "avg_accuracy_reward": trained_metrics["avg_accuracy_reward"],
            "avg_total_reward": trained_metrics["avg_total_reward"],
            "total_format_reward": trained_metrics["total_format_reward"],
            "total_accuracy_reward": trained_metrics["total_accuracy_reward"],
        },
        "improvements": {
            "accuracy": trained_metrics["accuracy"] - base_metrics["accuracy"],
            "format_rate": trained_metrics["format_rate"] - base_metrics["format_rate"],
            "avg_format_reward": trained_metrics["avg_format_reward"] - base_metrics["avg_format_reward"],
            "avg_accuracy_reward": trained_metrics["avg_accuracy_reward"] - base_metrics["avg_accuracy_reward"],
            "avg_total_reward": trained_metrics["avg_total_reward"] - base_metrics["avg_total_reward"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_path.absolute()}")


def print_examples(base_results, trained_results, num_examples=3):
    """Print example comparisons."""
    print("\n" + "=" * 60)
    print("EXAMPLE COMPARISONS")
    print("=" * 60)

    for i in range(min(num_examples, len(base_results))):
        base = base_results[i]
        trained = trained_results[i]

        print(f"\n--- Example {i+1} ---")
        print(f"Question: {base['question'][:100]}...")
        print(f"Gold Answer: {base['gold_answer']}")
        print(f"\nBase Model:")
        print(f"  Answer: {base['pred_answer']} {'✓' if base['is_correct'] else '✗'}")
        print(f"  Response: {base['completion'][:200]}...")
        print(f"\nTrained Model:")
        print(f"  Answer: {trained['pred_answer']} {'✓' if trained['is_correct'] else '✗'}")
        print(f"  Response: {trained['completion'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Compare base and trained models")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=3,
        help="Number of example comparisons to show (default: 3)",
    )
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test dataset
    print("Loading GSM8K test set...")
    test_data = load_dataset("openai/gsm8k", "main")["test"]

    if args.num_samples:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
        print(f"Using {len(test_data)} samples")
    else:
        print(f"Using full test set: {len(test_data)} samples")

    # Load models
    tokenizer, base_model, trained_model = load_models(device)

    # Evaluate both models
    print("\n" + "=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)

    base_metrics = evaluate_model(base_model, tokenizer, test_data, device, "Base Model")
    trained_metrics = evaluate_model(trained_model, tokenizer, test_data, device, "Trained Model")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Base Model':>15} {'Trained Model':>15} {'Improvement':>15}")
    print("-" * 70)

    acc_diff = trained_metrics["accuracy"] - base_metrics["accuracy"]
    fmt_diff = trained_metrics["format_rate"] - base_metrics["format_rate"]
    format_reward_diff = trained_metrics["avg_format_reward"] - base_metrics["avg_format_reward"]
    accuracy_reward_diff = trained_metrics["avg_accuracy_reward"] - base_metrics["avg_accuracy_reward"]
    total_reward_diff = trained_metrics["avg_total_reward"] - base_metrics["avg_total_reward"]

    print(f"{'Accuracy':<25} {base_metrics['accuracy']:>14.1f}% {trained_metrics['accuracy']:>14.1f}% {acc_diff:>+14.1f}%")
    print(f"{'Format Compliance':<25} {base_metrics['format_rate']:>14.1f}% {trained_metrics['format_rate']:>14.1f}% {fmt_diff:>+14.1f}%")
    print(f"{'Correct / Total':<25} {base_metrics['correct']:>7}/{base_metrics['total']:<6} {trained_metrics['correct']:>7}/{trained_metrics['total']:<6}")

    print(f"\n{'Reward Metrics':<25} {'Base Model':>15} {'Trained Model':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Avg Format Reward':<25} {base_metrics['avg_format_reward']:>15.3f} {trained_metrics['avg_format_reward']:>15.3f} {format_reward_diff:>+15.3f}")
    print(f"{'Avg Accuracy Reward':<25} {base_metrics['avg_accuracy_reward']:>15.3f} {trained_metrics['avg_accuracy_reward']:>15.3f} {accuracy_reward_diff:>+15.3f}")
    print(f"{'Avg Total Reward':<25} {base_metrics['avg_total_reward']:>15.3f} {trained_metrics['avg_total_reward']:>15.3f} {total_reward_diff:>+15.3f}")

    # Save results to JSON
    save_results(base_metrics, trained_metrics, "comparison_results.json")

    # Print examples
    if args.show_examples > 0:
        print_examples(
            base_metrics["results"],
            trained_metrics["results"],
            args.show_examples,
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()