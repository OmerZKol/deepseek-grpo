"""
GRPO (Group Relative Policy Optimization) Training Script for Small LLMs

This script implements GRPO using Hugging Face's TRL library.
GRPO is a reinforcement learning algorithm that:
1. Generates multiple completions per prompt
2. Scores them with a reward function
3. Uses relative rankings within each group to compute advantages
4. Updates the policy using these group-relative advantages

Suitable for small LLMs (< 1B parameters) on consumer hardware.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import re
from typing import List, Dict, Any


# ============================================================================
# Configuration
# ============================================================================

# Model configuration - using small models suitable for consumer hardware
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B parameter model
# Alternative small models:
# - "microsoft/phi-2" (2.7B)
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B)
# - "Qwen/Qwen2.5-1.5B-Instruct" (1.5B)
# - "HuggingFaceTB/SmolLM-135M-Instruct" (135M)

# Training hyperparameters
OUTPUT_DIR = "./grpo_output"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 128
NUM_GENERATIONS = 4  # Number of completions per prompt for GRPO
BETA = 0.1  # KL penalty coefficient

# LoRA configuration for efficient fine-tuning
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ============================================================================
# Reward Functions
# ============================================================================

def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion follows a specific format.
    Example: Rewards responses that contain properly formatted answers.
    """
    rewards = []
    for completion in completions:
        reward = 0.0
        # Reward for having a clear answer structure
        if any(marker in completion.lower() for marker in ["answer:", "result:", "solution:"]):
            reward += 0.5
        # Reward for being concise (not too long)
        if len(completion.split()) < 100:
            reward += 0.3
        # Reward for proper punctuation at end
        if completion.strip().endswith((".", "!", "?")):
            reward += 0.2
        rewards.append(reward)
    return rewards


def reward_math_correctness(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Reward function for math problems.
    Extracts the final numerical answer and checks correctness.

    This is a simplified example - in practice you'd parse the expected answer
    from a dataset or use a more sophisticated evaluation.
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        reward = 0.0
        # Extract numbers from completion
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        if numbers:
            # Reward for providing a numerical answer
            reward += 0.3
            # Reward for showing work (multiple numbers = steps)
            if len(numbers) > 1:
                reward += 0.2
        # Reward for mathematical notation
        if any(op in completion for op in ['+', '-', '*', '/', '=', 'x']):
            reward += 0.2
        # Penalty for very short responses
        if len(completion.split()) < 5:
            reward -= 0.3
        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def reward_helpfulness(completions: List[str], **kwargs) -> List[float]:
    """
    General helpfulness reward function.
    Rewards clear, structured, and informative responses.
    """
    rewards = []
    for completion in completions:
        reward = 0.0
        words = completion.split()

        # Reward appropriate length (not too short, not too long)
        if 20 <= len(words) <= 150:
            reward += 0.3
        elif 10 <= len(words) < 20:
            reward += 0.15

        # Reward structured responses
        if any(marker in completion for marker in ['\n-', '\n*', '\n1.', '\n2.']):
            reward += 0.2

        # Reward explanatory phrases
        explanatory = ['because', 'therefore', 'this means', 'for example', 'in other words']
        if any(phrase in completion.lower() for phrase in explanatory):
            reward += 0.2

        # Penalty for repetition
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio > 0.7:
                reward += 0.2
            elif repetition_ratio < 0.4:
                reward -= 0.2

        # Reward for proper ending
        if completion.strip() and completion.strip()[-1] in '.!?':
            reward += 0.1

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def reward_code_quality(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function for code generation tasks.
    Rewards syntactically valid and well-structured code.
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # Check for code block markers
        if '```' in completion:
            reward += 0.2

        # Check for common programming constructs
        code_patterns = [
            r'\bdef\s+\w+\s*\(',  # Python function
            r'\bclass\s+\w+',     # Class definition
            r'\bif\s+.*:',        # If statement
            r'\bfor\s+\w+\s+in',  # For loop
            r'\breturn\b',        # Return statement
            r'\bimport\s+\w+',    # Import
        ]
        for pattern in code_patterns:
            if re.search(pattern, completion):
                reward += 0.1

        # Check balanced brackets/parens (basic syntax check)
        open_parens = completion.count('(')
        close_parens = completion.count(')')
        open_brackets = completion.count('[')
        close_brackets = completion.count(']')
        open_braces = completion.count('{')
        close_braces = completion.count('}')

        if open_parens == close_parens and open_brackets == close_brackets and open_braces == close_braces:
            reward += 0.2

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


# ============================================================================
# Dataset Preparation
# ============================================================================

def create_sample_dataset() -> Dataset:
    """
    Creates a sample dataset for demonstration.
    In practice, you would load a real dataset.
    """
    prompts = [
        "Explain what machine learning is in simple terms.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is the capital of France and why is it famous?",
        "Explain the water cycle.",
        "What causes the seasons to change?",
        "How do computers store information?",
        "What is gravity and how does it work?",
        "Why is the sky blue?",
        "How do plants grow?",
        "What is the difference between weather and climate?",
        "How does the internet work?",
        "What causes earthquakes?",
        "How do vaccines work?",
        "What is renewable energy?",
    ]

    return Dataset.from_dict({"prompt": prompts})


def load_and_prepare_dataset(dataset_name: str = None, split: str = "train") -> Dataset:
    """
    Load and prepare a dataset for GRPO training.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "openai/gsm8k")
        split: Dataset split to use

    Returns:
        Dataset with 'prompt' column
    """
    if dataset_name is None:
        print("Using sample dataset for demonstration...")
        return create_sample_dataset()

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    # Handle common dataset formats
    if "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "prompt")
    elif "instruction" in dataset.column_names:
        dataset = dataset.rename_column("instruction", "prompt")
    elif "input" in dataset.column_names:
        dataset = dataset.rename_column("input", "prompt")

    # Keep only the prompt column for GRPO
    if "prompt" in dataset.column_names:
        dataset = dataset.select_columns(["prompt"])

    return dataset


def format_prompt(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Format prompts using the model's chat template.
    """
    messages = [
        {"role": "user", "content": example["prompt"]}
    ]

    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted = f"User: {example['prompt']}\nAssistant:"

    return {"prompt": formatted}


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(model_name: str, use_lora: bool = True):
    """
    Load and configure the model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        use_lora: Whether to use LoRA for efficient fine-tuning

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model setup
    # Use bfloat16 for efficiency on supported hardware
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Check available device
    if torch.cuda.is_available():
        device_map = "auto"
        print(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
    elif torch.backends.mps.is_available():
        device_map = "mps"
        torch_dtype = torch.float32  # MPS works better with float32
        print("Using Apple MPS")
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        print("Using CPU (training will be slow)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Apply LoRA if enabled
    if use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def train_grpo(
    model_name: str = MODEL_NAME,
    dataset_name: str = None,
    reward_function: callable = reward_helpfulness,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = NUM_TRAIN_EPOCHS,
    use_lora: bool = USE_LORA,
):
    """
    Main training function for GRPO.

    Args:
        model_name: HuggingFace model identifier
        dataset_name: Dataset to use (None for sample dataset)
        reward_function: Function to compute rewards for completions
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        use_lora: Whether to use LoRA
    """
    print("=" * 60)
    print("GRPO Training Script")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_lora)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(dataset_name)

    # Format prompts with chat template
    dataset = dataset.map(
        lambda x: format_prompt(x, tokenizer),
        desc="Formatting prompts"
    )

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample prompt:\n{dataset[0]['prompt'][:200]}...")

    # GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir=output_dir,

        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,

        # GRPO-specific parameters
        num_generations=NUM_GENERATIONS,  # Number of completions per prompt
        beta=BETA,  # KL divergence coefficient

        # Generation parameters
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,

        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Logging
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,

        # Mixed precision
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,

        # Misc
        seed=42,
        report_to="none",  # Set to "wandb" for Weights & Biases logging
    )

    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    print("\nStarting GRPO training...")
    print(f"  - Model: {model_name}")
    print(f"  - Generations per prompt: {NUM_GENERATIONS}")
    print(f"  - Beta (KL coefficient): {BETA}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print()

    # Train
    trainer.train()

    # Save the final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTraining complete!")
    return trainer


# ============================================================================
# Inference
# ============================================================================

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Generate a response from the trained model.
    """
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    return response


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training for Small LLMs")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (uses sample data if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for saved model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="helpfulness",
        choices=["helpfulness", "format", "math", "code"],
        help="Reward function to use"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (use full fine-tuning)"
    )

    args = parser.parse_args()

    # Select reward function
    reward_functions = {
        "helpfulness": reward_helpfulness,
        "format": reward_format_compliance,
        "math": reward_math_correctness,
        "code": reward_code_quality,
    }
    reward_fn = reward_functions[args.reward]

    # Run training
    trainer = train_grpo(
        model_name=args.model,
        dataset_name=args.dataset,
        reward_function=reward_fn,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        use_lora=not args.no_lora,
    )

    # Test the trained model
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)

    test_prompts = [
        "What is the meaning of life?",
        "How can I learn programming?",
        "Explain quantum computing simply.",
    ]

    model = trainer.model
    tokenizer = trainer.processing_class

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}")
