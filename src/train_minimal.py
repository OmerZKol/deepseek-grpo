import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from utils import (
    format_prompt,
    format_reward_func,
    accuracy_reward_func,
)

# 1. Setup Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
OUTPUT_DIR = "grpo-math-model"

# GRPO specific parameters
NUM_GENERATIONS = 12  # Group size (G): How many outputs to generate per prompt
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 768
BATCH_SIZE = 8       # Increased to better utilize GPU


# 2. Load and Prep Dataset
def get_gsm8k_questions(data_split="train"):
    data = load_dataset('openai/gsm8k', 'main')[data_split]
    prompts = [format_prompt(example['question']) for example in data]
    return {"prompt": prompts, "answer": data["answer"]}


def prepare_dataset(test_mode=False):
    """
    Prepare the GSM8K dataset for training.

    Args:
        test_mode: If True, only use a small subset for quick testing

    Returns:
        Formatted dataset ready for training
    """
    dataset = load_dataset('openai/gsm8k', 'main')['train']

    if test_mode:
        dataset = dataset.select(range(min(100, len(dataset))))
        print(f"[TEST MODE] Using {len(dataset)} examples")
    else:
        print(f"[FULL TRAINING] Using {len(dataset)} examples")

    dataset = dataset.map(
        lambda x: {"prompt": format_prompt(x['question'])},
        num_proc=4  # Parallel preprocessing
    )
    return dataset

def main():
    """Main training function with command line argument support."""
    # Enable TF32 for faster training on Ampere+ GPUs (A100, RTX 3090/4090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GRPO Training for Math Problem Solving")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (small subset, limited steps for quick testing)"
    )
    args = parser.parse_args()

    # 2. Load Dataset
    dataset = prepare_dataset(test_mode=args.test)

    # 3. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in half-precision (bfloat16) to save memory
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # flash attention for speed
    )

    # Note: torch.compile() is incompatible with PEFT/GRPO trainer
    # Flash Attention 2 + gradient checkpointing provide significant speedup

    # 4. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # 5. Initialize GRPO Trainer
    # Determine training duration based on mode
    if args.test:
        max_steps = 100  # Quick test run
        save_steps = 50
        print("[TEST MODE] Training for 100 steps")
    else:
        max_steps = -1  # Train on full dataset (use num_train_epochs instead)
        save_steps = 500
        print("[FULL TRAINING] Training on complete dataset")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1000,
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS, # The G in GRPO
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=max_steps,
        num_train_epochs=1 if max_steps == -1 else None,  # Use epochs for full training
        save_steps=save_steps,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        gradient_checkpointing=True,    # Enable to save memory and allow larger batches
        dataloader_num_workers=8,       # Increased workers for faster data loading
        dataloader_pin_memory=True,     # Pin memory for faster transfers to GPU
        dataloader_prefetch_factor=4,   # Prefetch batches to reduce waiting
        report_to="tensorboard",        # Save training metrics to TensorBoard
        logging_dir=f"logs",  # TensorBoard log directory
    )

    # Generation kwargs for faster sampling during GRPO rollouts
    generation_kwargs = {
        "max_new_tokens": MAX_COMPLETION_LENGTH,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stop_strings": ["</answer>"],
    }

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        generation_kwargs=generation_kwargs,  # Pass optimized generation config
    )

    # 6. Start Training!
    print("Starting GRPO training...")
    trainer.train()

    # Save the final adapter
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()