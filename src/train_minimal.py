import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# 1. Setup Configuration
# ----------------------
# MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
OUTPUT_DIR = "grpo-math"

# GRPO specific parameters
NUM_GENERATIONS = 8  # Group size (G): How many outputs to generate per prompt
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 512
BATCH_SIZE = 4       # Depending on gpu VRAM, adjust this

# 2. Define Reward Functions
# --------------------------
# These functions take the model's output and the ground truth to calculate a score.

def format_reward_func(completions, **kwargs):
    """
    Reward Structure: Checks if the completion uses the specific XML format.
    Regex: <answer> ... </answer>
    """
    pattern = r"<answer>(.*?)</answer>"
    rewards = []
    
    for completion in completions:
        match = re.search(pattern, completion, flags=re.DOTALL)
        # Give a small reward for following the format instruction
        if match:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, answer, **kwargs):
    """
    Reward Correctness: Extracts the number from the XML and compares to ground truth.
    """
    pattern = r"<answer>(.*?)</answer>"
    rewards = []
    
    for completion, correct_answer in zip(completions, answer):
        match = re.search(pattern, completion, flags=re.DOTALL)
        
        # If no format match, they automatically fail accuracy too
        if not match:
            rewards.append(0.0)
            continue
            
        extracted_content = match.group(1).strip()
        
        try:
            # GSM8K answers often have the calculation logic. The final number is usually
            # at the end or we can try to parse the whole string. 
            # For this simple script, we assume the dataset 'answer' column is the numerical truth.
            # (In raw GSM8K, we often need to extract the number after "####")
            
            # Simple numeric cleaning
            pred_val = float(re.sub(r"[^\d\.]", "", extracted_content))
            
            # In GSM8K, the answer column is text like "The answer is 42 #### 42"
            # We extract the number after ####
            gold_val_str = correct_answer.split("####")[-1].strip()
            gold_val = float(re.sub(r"[^\d\.]", "", gold_val_str))
            
            if abs(pred_val - gold_val) < 1e-5:
                rewards.append(1.0) # Big reward for correct math!
            else:
                rewards.append(0.0)
        except Exception:
            # If parsing fails, 0 reward
            rewards.append(0.0)
            
    return rewards

# 3. Load and Prep Dataset
# ------------------------
def get_gsm8k_questions(data_split="train"):
    data = load_dataset('openai/gsm8k', 'main')[data_split]
    
    # We need to format the prompt to look like a chat interaction
    # so the model knows it is being instructed.
    prompts = []
    for example in data:
        # Structured system prompt
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful logic assistant. You must output your final answer "
            "wrapped in <answer> tags. Example: <answer>42</answer>.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{example['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)
        
    # Return as a dictionary with the prompt and the reference answer
    return {"prompt": prompts, "answer": data["answer"]}

dataset = load_dataset('openai/gsm8k', 'main')['train']
# We map the formatting function to add the "prompt" column
dataset = dataset.map(lambda x: {"prompt": (
    "<|im_start|>system\n"
    "You are a helpful logic assistant. You must output your final answer "
    "wrapped in <answer> tags. Example: <answer>42</answer>.<|im_end|>\n"
    "<|im_start|>user\n"
    f"{x['question']}<|im_end|>\n"
    "<|im_start|>assistant\n"
)})

# 4. Load Model and Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model in half-precision (bfloat16) to save memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 5. Configure LoRA (Parameter Efficient Fine-Tuning)
# ---------------------------------------------------
# Instead of training all 0.5B params, we train a tiny adapter.
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# 6. Initialize GRPO Trainer
# --------------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-6,           # Very low LR is key for RL stability
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS, # The G in GRPO
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_steps=100,               # Set short for testing (e.g., 100 steps)
    save_steps=50,
    gradient_accumulation_steps=4,
    report_to="none"             # Change to "wandb" if you want charts
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# 7. Start Training!
# ------------------
print("Starting GRPO training...")
trainer.train()

# Save the final adapter
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")