"""
GRPO Training with a Reward Model

This example shows how to use GRPO with a trained reward model
instead of rule-based reward functions. This is closer to
traditional RLHF pipelines.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List


class RewardModelScorer:
    """
    Wrapper class that uses a reward model to score completions.
    The reward model should be a sequence classification model that
    outputs a single score.
    """

    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = None,
    ):
        """
        Initialize the reward model.

        Args:
            model_name: HuggingFace model name for the reward model
            device: Device to run the model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def __call__(
        self,
        completions: List[str],
        prompts: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Score completions using the reward model.

        Args:
            completions: List of model completions to score
            prompts: List of original prompts (optional, for context)

        Returns:
            List of reward scores
        """
        rewards = []

        # Combine prompts with completions if available
        if prompts is not None:
            texts = [f"{p}\n{c}" for p, c in zip(prompts, completions)]
        else:
            texts = completions

        # Score in batches for efficiency
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get the reward scores (logits)
                scores = outputs.logits.squeeze(-1).cpu().tolist()

                # Handle single item case
                if isinstance(scores, float):
                    scores = [scores]

                rewards.extend(scores)

        return rewards


class MultiRewardCombiner:
    """
    Combines multiple reward functions/models with configurable weights.
    Useful for multi-objective RL training.
    """

    def __init__(self, reward_funcs: List[callable], weights: List[float] = None):
        """
        Args:
            reward_funcs: List of reward functions
            weights: Weights for each reward function (defaults to equal weights)
        """
        self.reward_funcs = reward_funcs
        self.weights = weights or [1.0 / len(reward_funcs)] * len(reward_funcs)

        assert len(self.reward_funcs) == len(self.weights), \
            "Number of reward functions must match number of weights"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Compute weighted combination of rewards."""
        all_rewards = []
        for func, weight in zip(self.reward_funcs, self.weights):
            rewards = func(completions, **kwargs)
            all_rewards.append([r * weight for r in rewards])

        # Sum up weighted rewards
        combined = [sum(r) for r in zip(*all_rewards)]
        return combined


def length_penalty_reward(completions: List[str], **kwargs) -> List[float]:
    """Penalize very short or very long responses."""
    rewards = []
    for c in completions:
        words = len(c.split())
        if words < 10:
            reward = -0.5
        elif words > 200:
            reward = -0.3
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


def safety_reward(completions: List[str], **kwargs) -> List[float]:
    """Simple safety filter - penalize potentially harmful content."""
    unsafe_patterns = [
        "kill", "harm", "attack", "hack into", "steal",
        "illegal", "dangerous", "exploit"
    ]
    rewards = []
    for c in completions:
        c_lower = c.lower()
        penalty = sum(0.2 for p in unsafe_patterns if p in c_lower)
        rewards.append(-min(penalty, 1.0))
    return rewards


def train_with_reward_model():
    """
    Train a small LLM using GRPO with a reward model.
    """
    # Policy model (the model we're training)
    policy_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 60)
    print("GRPO Training with Reward Model")
    print("=" * 60)

    # Load policy model
    print(f"\nLoading policy model: {policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize reward model
    # Note: This particular reward model requires significant memory
    # For smaller setups, use rule-based rewards or a smaller reward model
    try:
        reward_scorer = RewardModelScorer(
            model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
        )
        # Combine reward model with safety checks
        combined_reward = MultiRewardCombiner(
            reward_funcs=[reward_scorer, length_penalty_reward, safety_reward],
            weights=[0.7, 0.15, 0.15]
        )
        print("Using reward model + safety + length penalty")
    except Exception as e:
        print(f"Could not load reward model: {e}")
        print("Falling back to rule-based rewards")
        combined_reward = MultiRewardCombiner(
            reward_funcs=[length_penalty_reward, safety_reward],
            weights=[0.5, 0.5]
        )

    # Prepare dataset
    prompts = [
        "Explain the concept of artificial intelligence.",
        "What are the benefits of renewable energy?",
        "How can I improve my communication skills?",
        "What is the importance of biodiversity?",
        "Explain how a computer processor works.",
        "What are effective study techniques?",
        "How does encryption keep data secure?",
        "What causes climate change?",
        "How do neural networks learn?",
        "What is the scientific method?",
    ]

    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    dataset = Dataset.from_dict({"prompt": formatted_prompts})

    # GRPO config
    config = GRPOConfig(
        output_dir="./grpo_reward_model_output",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-6,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=128,
        beta=0.05,  # Lower KL penalty for more exploration
        logging_steps=1,
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=combined_reward,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    trainer.save_model("./grpo_reward_model_output")
    tokenizer.save_pretrained("./grpo_reward_model_output")
    print("\nTraining complete!")

    return trainer


if __name__ == "__main__":
    train_with_reward_model()
