# DeepSeek GRPO Training

Basic implementation of GRPO (Group Relative Policy Optimisation) for training small language models using reinforcement learning with rule-based and verifiable rewards. Based on the approach from the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300), further popularised by the success of [DeepSeekR1 Paper](https://arxiv.org/abs/2501.12948).

## What is GRPO?

GRPO is a reinforcement learning algorithm that:
1. Generates multiple completions per prompt
2. Scores them with reward functions
3. Uses relative rankings within each group to compute advantages
4. Updates the policy using these group-relative advantages

This implementation is optimised for small LLMs (< 1B parameters) on consumer hardware using LoRA fine-tuning.

## Repository Structure

```
src/
├── train_minimal.py                 # Main training script with rule-based rewards
└── train_grpo_with_reward_model.py  # Training with more complex reward models

notebooks/
└── compare_models.ipynb             # Compare base model vs trained model performance
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA (optional but recommended)

## Usage

### Math Training (Recommended for Getting Started)

Train on the GSM8K dataset with format and accuracy rewards:

```bash
# Full training (complete dataset, 1 epoch)
python src/train_minimal.py

# Test mode (50 examples, 100 steps - for quick iteration)
python src/train_minimal.py --test
```

This script:
- Uses the GSM8K (grade school math) dataset
- Trains the model to output answers in `<answer>...</answer>` format
- Rewards both format compliance and numerical correctness
- Optimised for memory efficiency with gradient checkpointing
- Includes TensorBoard logging

**Key Features:**
- **Reward Functions:** Format compliance (0.5) + accuracy (1.0)
- **Output:** Saves trained model to `./grpo-math/`
- **Monitoring:** View training progress with `tensorboard --logdir logs`

**Training Modes:**
- **Full Training** (default): 7,473 examples, 1 epoch, saves every 500 steps
- **Test Mode** (`--test`): 100 examples, 100 steps, saves every 50 steps

### Advanced: Training with Reward Models

Use a pre-trained reward model for more sophisticated reward shaping:

```bash
python src/train_grpo_with_reward_model.py
```

This script demonstrates:
- Using OpenAssistant's DeBERTa reward model
- Combining multiple reward signals (learned model + length penalty + safety)
- Multi-objective reinforcement learning with weighted rewards
- Fallback to rule-based rewards if the model is unavailable

**Note:** Requires additional GPU memory for the reward model (~2-3GB).

### Model Comparison

Use the notebook [compare_models.ipynb](notebooks/compare_models.ipynb) to evaluate the trained model against the base model:

## Configuration

### train_minimal.py
Edit constants at the top of [train_minimal.py](src/train_minimal.py) to customize:
- `MODEL_ID` - Base model (default: Qwen2.5-0.5B-Instruct)
- `NUM_GENERATIONS` - Group size for GRPO (default: 10)
- `BATCH_SIZE` - Training batch size (default: 5)

Command line options:
- `--test` - Enable test mode for quick iteration

### train_grpo_with_reward_model.py
Edit constants in [train_grpo_with_reward_model.py](src/train_grpo_with_reward_model.py):
- Policy model selection
- Reward model choice
- Reward function weights
- Training hyperparameters

## Hardware Requirements

- **Minimum:** 16GB RAM, CPU only (slow)
- **Recommended:** 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
