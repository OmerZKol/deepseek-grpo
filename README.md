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
├── train_grpo_with_reward_model.py  # Training with more complex reward models
├── compare_models.py                # Evaluate trained vs base model on test set
└── utils.py                         # Shared utilities (answer extraction, prompts)

notebooks/
└── compare_models.ipynb             # Interactive model comparison notebook
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

Evaluate trained model vs base model on the GSM8K test set:

```bash
# Full evaluation (1,319 test samples)
python src/compare_models.py

# Quick test with subset
python src/compare_models.py --num_samples 50

# Specify output file
python src/compare_models.py --output results/my_comparison.json
```

The comparison script:
- Evaluates both base and trained models on the test set
- Calculates accuracy and format compliance rates
- **Computes reward metrics** (format reward: 0.5, accuracy reward: 1.0)
- Shows average rewards per sample for each model
- **Saves detailed results to JSON** including all metrics and improvements
- Displays example comparisons to see model outputs side-by-side

Output includes:
- Accuracy and format compliance percentages
- Average format reward (0.0-0.5 per sample)
- Average accuracy reward (0.0-1.0 per sample)
- Average total reward (sum of format + accuracy)
- Improvements (trained model vs base model)

Or use the notebook [compare_models.ipynb](notebooks/compare_models.ipynb) for interactive exploration.

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
