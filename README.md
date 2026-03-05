# NPR-RL

Reinforcement learning fine-tuning (RLVR) for **puzzle–answer** tasks using **GRPO (Group Relative Policy Optimization)**. Built on Qwen2.5 with LoRA and optional difficulty-based sampling.

## Features

- **GRPO training**: within-group relative advantage, KL penalty, reference model
- **LoRA + 4-bit**: memory-efficient, suitable for 1.5B-scale models
- **Reward**: extract answer from model output, normalize and compare to ground truth for 0/1 reward
- **Sampling modes**: `uniform` (baseline) or `difficulty` (p*(1-p) weighting by per-example pass-rate EMA)

## Requirements

- Python 3.8+
- CUDA (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Data format

`data.json` should be a JSON array; each item must have:

- `challenge`: puzzle/question text
- `answer`: ground-truth answer (used for 0/1 reward)

Example:

```json
[
  {"challenge": "Start with a six-letter geographic term for an ancient refuse heap often studied by archaeologists.\n\nChange only the first letter to form a new six-letter word meaning “to enrage” or “to drive crazy.”\n\nWhat word do you get?", "answer": "madden"},
  {"challenge": "Here’s your Sunday Puzzle challenge:\n\nTake the two-word phrase meaning “one final.” Rearrange its five letters to name a common geography book found in classrooms — a collection of maps.\n\nWhat is it?","answer": "ATLAS"}
]
```

## Usage

```bash
python train_grpo.py --data_path data.json --output_dir ./grpo_model
```

### Main arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | Qwen/Qwen2.5-1.5B-Instruct | Base model |
| `--data_path` | data.json | Path to training data |
| `--output_dir` | ./grpo_model | Save directory |
| `--group_size` | 8 | GRPO group size (samples per step) |
| `--lr` | 1e-6 | Learning rate |
| `--kl_beta` | 0.02 | KL penalty coefficient |
| `--max_new_tokens` | 128 | Max generated tokens per step |
| `--max_steps` | 1000 | Training steps |
| `--sampling_mode` | uniform | `uniform` or `difficulty` |
| `--difficulty_alpha` | 1.0 | Exponent for difficulty weighting |
| `--ema_momentum` | 0.1 | EMA momentum for per-example pass rate |

### Difficulty-based sampling

With `--sampling_mode difficulty`, the script maintains a per-example pass-rate EMA and samples proportionally to `p*(1-p)`, favoring medium-difficulty items.

## Output

- Trained LoRA weights are saved under `--output_dir`.
- Every `--log_every` steps, the console prints loss, reward, effective_ratio, avg_pass_ema, etc.

## License

MIT
