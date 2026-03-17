"""
DAPO training for puzzle-answer RLVR (improved GRPO).

Key improvements over train_grpo.py:
  1. max_new_tokens default: 8192 (was 128 — QwQ needs long thinking chains)
  2. Prompt via apply_chat_template (activates QwQ thinking mode)
  3. DAPO token-level policy gradient loss (total-token normalisation)
  4. Clip-Higher: asymmetric eps (eps_low=0.2, eps_high=0.28)
  5. Dynamic Sampling: skip all-correct / all-wrong groups
  6. Expanded LoRA (r=64, all projection layers)
  7. Gradient clipping + cosine-decay LR scheduler with warmup
  8. --dr_grpo flag: mean-only advantage normalisation (no std division)

Reference: DAPO — arxiv 2503.14476 (Qwen team, 2025)
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from reward import compute_reward
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DAPO training for puzzle-answer RLVR")
    parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B")
    parser.add_argument("--data_path", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default="./dapo_model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Core RL settings
    parser.add_argument("--group_size", type=int, default=6,
                        help="Responses sampled per question (6 vs 8 saves memory with long sequences)")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.02,
                        help="KL penalty coefficient (set 0 to disable)")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="CRITICAL: was 128 in train_grpo.py; QwQ needs thousands of tokens")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save checkpoint every N steps (0 to disable)")

    # DAPO-specific
    parser.add_argument("--eps_low", type=float, default=0.2,
                        help="Clip lower bound (symmetric GRPO uses 0.2 for both)")
    parser.add_argument("--eps_high", type=float, default=0.28,
                        help="Clip upper bound — higher than eps_low prevents entropy collapse")
    parser.add_argument("--dr_grpo", action="store_true",
                        help="Dr. GRPO: normalise advantages by mean only, not std "
                             "(avoids gradient bias when std is near zero)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Linear LR warmup steps before cosine decay")

    # Difficulty-based sampling (same as train_grpo.py)
    parser.add_argument("--sampling_mode", type=str, default="uniform",
                        choices=["uniform", "difficulty"],
                        help="uniform: baseline; difficulty: p*(1-p) reweighting")
    parser.add_argument("--difficulty_alpha", type=float, default=1.0)
    parser.add_argument("--difficulty_eps", type=float, default=1e-3)
    parser.add_argument("--ema_momentum", type=float, default=0.1,
                        help="EMA momentum for per-example pass-rate tracking")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="npr-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(challenge: str, tokenizer) -> str:
    """Use chat template so QwQ-32B activates its thinking (<think>) mode."""
    messages = [{"role": "user", "content": challenge}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def selected_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token log-probabilities for the tokens that were sampled.

    logits  : [1, T, V]
    token_ids: [1, T]
    returns : [1, T-1]  — log p(token_t | context_{<t}) for t=1..T-1
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs.gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Avoid tp_plan='auto' when WORLD_SIZE is set by job scheduler but we run
    # single-process.  Unset distributed env vars so device_map="auto" uses
    # normal model parallelism (accelerate), not tensor parallelism.
    if os.environ.get("WORLD_SIZE") and "LOCAL_RANK" not in os.environ:
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)

    args = parse_args()
    set_seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        tp_plan=None,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        tp_plan=None,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Expanded LoRA: all projection layers, rank 64 (was r=16, q/v only)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_config)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    data = list(dataset)
    num_examples = len(data)
    if num_examples == 0:
        raise ValueError("Dataset is empty.")

    # Per-example difficulty tracking (EMA of pass rate)
    pass_rate_ema = np.full(num_examples, 0.5, dtype=np.float32)
    seen_count = np.zeros(num_examples, dtype=np.int32)

    effective_steps = 0
    running_reward_mean = 0.0
    running_reward_count = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for step in range(args.max_steps):

        # ---- Question sampling ----------------------------------------
        if args.sampling_mode == "difficulty":
            weights = pass_rate_ema * (1.0 - pass_rate_ema)
            weights = np.power(weights + args.difficulty_eps, args.difficulty_alpha)
            weights /= weights.sum()
            idx = int(np.random.choice(num_examples, p=weights))
        else:
            idx = np.random.randint(num_examples)

        example = data[idx]
        challenge = example["challenge"]
        answer = example["answer"]

        prompt = build_prompt(challenge, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        prompt_len = inputs["input_ids"].shape[1]

        # ---- Phase 1: Generate group_size responses (no grad) ----------
        all_outputs = []   # list of [1, T] token-ID tensors
        rewards = []

        for _ in range(args.group_size):
            with torch.no_grad():
                output = policy.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # <think> tags are regular text tokens in QwQ, not special tokens,
            # so skip_special_tokens=False is equivalent here — but it avoids
            # stripping any special control tokens that reward.py doesn't need.
            text = tokenizer.decode(output[0], skip_special_tokens=False)
            rewards.append(compute_reward(text, answer))
            all_outputs.append(output)

        rewards = np.array(rewards, dtype=np.float32)
        running_reward_mean += float(rewards.mean())
        running_reward_count += 1

        # Update per-example difficulty EMA
        obs_pass_rate = float(rewards.mean())
        pass_rate_ema[idx] = (
            (1.0 - args.ema_momentum) * pass_rate_ema[idx]
            + args.ema_momentum * obs_pass_rate
        )
        seen_count[idx] += 1

        # ---- DAPO Dynamic Sampling ------------------------------------
        # Skip groups where all answers are correct or all are wrong —
        # the advantage signal is zero and the update is pure noise.
        if rewards.std() == 0:
            continue

        effective_steps += 1

        # ---- Advantage computation ------------------------------------
        if args.dr_grpo:
            # Dr. GRPO: mean-only normalisation avoids bias when std ≈ 0
            advantages = rewards - rewards.mean()
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ---- Phase 2: Compute DAPO token-level loss -------------------
        # We need total_tokens for normalisation before the backward loop.
        gen_slice_start = max(prompt_len - 1, 0)
        total_tokens = sum(
            out.shape[1] - prompt_len for out in all_outputs
        )

        # Gradient accumulation: backward per sample to keep peak memory low.
        optimizer.zero_grad()

        loss_scalar = 0.0      # for logging only
        policy_loss_scalar = 0.0
        kl_scalar = 0.0
        inv_total = 1.0 / max(total_tokens, 1)

        for output, adv in zip(all_outputs, advantages):
            # Forward passes
            policy_logits = policy(output).logits           # grads flow here
            with torch.no_grad():
                ref_logits = ref_model(output).logits

            # Per-token log-probs for the full sequence
            policy_sel = selected_token_logprobs(policy_logits, output)
            ref_sel = selected_token_logprobs(ref_logits, output)

            # Slice to generated tokens only
            policy_gen = policy_sel[:, gen_slice_start:].squeeze(0)   # [T_gen]
            ref_gen = ref_sel[:, gen_slice_start:].squeeze(0)          # [T_gen]

            # DAPO token-level policy gradient with clip-higher
            adv_t = torch.tensor(float(adv), device=policy_gen.device)
            ratio = torch.exp(policy_gen - ref_gen.detach())
            clipped = torch.clamp(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
            token_losses = -adv_t * torch.min(ratio, clipped)  # [T_gen]

            # KL penalty (Monte Carlo estimate on sampled trajectory)
            kl = (policy_gen - ref_gen.detach()).mean()

            # Scale so that the sum across samples equals the DAPO objective:
            #   L = (1 / total_tokens) * sum_i sum_t loss_{i,t}
            #     + kl_beta * (1 / G) * sum_i KL_i
            sample_loss = (
                token_losses.sum() * inv_total
                + args.kl_beta * kl / args.group_size
            )
            sample_loss.backward()

            loss_scalar += sample_loss.item()
            policy_loss_scalar += (token_losses.sum() * inv_total).item()
            kl_scalar += kl.item() / args.group_size

            # Free activations before next forward pass
            del policy_logits, ref_logits, policy_sel, ref_sel
            del policy_gen, ref_gen, ratio, clipped, token_losses, kl
            torch.cuda.empty_cache()

        # Gradient clipping + optimiser step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # ---- Logging --------------------------------------------------
        if step % args.log_every == 0:
            effective_ratio = effective_steps / max(step + 1, 1)
            seen_mask = seen_count > 0
            avg_seen_ema = float(pass_rate_ema[seen_mask].mean()) if seen_mask.any() else 0.5
            avg_reward = running_reward_mean / max(running_reward_count, 1)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step={step:5d}  loss={loss_scalar:.4f}  "
                f"policy={policy_loss_scalar:.4f}  kl={kl_scalar:.4f}  "
                f"reward_mean={avg_reward:.4f}  batch={rewards.tolist()}  "
                f"eff_ratio={effective_ratio:.3f}  ema={avg_seen_ema:.3f}  "
                f"lr={current_lr:.2e}  tokens={total_tokens}"
            )
            if not args.no_wandb:
                wandb.log(
                    {
                        "train/loss": loss_scalar,
                        "train/policy_loss": policy_loss_scalar,
                        "train/kl_loss": kl_scalar,
                        "train/reward_mean": avg_reward,
                        "train/reward_std": float(rewards.std()),
                        "train/effective_ratio": effective_ratio,
                        "train/avg_pass_ema": avg_seen_ema,
                        "train/lr": current_lr,
                        "train/tokens_per_step": total_tokens,
                        "step": step,
                    }
                )

        if args.save_every and (step + 1) % args.save_every == 0 and step > 0:
            ckpt_dir = f"{args.output_dir}/checkpoint-{step + 1}"
            policy.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved → {ckpt_dir}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    policy.save_pretrained(args.output_dir)
    print(f"Model saved → {args.output_dir}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
