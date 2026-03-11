import argparse
import random

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from reward import compute_reward
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for puzzle-answer RLVR")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default="./grpo_model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Core GRPO settings
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N steps (0 to disable)")

    # Sampling settings for experiment ablations
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "difficulty"],
        help="uniform: baseline, difficulty: p*(1-p) reweighting",
    )
    parser.add_argument("--difficulty_alpha", type=float, default=1.0)
    parser.add_argument("--difficulty_eps", type=float, default=1e-3)
    parser.add_argument(
        "--ema_momentum",
        type=float,
        default=0.1,
        help="EMA momentum for per-example pass-rate tracking",
    )
    # W&B
    parser.add_argument("--wandb_project", type=str, default="npr-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_token_logprobs(logits, token_ids):
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    selected = log_probs.gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return selected


def compute_sampled_kl(policy_selected_log_probs, ref_selected_log_probs):
    # Monte Carlo estimate on sampled trajectory: E_pi[log pi - log pi_ref]
    return (policy_selected_log_probs - ref_selected_log_probs).mean()


def build_prompt(challenge: str) -> str:
    return f"Solve step by step:\n{challenge}\nAnswer:"


def main():
    args = parse_args()
    set_seed(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    policy = get_peft_model(policy, lora_config)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    data = list(dataset)
    num_examples = len(data)
    if num_examples == 0:
        raise ValueError("Dataset is empty.")

    # Per-example difficulty tracking
    pass_rate_ema = np.full(num_examples, 0.5, dtype=np.float32)
    seen_count = np.zeros(num_examples, dtype=np.int32)

    effective_steps = 0
    running_reward_mean = 0.0
    running_reward_count = 0

    for step in range(args.max_steps):
        if args.sampling_mode == "difficulty":
            weights = pass_rate_ema * (1.0 - pass_rate_ema)
            weights = np.power(weights + args.difficulty_eps, args.difficulty_alpha)
            weights = weights / weights.sum()
            idx = np.random.choice(num_examples, p=weights)
        else:
            idx = np.random.randint(num_examples)

        example = data[idx]
        challenge = example["challenge"]
        answer = example["answer"]

        prompt = build_prompt(challenge)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        prompt_len = inputs["input_ids"].shape[1]

        sample_logprobs = []
        sample_kls = []
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

            policy_logits = policy(output).logits
            with torch.no_grad():
                ref_logits = ref_model(output).logits

            policy_selected = selected_token_logprobs(policy_logits, output)
            ref_selected = selected_token_logprobs(ref_logits, output)

            # keep only generated segment
            gen_slice_start = max(prompt_len - 1, 0)
            policy_gen_selected = policy_selected[:, gen_slice_start:]
            ref_gen_selected = ref_selected[:, gen_slice_start:]

            logprob = policy_gen_selected.mean()
            kl = compute_sampled_kl(policy_gen_selected, ref_gen_selected)
            sample_logprobs.append(logprob)
            sample_kls.append(kl)

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            reward = compute_reward(text, answer)
            rewards.append(reward)

        rewards = np.array(rewards, dtype=np.float32)
        running_reward_mean += rewards.mean()
        running_reward_count += 1

        # update per-example pass-rate EMA
        obs_pass_rate = float(rewards.mean())
        pass_rate_ema[idx] = (1.0 - args.ema_momentum) * pass_rate_ema[idx] + args.ema_momentum * obs_pass_rate
        seen_count[idx] += 1

        if rewards.std() == 0:
            continue

        effective_steps += 1
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        policy_loss = 0.0
        kl_loss = 0.0
        for logprob, kl, adv in zip(sample_logprobs, sample_kls, normalized_rewards):
            policy_loss = policy_loss + (-adv * logprob)
            kl_loss = kl_loss + kl

        policy_loss = policy_loss / args.group_size
        kl_loss = kl_loss / args.group_size
        loss = policy_loss + args.kl_beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            effective_ratio = effective_steps / max(step + 1, 1)
            seen_mask = seen_count > 0
            avg_seen_ema = pass_rate_ema[seen_mask].mean() if seen_mask.any() else 0.5
            avg_reward = running_reward_mean / max(running_reward_count, 1)
            print(
                f"step={step} loss={loss.item():.4f} "
                f"policy={policy_loss.item():.4f} kl={kl_loss.item():.4f} "
                f"reward_mean={avg_reward:.4f} batch_rewards={rewards.tolist()} "
                f"effective_ratio={effective_ratio:.3f} avg_pass_ema={avg_seen_ema:.3f}"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/kl_loss": kl_loss.item(),
                    "train/reward_mean": avg_reward,
                    "train/reward_std": float(rewards.std()),
                    "train/effective_ratio": effective_ratio,
                    "train/avg_pass_ema": avg_seen_ema,
                    "step": step,
                }
            )
        if args.save_every and (step + 1) % args.save_every == 0 and step > 0:
            ckpt_dir = f"{args.output_dir}/checkpoint-{step + 1}"
            policy.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

    policy.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()