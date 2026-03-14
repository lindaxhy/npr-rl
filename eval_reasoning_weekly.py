#!/usr/bin/env python3
"""
Evaluate a model on nuprl/reasoning-weekly. Uses same prompt as train_grpo.
Correctness uses phrase-based ground-truth matching from reference_code (alternatives
and phrase sets), not exact string match.
Usage:
  # Base Qwen3-0.6B
  python eval_reasoning_weekly.py --model_path Qwen/Qwen3-0.6B --output results_base.json

  # GRPO fine-tuned (PEFT adapter)
  python eval_reasoning_weekly.py --model_path ./grpo_qwen3_0.6b --output results_grpo.json
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from reward import compute_reward
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import extract_answer, normalize


def parse_args():
    p = argparse.ArgumentParser(description="Eval on nuprl/reasoning-weekly")
    p.add_argument("--model_path", type=str, required=True, help="Base model name or path; if dir with adapter_config.json, loads PEFT")
    p.add_argument("--output", type=str, default=None, help="Optional JSON path to save per-example results")
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (default: all)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def build_prompt(challenge: str) -> str:
    return f"Solve step by step:\n{challenge}\nAnswer:"


def main():
    import os
    # Avoid tp_plan='auto' when WORLD_SIZE is set by job scheduler but we run single-process.
    # In that case LOCAL_RANK is unset, causing KeyError. Unset these so device_map="auto"
    # uses normal model parallelism (accelerate), not tensor parallelism.
    if os.environ.get("WORLD_SIZE") and "LOCAL_RANK" not in os.environ:
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)
    args = parse_args()
    model_path = args.model_path.rstrip("/")

    # Detect PEFT: path is a directory containing adapter_config.json (no tokenizer there)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-0.6B")
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        print(f"Loading base model {base_name} then PEFT from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loading base model {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()

    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("nuprl/reasoning-weekly", split="test")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    correct = 0
    total = len(ds)
    results = []
    print(f"Evaluating on {total} samples...")
    last_pct = -1
    log_every = max(1, total // 20)  # 约每 5% 打一次

    for i in range(0, total, args.batch_size):
        batch = ds.select(range(i, min(i + args.batch_size, total)))
        for j, ex in enumerate(batch):
            challenge = ex["challenge"]
            answer = ex["answer"]
            prompt = build_prompt(challenge)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            text = tokenizer.decode(out[0], skip_special_tokens=True)
            reward = compute_reward(text, answer)
            correct += reward
            pred_norm = normalize(extract_answer(text))
            gt_norm = normalize(answer)
            results.append({
                "challenge_id": ex.get("ID", i + j),
                "correct": bool(reward),
                "pred": pred_norm,
                "gold": gt_norm,
                "raw_output": text,
            })
        processed = min(i + args.batch_size, total)
        pct = int(100 * processed / total)
        if processed % log_every == 0 or processed == total or pct > last_pct:
            last_pct = pct
            acc_so_far = correct / processed if processed else 0.0
            print(f"  [{processed}/{total}] {pct}% | 当前正确: {correct}/{processed} ({100*acc_so_far:.1f}%)")

    accuracy = correct / total if total else 0.0
    print(f"\nModel: {args.model_path}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f} ({100*accuracy:.2f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": accuracy, "correct": correct, "total": total, "results": results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
