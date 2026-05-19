"""
Compare baseline DPO vs GA-DPO checkpoints on reward metrics and stability.

Usage:
    python scripts/evaluate.py --config configs/gadpo_qwen_1_5b.yaml \\
        --baseline outputs/dpo_baseline_qwen_1_5b/final \\
        --gadpo outputs/gadpo_qwen_1_5b/final \\
        --output outputs/eval_results.json
"""
import os
import sys
import argparse
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gadpo.utils.config import load_config
from gadpo.utils.reproducibility import set_seed
from gadpo.utils.logging_utils import setup_logging
from gadpo.data.loader import load_ultrafeedback, build_train_val_split
from gadpo.data.dpo_dataset import build_dpo_dataset
from gadpo.models.model_factory import load_sft_checkpoint
from gadpo.evaluation.reward_metrics import compute_reward_metrics
from gadpo.evaluation.stability import compute_preference_consistency


def eval_checkpoint(name, checkpoint_dir, tokenizer, eval_ds, cfg, device):
    print(f"\n--- Evaluating: {name} ---")
    model, _ = load_sft_checkpoint(checkpoint_dir, cfg)
    model.eval()

    metrics = compute_reward_metrics(
        model, tokenizer, eval_ds,
        max_length=cfg.data.max_length,
        device=device,
    )
    stability = compute_preference_consistency(
        model, tokenizer, eval_ds,
        max_length=cfg.data.max_length,
        device=device,
        n_samples=200,
    )
    metrics.update(stability)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gadpo", required=True)
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shared eval data
    raw = load_ultrafeedback(
        dataset_name=cfg.data.dataset_name,
        max_samples=500,
        seed=cfg.data.seed,
    )
    _, val_raw = build_train_val_split(raw, val_ratio=0.5, seed=cfg.data.seed)
    eval_ds = build_dpo_dataset(val_raw)

    # Load tokenizer from baseline checkpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    results["baseline"] = eval_checkpoint("Baseline DPO", args.baseline, tokenizer, eval_ds, cfg, device)
    results["gadpo"] = eval_checkpoint("GA-DPO", args.gadpo, tokenizer, eval_ds, cfg, device)

    # Print comparison
    print("\n=== Comparison ===")
    for k in results["baseline"]:
        b = results["baseline"][k]
        g = results["gadpo"][k]
        delta = g - b
        print(f"  {k:35s}  baseline={b:.4f}  gadpo={g:.4f}  Δ={delta:+.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
