"""
Vanilla DPO baseline training script.

Usage:
    python scripts/train_dpo_baseline.py --config configs/dpo_baseline_qwen_1_5b.yaml \\
        --sft_checkpoint outputs/sft_qwen_1_5b/final
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gadpo.utils.config import load_config
from gadpo.utils.reproducibility import set_seed
from gadpo.utils.logging_utils import setup_logging
from gadpo.data.loader import load_ultrafeedback, build_train_val_split
from gadpo.data.dpo_dataset import build_dpo_dataset
from gadpo.models.model_factory import load_sft_checkpoint, load_base_model_and_tokenizer, apply_lora
from gadpo.training.dpo_baseline_trainer import build_baseline_dpo_trainer


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_checkpoint", default=None,
                        help="Path to SFT adapter checkpoint. If omitted, loads fresh base model.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.training.seed)

    # Data
    raw = load_ultrafeedback(
        dataset_name=cfg.data.dataset_name,
        max_samples=cfg.data.max_samples,
        min_margin=cfg.data.min_margin,
        seed=cfg.data.seed,
    )
    train_raw, val_raw = build_train_val_split(raw, val_ratio=cfg.data.val_ratio, seed=cfg.data.seed)
    train_ds = build_dpo_dataset(train_raw)
    val_ds = build_dpo_dataset(val_raw)

    # Model
    if args.sft_checkpoint:
        model, tokenizer = load_sft_checkpoint(args.sft_checkpoint, cfg)
    else:
        model, tokenizer = load_base_model_and_tokenizer(cfg)
        model = apply_lora(model, cfg)

    # Train
    trainer = build_baseline_dpo_trainer(model, tokenizer, train_ds, val_ds, cfg)
    trainer.train()
    trainer.save_model(os.path.join(cfg.training.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(cfg.training.output_dir, "final"))
    print(f"Baseline DPO checkpoint saved to {cfg.training.output_dir}/final")


if __name__ == "__main__":
    main()
