"""
Geometry-Aware DPO training script.

Usage:
    python scripts/train_gadpo.py --config configs/gadpo_qwen_1_5b.yaml \\
        --sft_checkpoint outputs/sft_qwen_1_5b/final
"""
import os
import sys
import argparse
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gadpo.utils.config import load_config
from gadpo.utils.reproducibility import set_seed
from gadpo.utils.logging_utils import setup_logging
from gadpo.data.loader import load_ultrafeedback, build_train_val_split
from gadpo.data.dpo_dataset import build_dpo_dataset
from gadpo.models.model_factory import load_sft_checkpoint, load_base_model_and_tokenizer, apply_lora
from gadpo.training.gadpo_trainer import build_gadpo_trainer


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_checkpoint", default=None)
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

    # Estimate total training steps for curriculum scheduler
    steps_per_epoch = math.ceil(
        len(train_ds)
        / (cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * cfg.training.num_train_epochs

    # Train
    trainer = build_gadpo_trainer(model, tokenizer, train_ds, val_ds, cfg, total_steps)
    trainer.train()
    trainer.save_model(os.path.join(cfg.training.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(cfg.training.output_dir, "final"))
    print(f"GA-DPO checkpoint saved to {cfg.training.output_dir}/final")


if __name__ == "__main__":
    main()
