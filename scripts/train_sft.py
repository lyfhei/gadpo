"""
SFT training script.

Usage:
    python scripts/train_sft.py --config configs/sft_qwen_1_5b.yaml
    python scripts/train_sft.py --config configs/sft_qwen_1_5b.yaml training.num_train_epochs=1
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gadpo.utils.config import parse_args_and_config
from gadpo.utils.reproducibility import set_seed
from gadpo.utils.logging_utils import setup_logging
from gadpo.data.loader import load_ultrafeedback, build_train_val_split
from gadpo.data.sft_dataset import build_sft_dataset
from gadpo.models.model_factory import load_base_model_and_tokenizer, apply_lora
from gadpo.training.sft_trainer import build_sft_trainer


def main():
    setup_logging()
    cfg = parse_args_and_config()
    set_seed(cfg.training.seed)

    # Data
    raw = load_ultrafeedback(
        dataset_name=cfg.data.dataset_name,
        max_samples=cfg.data.max_samples,
        min_margin=cfg.data.min_margin,
        seed=cfg.data.seed,
    )
    train_raw, val_raw = build_train_val_split(raw, val_ratio=cfg.data.val_ratio, seed=cfg.data.seed)

    # Model
    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Dataset
    train_ds = build_sft_dataset(train_raw, tokenizer, max_length=cfg.data.max_length)
    val_ds = build_sft_dataset(val_raw, tokenizer, max_length=cfg.data.max_length)

    # Train
    trainer = build_sft_trainer(model, tokenizer, train_ds, val_ds, cfg)
    trainer.train()
    trainer.save_model(os.path.join(cfg.training.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(cfg.training.output_dir, "final"))
    print(f"SFT checkpoint saved to {cfg.training.output_dir}/final")


if __name__ == "__main__":
    main()
