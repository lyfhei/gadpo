from __future__ import annotations

from datasets import Dataset
from transformers import PreTrainedTokenizer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

from gadpo.utils.config import Config


def build_baseline_dpo_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: Config,
) -> DPOTrainer:
    dpo_cfg = DPOConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        beta=cfg.training.beta,
        bf16=cfg.training.bf16,
        fp16=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        max_grad_norm=cfg.training.max_grad_norm,
        max_length=cfg.data.max_length,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=dpo_cfg,
    )
    return trainer
