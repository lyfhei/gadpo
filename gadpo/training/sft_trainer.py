from __future__ import annotations

from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
from peft import PeftModel

from gadpo.utils.config import Config


class SFTCollator:
    """Pads pre-tokenized (input_ids, attention_mask, labels) to the same length."""

    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)

        def pad(seqs, pad_val):
            out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(0)] = s
            return out

        return {
            "input_ids": pad(input_ids, self.pad_token_id),
            "attention_mask": pad(attention_mask, 0),
            "labels": pad(labels, self.label_pad_id),
        }


def build_sft_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: Config,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        fp16=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        seed=cfg.training.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = SFTCollator(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_id=-100,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=collator,
    )
    return trainer
