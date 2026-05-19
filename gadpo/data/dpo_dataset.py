from __future__ import annotations

import logging
from typing import Dict

from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _get_text(messages) -> str:
    if isinstance(messages, list):
        return messages[-1]["content"] if messages else ""
    return str(messages)


def format_dpo_example(example: dict) -> Dict:
    """
    Returns flat fields expected by TRL's DPOTrainer plus score columns
    needed by GeometryAwareDPOTrainer.
    """
    prompt = example.get("prompt", "")
    if not prompt and isinstance(example.get("chosen"), list):
        # Extract prompt from message list (user turn)
        for msg in example["chosen"]:
            if msg["role"] == "user":
                prompt = msg["content"]
                break

    chosen_text = _get_text(example["chosen"])
    rejected_text = _get_text(example["rejected"])

    score_chosen = float(example.get("score_chosen", 1.0))
    score_rejected = float(example.get("score_rejected", 0.0))

    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
        "margin": score_chosen - score_rejected,
    }


def build_dpo_dataset(
    raw_dataset: Dataset,
    num_proc: int = 1,
) -> Dataset:
    logger.info("Formatting DPO dataset...")
    keep_cols = {"prompt", "chosen", "rejected", "score_chosen", "score_rejected", "margin"}
    dataset = raw_dataset.map(
        format_dpo_example,
        batched=False,
        num_proc=num_proc,
        remove_columns=[c for c in raw_dataset.column_names if c not in keep_cols],
        desc="Formatting DPO pairs",
    )
    return dataset
