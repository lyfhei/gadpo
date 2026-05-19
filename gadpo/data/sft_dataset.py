from __future__ import annotations

import logging
from typing import Dict

from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _to_list(ids) -> list:
    """Ensure apply_chat_template output is a plain Python list of native ints."""
    # BatchEncoding (Transformers 5.x) is a Mapping but may not subclass dict
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return [int(x) for x in ids]


def _extract_messages(example: dict) -> list:
    chosen = example["chosen"]
    if isinstance(chosen, list):
        return chosen
    return [{"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": str(chosen)}]


def format_sft_example(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    messages = _extract_messages(example)

    full_ids = _to_list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        max_length=max_length,
        truncation=True,
    ))

    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_ids = _to_list(tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        max_length=max_length,
        truncation=True,
    ))

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[:max_length]
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_sft_dataset(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    num_proc: int = 1,
) -> Dataset:
    logger.info("Tokenizing SFT dataset...")
    dataset = raw_dataset.map(
        lambda ex: format_sft_example(ex, tokenizer, max_length),
        batched=False,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing SFT",
    )
    # Remove samples where the response was fully truncated (all labels = -100)
    before = len(dataset)
    dataset = dataset.filter(lambda x: any(l != IGNORE_INDEX for l in x["labels"]))
    removed = before - len(dataset)
    if removed:
        logger.warning(f"Removed {removed} samples with fully-masked labels (prompt too long)")
    return dataset
