from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_ultrafeedback(
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    split: str = "train",
    max_samples: Optional[int] = None,
    min_margin: float = 0.0,
    seed: int = 42,
) -> Dataset:
    logger.info(f"Loading {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split)

    if min_margin > 0.0:
        before = len(dataset)
        dataset = dataset.filter(
            lambda x: (x["score_chosen"] - x["score_rejected"]) >= min_margin
        )
        logger.info(f"Margin filter ({min_margin}): {before} → {len(dataset)} samples")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Subsampled to {max_samples} samples")

    return dataset


def build_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    logger.info(f"Train: {len(split['train'])}  Val: {len(split['test'])}")
    return split["train"], split["test"]
