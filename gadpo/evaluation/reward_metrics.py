from __future__ import annotations

import logging
from typing import Dict

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_sequence_logp(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Returns mean per-token log-probability for each sequence in the batch."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]          # (B, T-1, V)
    shift_labels = labels[:, 1:]             # (B, T-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    valid = (shift_labels != -100).float()
    return (token_logps * valid).sum(-1) / valid.sum(-1).clamp(min=1)


@torch.no_grad()
def compute_reward_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    max_length: int = 512,
    batch_size: int = 2,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    For each eval preference pair, compute:
    - chosen_logp and rejected_logp under the current model
    - reward margin = chosen_logp - rejected_logp
    - win_rate = fraction of pairs where chosen_logp > rejected_logp
    """
    model.eval()
    chosen_logps, rejected_logps = [], []

    def tokenize(texts):
        return tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )

    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i: i + batch_size]
        chosen_inputs = tokenize(batch["chosen"])
        rejected_inputs = tokenize(batch["rejected"])

        c_ids = chosen_inputs["input_ids"].to(device)
        c_mask = chosen_inputs["attention_mask"].to(device)
        r_ids = rejected_inputs["input_ids"].to(device)
        r_mask = rejected_inputs["attention_mask"].to(device)

        c_lp = compute_sequence_logp(model, c_ids, c_mask, c_ids.clone())
        r_lp = compute_sequence_logp(model, r_ids, r_mask, r_ids.clone())

        chosen_logps.append(c_lp.cpu())
        rejected_logps.append(r_lp.cpu())

    chosen_logps = torch.cat(chosen_logps)
    rejected_logps = torch.cat(rejected_logps)
    margins = chosen_logps - rejected_logps

    return {
        "reward_margin_mean": margins.mean().item(),
        "reward_margin_std": margins.std().item(),
        "reward_margin_median": margins.median().item(),
        "win_rate": (margins > 0).float().mean().item(),
        "chosen_logp_mean": chosen_logps.mean().item(),
        "rejected_logp_mean": rejected_logps.mean().item(),
    }
