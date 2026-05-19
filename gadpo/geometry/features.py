from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GeometryFeatures:
    cos_sim: torch.Tensor        # (B,) cosine similarity between h_w and h_l
    l2_distance: torch.Tensor    # (B,) Euclidean distance
    margin: torch.Tensor         # (B,) score_chosen - score_rejected
    entropy: torch.Tensor        # (B,) entropy of softmax([score_w, score_l])
    local_density: torch.Tensor  # (B,) approx. density of chosen states in batch


def compute_cosine_similarity(h_w: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
    h_w_n = F.normalize(h_w, dim=-1)
    h_l_n = F.normalize(h_l, dim=-1)
    return (h_w_n * h_l_n).sum(dim=-1)   # (B,)


def compute_l2_distance(h_w: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
    return torch.norm(h_w - h_l, dim=-1)  # (B,)


def compute_local_density(h: torch.Tensor, radius: float = 0.5) -> torch.Tensor:
    """
    Fraction of batch neighbors within cosine-distance `radius`.
    Returns (B,) in [0, 1]. Uses pairwise cosine similarity — O(B^2 * D).
    Efficient for small batch sizes (B=1..8).
    """
    B = h.size(0)
    if B < 2:
        return torch.zeros(B, dtype=h.dtype, device=h.device)
    h_n = F.normalize(h, dim=-1)
    sim = h_n @ h_n.T                        # (B, B)
    sim.fill_diagonal_(-2.0)                 # exclude self
    within = (sim > (1.0 - radius)).float()  # cosine dist = 1 - cos_sim
    return within.sum(dim=1) / (B - 1)       # (B,)


def compute_reward_entropy(score_w: torch.Tensor, score_l: torch.Tensor) -> torch.Tensor:
    scores = torch.stack([score_w, score_l], dim=-1)   # (B, 2)
    probs = F.softmax(scores, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
    return entropy / math.log(2)   # normalize to [0, 1]


def compute_all_features(
    h_w: torch.Tensor,
    h_l: torch.Tensor,
    score_w: torch.Tensor,
    score_l: torch.Tensor,
    density_radius: float = 0.5,
) -> GeometryFeatures:
    return GeometryFeatures(
        cos_sim=compute_cosine_similarity(h_w, h_l),
        l2_distance=compute_l2_distance(h_w, h_l),
        margin=(score_w - score_l).clamp(min=0.0),
        entropy=compute_reward_entropy(score_w, score_l),
        local_density=compute_local_density(h_w, radius=density_radius),
    )
