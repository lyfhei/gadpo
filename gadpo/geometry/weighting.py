from __future__ import annotations

import torch
import torch.nn.functional as F

from gadpo.utils.config import GeometryConfig
from gadpo.geometry.features import GeometryFeatures


def _normalize_to_01(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    if (mx - mn).abs() < 1e-8:
        return torch.ones_like(x) * 0.5
    return (x - mn) / (mx - mn)


class GeometryWeighter:
    """
    Computes per-sample DPO loss weights from geometric features.

    Formula:
        raw_i = alpha * cos_penalty_i + beta * dist_norm_i + margin_norm_i + (1 - entropy_i)
        w_i   = softmax(raw / temperature) * B   →  mean(w) = 1
        w_i   = clamp(w_i, clip_min, clip_max)

    All outputs are detached from the computation graph.
    """

    def __init__(self, cfg: GeometryConfig):
        self.alpha = cfg.cos_sim_scale
        self.beta = cfg.distance_scale
        self.temperature = cfg.weight_temperature
        self.clip_min = cfg.clip_min
        self.clip_max = cfg.clip_max

    def __call__(self, features: GeometryFeatures, current_quantile: float = 1.0) -> torch.Tensor:
        # 1. cos_penalty: high cos_sim → ambiguous → penalize
        cos_penalty = (1.0 - features.cos_sim) / 2.0        # [0, 1]

        # 2. distance bonus: large L2 → clear separation → reward
        dist_norm = _normalize_to_01(features.l2_distance)  # [0, 1]

        # 3. margin bonus from reward scores
        margin_norm = _normalize_to_01(features.margin)     # [0, 1]

        # 4. entropy penalty: high entropy → uncertain preference → penalize
        entropy_pen = 1.0 - features.entropy                # [0, 1]

        raw = (self.alpha * cos_penalty
               + self.beta * dist_norm
               + margin_norm
               + entropy_pen)

        B = raw.size(0)
        if B > 1:
            # Normalize so mean(w) = 1 while sharpening the distribution
            w = F.softmax(raw / self.temperature, dim=0) * B
        else:
            w = torch.ones(1, dtype=raw.dtype)

        w = w.clamp(self.clip_min, self.clip_max)
        w = torch.nan_to_num(w, nan=1.0)

        # Curriculum: zero-out pairs below the margin quantile threshold
        if current_quantile < 1.0 and B > 1:
            threshold = torch.quantile(features.margin, 1.0 - current_quantile)
            mask = (features.margin >= threshold).float()
            w = w * mask

        return w.detach()
