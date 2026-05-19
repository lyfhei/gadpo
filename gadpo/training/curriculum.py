from __future__ import annotations

import math

from gadpo.utils.config import CurriculumConfig


class CurriculumScheduler:
    """
    Controls what fraction of preference pairs are active at each training step,
    sorted by reward margin (easiest = largest margin first).

    At step 0:  only top `start_quantile` fraction (easiest pairs).
    At final step: all pairs (`end_quantile` = 1.0).
    """

    def __init__(self, cfg: CurriculumConfig, total_steps: int):
        self.start_q = cfg.start_quantile
        self.end_q = cfg.end_quantile
        self.strategy = cfg.strategy
        self.warmup_steps = int(cfg.warmup_ratio * total_steps)
        self.active_steps = max(total_steps - self.warmup_steps, 1)

    def get_current_quantile(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start_q
        progress = min((step - self.warmup_steps) / self.active_steps, 1.0)
        if self.strategy == "linear":
            q = self.start_q + progress * (self.end_q - self.start_q)
        elif self.strategy == "cosine":
            q = self.start_q + (self.end_q - self.start_q) * (1 - math.cos(math.pi * progress)) / 2
        elif self.strategy == "step":
            phase = min(int(progress * 3), 2)
            q = self.start_q + phase * (self.end_q - self.start_q) / 3
        else:
            q = self.end_q
        return float(q)
