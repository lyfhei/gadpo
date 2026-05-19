from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the last non-padding token's hidden state per sequence.

    hidden: (B, T, D)
    attention_mask: (B, T)  — 1 for real tokens, 0 for padding
    returns: (B, D)
    """
    last_idx = attention_mask.sum(dim=1).long() - 1          # (B,)
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]                        # (B, D)


class HiddenStateExtractor:
    """
    Registers a forward hook on the last transformer decoder layer to capture
    hidden states during DPO's concatenated forward pass.

    TRL's DPOTrainer stacks [chosen_1..chosen_B, rejected_1..rejected_B] into
    one batch and calls model() once. We capture the resulting (2B, T, D) tensor
    in that single call — no extra overhead vs standard DPO.

    Usage:
        extractor = HiddenStateExtractor(model)
        extractor.register()
        # ... forward pass happens ...
        hidden = extractor.get_and_clear()   # (2B, T, D)
        extractor.remove()
    """

    def __init__(self, model: PreTrainedModel, layer_index: int = -1):
        self.model = model
        self.layer_index = layer_index
        self._captured: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _get_last_layer(self) -> nn.Module:
        # Navigate through PEFT wrapping: PeftModel → base_model → Qwen2Model → layers
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        # Qwen2.5 architecture: model.model.layers
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers[self.layer_index]
        # Fallback for other architectures
        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            return base.transformer.h[self.layer_index]
        raise RuntimeError(
            f"Cannot locate decoder layers in {type(base).__name__}. "
            "Override HiddenStateExtractor._get_last_layer() for your architecture."
        )

    def _hook_fn(self, module, input, output):
        # output is a tuple; output[0] is the hidden state tensor (B, T, D)
        if isinstance(output, tuple):
            self._captured = output[0].detach()
        else:
            self._captured = output.detach()

    def register(self):
        layer = self._get_last_layer()
        self._hook_handle = layer.register_forward_hook(self._hook_fn)
        logger.debug(f"Registered hidden state hook on {type(layer).__name__}")

    def get_and_clear(self) -> Optional[torch.Tensor]:
        states = self._captured
        self._captured = None
        return states

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
