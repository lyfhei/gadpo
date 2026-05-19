from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedTokenizer
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import selective_log_softmax
from peft import PeftModel

from gadpo.utils.config import Config
from gadpo.geometry.extractor import HiddenStateExtractor, pool_last_token
from gadpo.geometry.features import compute_all_features
from gadpo.geometry.weighting import GeometryWeighter
from gadpo.training.curriculum import CurriculumScheduler

logger = logging.getLogger(__name__)


class GeometryAwareDPOTrainer(DPOTrainer):
    """
    Extends TRL's DPOTrainer with per-sample geometry-aware loss weighting.

    Each DPO loss L_i is multiplied by a weight w_i derived from the cosine
    similarity, L2 distance, reward margin, and entropy of the chosen/rejected
    hidden representations. Pairs that are ambiguous (high cos_sim) or
    uncertain (high entropy) receive lower weights; clear preferences receive
    higher weights.

    The geometry weights are computed on CPU in fp32 and detached from the
    computation graph — they do not create gradient paths back into the model.
    """

    def __init__(
        self,
        *args,
        geometry_cfg,
        curriculum_cfg=None,
        total_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.extractor = HiddenStateExtractor(self.model)
        self.extractor.register()
        self.weighter = GeometryWeighter(geometry_cfg)
        self.curriculum: Optional[CurriculumScheduler] = None
        if curriculum_cfg is not None and curriculum_cfg.enabled:
            self.curriculum = CurriculumScheduler(curriculum_cfg, total_steps)
        self._geo_metrics: dict = defaultdict(list)
        self._last_print_step: int = -1

    # ------------------------------------------------------------------
    # Core override: replaces TRL 1.4.x _compute_loss with geometry-aware
    # weighted sigmoid DPO loss.
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if model.training else "eval"

        # --- Policy forward pass (hook fires here) ---
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        all_hidden = self.extractor.get_and_clear()  # (2B, T, D) or None

        # --- Per-token log probs ---
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        logps = per_token_logps.sum(dim=1)

        B = logps.size(0) // 2
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # --- Reference log probs (adapter-disabling trick, no extra model) ---
        with torch.no_grad():
            unwrapped = self.accelerator.unwrap_model(self.model)
            with unwrapped.disable_adapter():
                ref_outputs = self.model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # --- Sigmoid DPO per-sequence losses ---
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = chosen_logratios - rejected_logratios
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

        # --- Geometry weights (train only; eval uses uniform for fair comparison) ---
        if mode == "train":
            score_w_proxy = chosen_logratios.detach().float().cpu()
            score_l_proxy = rejected_logratios.detach().float().cpu()
            weights = self._compute_geometry_weights(
                all_hidden, inputs, B,
                device=logps.device,
                score_w=score_w_proxy,
                score_l=score_l_proxy,
            )
        else:
            weights = torch.ones(B, device=logps.device)

        loss = (weights * per_sequence_loss).mean()

        # --- Log key metrics into TRL's _metrics dict ---
        try:
            chosen_rewards = self.beta * chosen_logratios.detach()
            rejected_rewards = self.beta * rejected_logratios.detach()
            m = self._metrics[mode]
            m["rewards/chosen"].append(chosen_rewards.mean().item())
            m["rewards/rejected"].append(rejected_rewards.mean().item())
            m["rewards/margins"].append((chosen_rewards - rejected_rewards).mean().item())
            m["rewards/accuracies"].append(
                (chosen_rewards > rejected_rewards).float().mean().item()
            )
            m["logps/chosen"].append(chosen_logps.detach().mean().item())
            m["logps/rejected"].append(rejected_logps.detach().mean().item())
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _compute_geometry_weights(
        self, all_hidden, inputs, B: int, device,
        score_w: torch.Tensor = None,
        score_l: torch.Tensor = None,
    ) -> torch.Tensor:
        weights = torch.ones(B, device=device)
        if all_hidden is None or all_hidden.size(0) != 2 * B:
            if all_hidden is None:
                logger.warning("Hidden states not captured; using uniform weights")
            return weights
        try:
            h_w_raw = all_hidden[:B]   # (B, T, D)
            h_l_raw = all_hidden[B:]   # (B, T, D)

            # Use attention_mask for last-token pooling
            attn_mask = inputs.get("attention_mask")
            if attn_mask is not None:
                chosen_mask = attn_mask[:B]
                rejected_mask = attn_mask[B:]
            else:
                chosen_mask = torch.ones(B, h_w_raw.size(1), dtype=torch.long)
                rejected_mask = chosen_mask

            h_w = pool_last_token(h_w_raw, chosen_mask).float().cpu()
            h_l = pool_last_token(h_l_raw, rejected_mask).float().cpu()

            # Use passed-in proxies, or fall back to dataset score columns
            if score_w is None:
                score_w = inputs.get("score_chosen", torch.ones(B)).float().cpu()
            if score_l is None:
                score_l = inputs.get("score_rejected", torch.zeros(B)).float().cpu()

            features = compute_all_features(
                h_w, h_l, score_w, score_l,
                density_radius=self.weighter.clip_min,
            )

            current_q = 1.0
            if self.curriculum is not None:
                current_q = self.curriculum.get_current_quantile(self.state.global_step)

            weights = self.weighter(features, current_quantile=current_q).to(device)

            # Print per-sample breakdown every 50 steps (once per optimizer step)
            if (self.state.global_step % 50 == 0 and B > 1
                    and self.state.global_step != self._last_print_step):
                print(f"\n[Step {self.state.global_step}] Geometry weights breakdown:")
                for i in range(B):
                    print(
                        f"  sample {i}: "
                        f"cos_sim={features.cos_sim[i]:.3f}  "
                        f"l2={features.l2_distance[i]:.1f}  "
                        f"margin={features.margin[i]:.3f}  "
                        f"entropy={features.entropy[i]:.3f}  "
                        f"→ weight={weights[i].item():.3f}"
                    )
                print(f"  weight_std={weights.std().item():.4f}  "
                      f"weight_range=[{weights.min().item():.3f}, {weights.max().item():.3f}]")
                self._last_print_step = self.state.global_step

            self._geo_metrics["cos_sim_mean"].append(features.cos_sim.mean().item())
            self._geo_metrics["l2_dist_mean"].append(features.l2_distance.mean().item())
            self._geo_metrics["weight_mean"].append(weights.mean().item())
            self._geo_metrics["weight_std"].append(weights.std().item() if B > 1 else 0.0)
            if self.curriculum:
                self._geo_metrics["curriculum_q"].append(current_q)

        except Exception as e:
            logger.warning(f"Geometry weighting failed ({e}); using uniform weights")
            weights = torch.ones(B, device=device)

        return weights

    def log(self, logs: dict, start_time=None):
        for k, vals in self._geo_metrics.items():
            if vals:
                logs[f"geometry/{k}"] = sum(vals) / len(vals)
        self._geo_metrics.clear()
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def __del__(self):
        try:
            self.extractor.remove()
        except Exception:
            pass


def build_gadpo_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: Config,
    total_steps: int,
) -> GeometryAwareDPOTrainer:
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

    trainer = GeometryAwareDPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=dpo_cfg,
        geometry_cfg=cfg.geometry,
        curriculum_cfg=cfg.curriculum,
        total_steps=total_steps,
    )
    return trainer
