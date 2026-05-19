from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, PeftModel

from gadpo.utils.config import Config
from gadpo.models.qlora_config import get_bnb_config, get_lora_config

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(cfg: Config) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = cfg.model.model_name
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = get_bnb_config(cfg.model) if cfg.model.load_in_4bit else None

    if cfg.model.use_flash_attention:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"  # PyTorch built-in, uses flash kernels on supported GPUs

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.model.bnb_4bit_compute_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
        dtype=torch_dtype,
    )
    model.config.use_cache = False

    return model, tokenizer


def apply_lora(model: PreTrainedModel, cfg: Config) -> PeftModel:
    lora_config = get_lora_config(cfg.lora)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_sft_checkpoint(checkpoint_dir: str, cfg: Config) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Load a saved SFT checkpoint for DPO fine-tuning."""
    logger.info(f"Loading SFT checkpoint from {checkpoint_dir}")
    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    model.config.use_cache = False
    return model, tokenizer
