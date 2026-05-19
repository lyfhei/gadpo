import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

from gadpo.utils.config import ModelConfig, LoraConfig as LoraCfg


def get_bnb_config(model_cfg: ModelConfig) -> BitsAndBytesConfig:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    compute_dtype = dtype_map.get(model_cfg.bnb_4bit_compute_dtype, torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def get_lora_config(lora_cfg: LoraCfg) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
