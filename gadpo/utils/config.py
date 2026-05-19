from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_flash_attention: bool = False


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataConfig:
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    max_prompt_length: int = 256
    max_length: int = 512
    val_ratio: float = 0.05
    min_margin: float = 0.0
    max_samples: Optional[int] = None
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/run"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    beta: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    seed: int = 42
    report_to: str = "wandb"
    run_name: Optional[str] = None


@dataclass
class GeometryConfig:
    cos_sim_scale: float = 1.0
    distance_scale: float = 1.0
    weight_temperature: float = 2.0
    clip_min: float = 0.1
    clip_max: float = 5.0
    density_radius: float = 0.5
    density_k: int = 4


@dataclass
class CurriculumConfig:
    enabled: bool = True
    strategy: str = "cosine"
    start_quantile: float = 0.3
    end_quantile: float = 1.0
    warmup_ratio: float = 0.2


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


def _update_dataclass(obj, d: dict):
    for k, v in d.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _update_dataclass(attr, v)
            else:
                setattr(obj, k, v)


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()

    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    if d:
        _update_dataclass(cfg, d)

    if overrides:
        for override in overrides:
            key_path, _, value_str = override.partition("=")
            keys = key_path.strip().split(".")
            obj = cfg
            for k in keys[:-1]:
                obj = getattr(obj, k)
            current = getattr(obj, keys[-1])
            if isinstance(current, bool):
                value = value_str.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value_str)
            elif isinstance(current, float):
                value = float(value_str)
            elif isinstance(current, list):
                value = yaml.safe_load(value_str)
            else:
                value = value_str
            setattr(obj, keys[-1], value)

    return cfg


def parse_args_and_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
