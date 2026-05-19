import os
import logging
from typing import Optional


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def init_wandb(project: str, run_name: Optional[str] = None, config: Optional[dict] = None):
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
    except ImportError:
        logging.warning("wandb not installed, skipping W&B logging")
