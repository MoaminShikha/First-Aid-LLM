#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from first_aid_llm.config import AppConfig
from first_aid_llm.data import load_and_prepare_dataset
from first_aid_llm.training import build_model_and_tokenizer, build_trainer, save_model_and_tokenizer, show_memory_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("first_aid_llm.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train First Aid Llama with Unsloth")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Where to save adapters")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    datasets = load_and_prepare_dataset(cfg)
    model, tokenizer = build_model_and_tokenizer(cfg)
    trainer = build_trainer(cfg, model, tokenizer, datasets)

    logger.info(show_memory_stats())
    trainer_stats = trainer.train()
    logger.info("Finished training: %s", trainer_stats)

    save_model_and_tokenizer(trainer, str(args.output_dir))


if __name__ == "__main__":
    main()
