#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from first_aid_llm.config import AppConfig
from first_aid_llm.inference import generate_response, load_inference_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("first_aid_llm.infer")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with First Aid Llama")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config")
    parser.add_argument("--model_path", type=str, required=True, help="Path or hub id of the fine-tuned model")
    parser.add_argument("--prompt", type=str, help="User prompt for inference")
    parser.add_argument("--messages_json", type=str, help="Optional JSON list of chat messages")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig.from_yaml(args.config)

    if not args.prompt and not args.messages_json:
        raise SystemExit("Provide --prompt or --messages_json")

    messages = (
        json.loads(args.messages_json)
        if args.messages_json
        else [{"role": "user", "content": args.prompt}]
    )

    model, tokenizer = load_inference_model(args.model_path, cfg)
    response = generate_response(model, tokenizer, messages, cfg)
    print(response)


if __name__ == "__main__":
    main()
