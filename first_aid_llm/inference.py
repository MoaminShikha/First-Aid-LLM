from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import GenerationConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from .config import AppConfig

logger = logging.getLogger(__name__)


def load_inference_model(model_path: str, cfg: AppConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: List[dict],
    cfg: AppConfig,
    device: Optional[str] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(inputs, device=device)

    generation_config = GenerationConfig(
        max_new_tokens=cfg.inference.max_new_tokens,
        temperature=cfg.inference.temperature,
        top_p=cfg.inference.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        generation_config=generation_config,
        use_cache=True,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return f"{cfg.inference.safety_prefix}\n\n{decoded}"
