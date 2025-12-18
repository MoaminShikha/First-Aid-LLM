from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from .config import AppConfig
from .data import DatasetSplits

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(cfg: AppConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
    )
    return model, tokenizer


def build_trainer(cfg: AppConfig, model, tokenizer, datasets: DatasetSplits) -> SFTTrainer:
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        evaluation_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        logging_steps=cfg.training.logging_steps,
        max_steps=cfg.training.max_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        output_dir=cfg.training.output_dir,
        report_to=cfg.training.report_to,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets.train,
        eval_dataset=datasets.validation,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    return trainer


def save_model_and_tokenizer(trainer: SFTTrainer, output_dir: str) -> Tuple[Path, Path]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    logger.info("Saved LoRA adapters and tokenizer to %s", output_dir)
    return Path(output_dir) / "adapter_config.json", Path(output_dir) / "tokenizer_config.json"


def show_memory_stats(device_id: int = 0) -> str:
    if not torch.cuda.is_available():
        return "CUDA not available; memory stats skipped."
    gpu_stats = torch.cuda.get_device_properties(device_id)
    used = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    msg = f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB. Used = {used} GB."
    logger.info(msg)
    return msg
