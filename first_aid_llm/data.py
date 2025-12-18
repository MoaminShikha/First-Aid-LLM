from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    train: Dataset
    validation: Dataset
    test: Dataset


def load_and_prepare_dataset(cfg: AppConfig) -> DatasetSplits:
    raw = load_dataset(cfg.dataset.name, cfg.dataset.config, split="train")
    logger.info("Loaded dataset %s with %d rows", cfg.dataset.name, len(raw))

    cleaned = _clean_dataset(raw, cfg)
    train_ds, val_ds, test_ds = _split_dataset(cleaned, cfg)

    formatted_train = _format_dataset(train_ds, cfg)
    formatted_val = _format_dataset(val_ds, cfg)
    formatted_test = _format_dataset(test_ds, cfg)

    return DatasetSplits(train=formatted_train, validation=formatted_val, test=formatted_test)


def _clean_dataset(dataset: Dataset, cfg: AppConfig) -> Dataset:
    df = dataset.to_pandas()
    initial = len(df)

    df = df.dropna(subset=[cfg.dataset.question_field, cfg.dataset.answer_field])
    df = df[df[cfg.dataset.question_field].str.len() >= cfg.dataset.min_question_length]
    df = df[df[cfg.dataset.answer_field].str.len() >= cfg.dataset.min_answer_length]
    df = df[df[cfg.dataset.question_field].str.len() <= cfg.dataset.max_question_length]
    df = df[df[cfg.dataset.answer_field].str.len() <= cfg.dataset.max_answer_length]
    df = df.drop_duplicates(subset=[cfg.dataset.question_field])
    df = df.reset_index(drop=True)

    logger.info(
        "Cleaned dataset: %d -> %d rows (removed %d)",
        initial,
        len(df),
        initial - len(df),
    )
    return Dataset.from_pandas(df, preserve_index=False)


def _split_dataset(dataset: Dataset, cfg: AppConfig) -> Tuple[Dataset, Dataset, Dataset]:
    dataset = dataset.train_test_split(
        test_size=cfg.splits.validation_size + cfg.splits.test_size,
        shuffle=cfg.splits.shuffle,
        seed=cfg.seed,
    )
    temp = dataset["test"].train_test_split(
        test_size=cfg.splits.test_size / (cfg.splits.validation_size + cfg.splits.test_size),
        shuffle=cfg.splits.shuffle,
        seed=cfg.seed,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(dataset["train"]),
        len(temp["train"]),
        len(temp["test"]),
    )
    return dataset["train"], temp["train"], temp["test"]


def _format_dataset(dataset: Dataset, cfg: AppConfig) -> Dataset:
    def _format_examples(batch: Dict[str, List[str]]) -> Dict[str, Iterable[str]]:
        questions = batch[cfg.dataset.question_field]
        answers = batch[cfg.dataset.answer_field]
        texts: List[str] = []

        for question, answer in zip(questions, answers):
            q = (question or "").strip() or "What is a basic health tip?"
            a = (answer or "").strip() or "Stay hydrated and get enough rest."
            text = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{a}<|eot_id|>"
            )
            texts.append(text)

        return {"text": texts}

    return dataset.map(_format_examples, batched=True, remove_columns=dataset.column_names)
