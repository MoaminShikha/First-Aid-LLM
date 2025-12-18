from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class ModelConfig:
    name: str
    max_seq_length: int
    dtype: Optional[str]
    load_in_4bit: bool


@dataclasses.dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclasses.dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    eval_strategy: str
    eval_steps: int
    logging_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    lr_scheduler_type: str
    output_dir: str
    report_to: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


@dataclasses.dataclass
class SplitConfig:
    validation_size: float
    test_size: float
    shuffle: bool


@dataclasses.dataclass
class DatasetConfig:
    name: str
    config: str
    question_field: str
    answer_field: str
    min_question_length: int
    min_answer_length: int
    max_question_length: int
    max_answer_length: int


@dataclasses.dataclass
class InferenceConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    safety_prefix: str


@dataclasses.dataclass
class AppConfig:
    run_name: str
    seed: int
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    splits: SplitConfig
    dataset: DatasetConfig
    inference: InferenceConfig

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        return cls(
            run_name=raw["run_name"],
            seed=raw["seed"],
            model=ModelConfig(**raw["model"]),
            lora=LoraConfig(**raw["lora"]),
            training=TrainingConfig(**raw["training"]),
            splits=SplitConfig(**raw["splits"]),
            dataset=DatasetConfig(**raw["dataset"]),
            inference=InferenceConfig(**raw["inference"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
