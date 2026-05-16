from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    dataset_config_name: str
    train_split: str
    eval_split: str
    question_field: str
    answer_field: str
    completion_field: str
    answer_extraction: str
    eval_exclude_train_subset: bool

    def with_overrides(self, overrides: dict[str, Any]) -> "DatasetSpec":
        allowed = set(self.__dataclass_fields__)
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"Unknown dataset override field(s): {', '.join(unknown)}")
        return replace(self, **overrides)


_REGISTRY: dict[str, DatasetSpec] = {
    "deepscaler": DatasetSpec(
        key="deepscaler",
        dataset_name="agentica-org/DeepScaleR-Preview-Dataset",
        dataset_config_name="default",
        train_split="train",
        eval_split="train",
        question_field="problem",
        answer_field="answer",
        completion_field="solution",
        answer_extraction="boxed",
        eval_exclude_train_subset=True,
    ),
    "gsm8k": DatasetSpec(
        key="gsm8k",
        dataset_name="openai/gsm8k",
        dataset_config_name="main",
        train_split="train",
        eval_split="test",
        question_field="question",
        answer_field="answer",
        completion_field="answer",
        answer_extraction="hash_fallback",
        eval_exclude_train_subset=False,
    ),
}


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_dataset_spec(key: str, overrides: dict[str, Any] | None = None) -> DatasetSpec:
    normalized_key = key.strip().lower()
    if normalized_key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset `{key}`. Available datasets: {', '.join(available_datasets())}"
        )
    spec = _REGISTRY[normalized_key]
    if overrides:
        spec = spec.with_overrides(overrides)
    return spec

