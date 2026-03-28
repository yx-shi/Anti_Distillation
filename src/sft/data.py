from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# `datasets`/`pyarrow` 需要优先于 torch 导入，避免 Conda 环境里的新 libstdc++
# 被系统默认版本覆盖，从而触发 `GLIBCXX_x.x.x not found`。
from datasets import load_dataset as _preload_datasets  # noqa: F401
import torch
from torch.utils.data import Dataset

from sft.config import TrainConfig


def format_gsm8k_sample(example: dict[str, Any]) -> dict[str, str]:
    """Convert a GSM8K record into a prompt/completion pair used by the SFT pipeline."""

    question = example["question"].strip()
    answer = example["answer"].strip()
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    return {
        "prompt": prompt,
        "completion": answer,
        "full_text": prompt + answer,
    }


class SupervisedFineTuningDataset(Dataset):
    """Tokenize one sample at a time so the DataLoader can assemble dynamic batches."""

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = format_gsm8k_sample(self.dataset[idx])

        prompt_ids = self.tokenizer(
            item["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            item["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        return {
            "prompt_ids": prompt_ids,
            "input_ids": full_ids,
        }


@dataclass
class SupervisedFineTuningCollator:
    """Pad the batch and optionally mask the prompt so loss is only computed on the answer."""

    tokenizer: Any
    max_length: int
    ignore_index: int = -100
    train_on_prompt: bool = False

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch_input_ids = [feature["input_ids"][: self.max_length] for feature in features]
        batch_prompt_ids = [feature["prompt_ids"][: self.max_length] for feature in features]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before building batches.")

        max_seq_len = max(len(input_ids) for input_ids in batch_input_ids)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for input_ids, prompt_ids in zip(batch_input_ids, batch_prompt_ids):
            seq_len = len(input_ids)
            pad_len = max_seq_len - seq_len

            padded_input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
            labels = padded_input_ids.copy()

            if not self.train_on_prompt:
                prompt_len = min(len(prompt_ids), seq_len)
                labels[:prompt_len] = [self.ignore_index] * prompt_len

            labels[seq_len:] = [self.ignore_index] * pad_len

            input_ids_list.append(torch.tensor(padded_input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }


def unwrap_external_dataset(dataset: Any) -> Any:
    """Normalize third-party dataset wrappers into an indexable dataset when possible."""

    if hasattr(dataset, "to_hf_dataset"):
        return dataset.to_hf_dataset()
    if hasattr(dataset, "_hf_ds"):
        return dataset._hf_ds
    return dataset


def load_huggingface_dataset_splits(config: TrainConfig) -> tuple[Any, Any]:
    """Load GSM8K from Hugging Face as a fallback backend."""

    from datasets import load_dataset

    raw_dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    return raw_dataset[config.train_split], raw_dataset[config.eval_split]


def load_modelscope_dataset_splits(config: TrainConfig) -> tuple[Any, Any]:
    """Load GSM8K from ModelScope with a lazy import so the dependency is only required when selected."""

    modelscope_root = os.path.abspath(os.path.join(os.getcwd(), ".modelscope"))
    os.makedirs(modelscope_root, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", modelscope_root)
    os.environ.setdefault(
        "MODELSCOPE_CREDENTIALS_PATH",
        os.path.join(modelscope_root, "credentials"),
    )

    try:
        # 先显式导入 datasets/pyarrow，避免后续由 modelscope 间接导入时
        # 又回到系统 libstdc++，触发 GLIBCXX 版本冲突。
        from datasets import load_dataset as _hf_load_dataset  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to preload `datasets` before importing ModelScope. "
            "Check the training environment's pyarrow/libstdc++ compatibility."
        ) from exc

    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as exc:
        if exc.name != "modelscope":
            raise
        raise ImportError(
            "ModelScope dataset backend is selected, but `modelscope` is not installed. "
            "Install it with `pip install modelscope` in the training environment."
        ) from exc

    def load_one_split(split: str) -> Any:
        load_kwargs = {
            "namespace": config.dataset_namespace,
            "split": split,
            "trust_remote_code": config.modelscope_trust_remote_code,
        }
        if config.dataset_config_name:
            load_kwargs["subset_name"] = config.dataset_config_name

        try:
            dataset = MsDataset.load(config.dataset_name, **load_kwargs)
        except Exception:
            if "subset_name" not in load_kwargs:
                raise
            retry_kwargs = dict(load_kwargs)
            retry_kwargs.pop("subset_name")
            dataset = MsDataset.load(config.dataset_name, **retry_kwargs)

        return unwrap_external_dataset(dataset)

    return load_one_split(config.train_split), load_one_split(config.eval_split)


def load_supervised_dataset_splits(config: TrainConfig) -> tuple[Any, Any]:
    """Dispatch dataset loading to the configured backend."""

    if config.dataset_backend == "modelscope":
        return load_modelscope_dataset_splits(config)
    if config.dataset_backend == "huggingface":
        return load_huggingface_dataset_splits(config)

    raise ValueError(f"Unsupported dataset backend: {config.dataset_backend}")
