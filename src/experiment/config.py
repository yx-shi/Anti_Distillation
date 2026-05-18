from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_registry import DatasetSpec, get_dataset_spec
from .run_id import build_group_run_id, build_run_id


MODE_ALIASES = {
    "plain": "teacher_plain",
    "teacher_plain": "teacher_plain",
    "hard": "teacher_token_hard",
    "teacher_token_hard": "teacher_token_hard",
    "soft": "teacher_token_soft",
    "teacher_token_soft": "teacher_token_soft",
}
MODE_TO_GENERATION_MODE = {
    "teacher_plain": "plain",
    "teacher_token_hard": "hard",
    "teacher_token_soft": "soft",
}
STAGE_ALIASES = {
    "teacher_generate": "teacher_generate",
    "student_score": "student_score",
    "students_score": "student_score",
    "build_distill": "build_distill",
    "data_summary": "data_summary",
    "summarize_data": "data_summary",
    "analyze_data": "data_summary",
    "final_summary": "final_summary",
    "result_summary": "final_summary",
    "summarize_results": "final_summary",
    "analyze_results": "final_summary",
    "train": "train",
    "eval": "eval",
    "rollout_eval": "rollout_eval",
    "checkpoint_eval": "rollout_eval",
    "checkpoints_eval": "rollout_eval",
    "plot": "plot",
    "plots": "plot",
    "curves": "plot",
    "plot_curves": "plot",
}


def normalize_mode(mode: str) -> str:
    normalized = mode.strip()
    if normalized not in MODE_ALIASES:
        raise ValueError(
            f"Unsupported mode `{mode}`. Expected one of: {', '.join(sorted(MODE_ALIASES))}."
        )
    return MODE_ALIASES[normalized]


def normalize_stage(stage: str) -> str:
    normalized = stage.strip()
    if normalized not in STAGE_ALIASES:
        raise ValueError(
            f"Unsupported stage `{stage}`. Expected one of: {', '.join(sorted(STAGE_ALIASES))}."
        )
    return STAGE_ALIASES[normalized]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read experiment configs. "
            "Run via the `adistill-unified` conda environment."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {path}")
    return payload


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_group": "vllm_dual",
    "dataset": "deepscaler",
    "modes": ["teacher_plain", "teacher_token_hard", "teacher_token_soft"],
    "models": {
        "teacher": "/home/disk1/public_checkpoint/Qwen3-8B",
        "student": "/home/disk1/public_checkpoint/Qwen3-1.7B",
    },
    "paths": {
        "result_root": "result/vllm_dual_decoding",
        "run_root": "/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/runs",
    },
    "execution": {
        "conda_run": "",
        "python_bin": "python",
    },
    "generation": {
        "max_samples": 8000,
        "subset_seed": 42,
        "num_shards": 8,
        "gpu_ids": "0 1 2 3 4 5 6 7",
        "num_candidates": 1,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_new_tokens": 4096,
        "seed": 42,
        "prompt_batch_size": 96,
        "save_every_prompts": 96,
        "tensor_parallel_size": 1,
        "small_tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "max_num_seqs": 80,
        "gpu_memory_utilization": 0.8,
        "small_gpu_memory_utilization": 0.35,
        "hard_candidate_top_k": 20,
        "hard_candidate_top_p": 0.95,
        "soft_student_weight": 1.0,
        "soft_temperature": 1.0,
        "debug_log_interval": 16,
        "enforce_eager": False,
        "trust_remote_code": True,
        "allow_remote_model_files": False,
        "use_tqdm": True,
    },
    "scoring": {
        "batch_size": 2,
        "max_length": 8192,
        "device": "cuda:0",
        "gpu_ids": "0",
        "attn_implementation": "flash_attention_2",
        "allow_remote_model_files": False,
    },
    "training": {
        "gpu_ids": "0,1,2,3,4,5,6,7",
        "nproc_per_node": 8,
        "master_port": 29631,
        "max_steps": 1600,
        "eval_every": 200,
        "checkpoint_every": 200,
        "max_checkpoints_to_keep": 8,
        "max_length": 5120,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": 42,
        "attn_implementation": "flash_attention_2",
        "allow_remote_model_files": False,
    },
    "eval": {
        "max_samples": 2048,
        "holdout_max_samples": 1024,
        "num_rollouts": 4,
        "temperature": 0.7,
        "top_p": 0.8,
        "sampling_seed": 42,
        "max_new_tokens": 4096,
        "engine": "vllm",
        "gpu_ids": "0",
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.85,
        "trust_remote_code": True,
        "allow_remote_model_files": False,
    },
    "rollout_eval": {
        "max_samples": 1024,
        "num_rollouts": 4,
        "temperature": 0.7,
        "top_p": 0.8,
        "sampling_seed": 42,
        "max_new_tokens": 4096,
        "engine": "vllm",
        "gpu_ids": "0 1 2 3 4 5 6 7",
        "checkpoint_glob": "checkpoint-step-*",
        "include_initial": True,
        "include_final": True,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.85,
        "trust_remote_code": True,
        "allow_remote_model_files": False,
    },
    "mode_overrides": {},
}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    experiment_group: str
    dataset: DatasetSpec
    modes: list[str]
    raw: dict[str, Any]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw["execution"]

    def mode_config(self, mode: str) -> dict[str, Any]:
        normalized_mode = normalize_mode(mode)
        mode_overrides = self.raw.get("mode_overrides", {})
        merged = deepcopy(self.raw)
        merged.pop("mode_overrides", None)
        if normalized_mode in mode_overrides:
            merged = _deep_merge(merged, mode_overrides[normalized_mode])
        return merged

    def run_id_for_mode(self, mode: str) -> str:
        normalized_mode = normalize_mode(mode)
        mode_cfg = self.mode_config(normalized_mode)
        generation = mode_cfg["generation"]
        return build_run_id(
            experiment_group=self.experiment_group,
            dataset=self.dataset.key,
            mode=normalized_mode,
            hard_candidate_top_k=generation.get("hard_candidate_top_k"),
            soft_student_weight=generation.get("soft_student_weight"),
            temperature=generation.get("temperature"),
            top_p=generation.get("top_p"),
            max_samples=generation.get("max_samples"),
            subset_seed=generation.get("subset_seed"),
        )

    def group_run_id_for_modes(self, modes: list[str] | None = None) -> str:
        normalized_modes = [normalize_mode(mode) for mode in (modes or self.modes)]
        generations = [self.mode_config(mode)["generation"] for mode in normalized_modes]

        def common_value(key: str) -> Any:
            values = [generation.get(key) for generation in generations]
            first = values[0]
            return first if all(value == first for value in values) else "mixed"

        return build_group_run_id(
            experiment_group=self.experiment_group,
            dataset=self.dataset.key,
            temperature=common_value("temperature"),
            top_p=common_value("top_p"),
            max_samples=common_value("max_samples"),
            subset_seed=common_value("subset_seed"),
        )

    def modes_for_cli(self, mode: str | None) -> list[str]:
        if mode:
            return [normalize_mode(mode)]
        return list(self.modes)

    def allows_hash_answer_fallback(self) -> bool:
        return self.dataset.answer_extraction == "hash_fallback"


def _normalize_mode_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_mode, value in raw.items():
        mode = normalize_mode(str(raw_mode))
        if not isinstance(value, dict):
            raise ValueError(f"mode_overrides.{raw_mode} must be a mapping.")
        normalized[mode] = value
    return normalized


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    user_config = _load_yaml(config_path)
    merged = _deep_merge(DEFAULT_CONFIG, user_config)

    dataset_overrides = merged.pop("dataset_overrides", {})
    if not isinstance(dataset_overrides, dict):
        raise ValueError("dataset_overrides must be a mapping when provided.")
    dataset = get_dataset_spec(str(merged["dataset"]), dataset_overrides)

    modes = [normalize_mode(str(mode)) for mode in merged.get("modes", [])]
    if not modes:
        raise ValueError("Experiment config must define at least one mode.")
    merged["modes"] = modes
    merged["mode_overrides"] = _normalize_mode_overrides(merged.get("mode_overrides", {}))

    return ExperimentConfig(
        path=config_path,
        experiment_group=str(merged["experiment_group"]),
        dataset=dataset,
        modes=modes,
        raw=merged,
    )
