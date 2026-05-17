from __future__ import annotations

from typing import Any


def _format_component(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, str) and value.strip() == "":
        return "na"
    return str(value)


def _safe(value: Any) -> str:
    text = _format_component(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def build_run_id(
    *,
    experiment_group: str,
    dataset: str,
    mode: str,
    hard_candidate_top_k: Any,
    soft_student_weight: Any,
    temperature: Any,
    top_p: Any,
    max_samples: Any,
    subset_seed: Any,
) -> str:
    return (
        f"{_safe(experiment_group)}"
        f"__ds-{_safe(dataset)}"
        f"__mode-{_safe(mode)}"
        f"__k-{_safe(hard_candidate_top_k)}"
        f"__w-{_safe(soft_student_weight)}"
        f"__t-{_safe(temperature)}"
        f"__p-{_safe(top_p)}"
        f"__n-{_safe(max_samples)}"
        f"__seed-{_safe(subset_seed)}"
    )


def build_group_run_id(
    *,
    experiment_group: str,
    dataset: str,
    temperature: Any,
    top_p: Any,
    max_samples: Any,
    subset_seed: Any,
) -> str:
    return (
        f"{_safe(experiment_group)}"
        f"__ds-{_safe(dataset)}"
        f"__t-{_safe(temperature)}"
        f"__p-{_safe(top_p)}"
        f"__n-{_safe(max_samples)}"
        f"__seed-{_safe(subset_seed)}"
    )
