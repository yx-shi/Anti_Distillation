from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from src.vllm_dual_decoding.common import read_jsonl, write_json


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "median": None, "std": None}
    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def rate(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def bool_count(records: list[dict[str, Any]], key: str) -> int:
    return sum(1 for record in records if bool(record.get(key, False)))


def text_lengths(records: list[dict[str, Any]]) -> list[float]:
    return [float(len(str(record.get("candidate_text", "")))) for record in records]


def summarize_mode(records: list[dict[str, Any]], distill_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    total = len(records)
    correct_count = bool_count(records, "is_correct")
    valid_count = bool_count(records, "is_valid_candidate")
    extractable_count = bool_count(records, "is_extractable")
    truncated_count = bool_count(records, "is_generation_truncated")
    empty_count = bool_count(records, "is_empty")
    score_truncated_count = bool_count(records, "score_truncated")
    scoreable_count = sum(1 for record in records if record.get("student_mean_nll") is not None)
    valid_and_correct_count = sum(
        1
        for record in records
        if bool(record.get("is_valid_candidate", False)) and bool(record.get("is_correct", False))
    )
    sample_ids = {int(record["sample_id"]) for record in records if "sample_id" in record}
    return {
        "record_count": total,
        "sample_count": len(sample_ids),
        "correct_count": correct_count,
        "correct_rate": rate(correct_count, total),
        "valid_count": valid_count,
        "valid_rate": rate(valid_count, total),
        "valid_and_correct_count": valid_and_correct_count,
        "valid_and_correct_rate": rate(valid_and_correct_count, total),
        "extractable_count": extractable_count,
        "extractable_rate": rate(extractable_count, total),
        "generation_truncated_count": truncated_count,
        "generation_truncated_rate": rate(truncated_count, total),
        "empty_count": empty_count,
        "empty_rate": rate(empty_count, total),
        "score_truncated_count": score_truncated_count,
        "score_truncated_rate": rate(score_truncated_count, total),
        "scoreable_count": scoreable_count,
        "scoreable_rate": rate(scoreable_count, total),
        "num_generated_tokens": summarize_numeric(numeric_values(records, "num_generated_tokens")),
        "student_token_count": summarize_numeric(numeric_values(records, "student_token_count")),
        "candidate_text_chars": summarize_numeric(text_lengths(records)),
        "student_mean_nll": summarize_numeric(numeric_values(records, "student_mean_nll")),
        "distill_summary": distill_summary or {},
    }


def records_by_sample(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_sample: dict[int, dict[str, Any]] = {}
    for record in records:
        if "sample_id" not in record:
            continue
        by_sample[int(record["sample_id"])] = record
    return by_sample


def compare_modes(left_records: list[dict[str, Any]], right_records: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_sample = records_by_sample(left_records)
    right_by_sample = records_by_sample(right_records)
    shared_sample_ids = sorted(set(left_by_sample) & set(right_by_sample))
    nll_deltas: list[float] = []
    token_deltas: list[float] = []
    generated_token_deltas: list[float] = []
    correctness_delta = 0
    valid_delta = 0
    higher_nll_count = 0
    lower_nll_count = 0
    equal_nll_count = 0
    for sample_id in shared_sample_ids:
        left = left_by_sample[sample_id]
        right = right_by_sample[sample_id]
        correctness_delta += int(bool(left.get("is_correct", False))) - int(bool(right.get("is_correct", False)))
        valid_delta += int(bool(left.get("is_valid_candidate", False))) - int(bool(right.get("is_valid_candidate", False)))

        left_nll = left.get("student_mean_nll")
        right_nll = right.get("student_mean_nll")
        if left_nll is not None and right_nll is not None:
            delta = float(left_nll) - float(right_nll)
            nll_deltas.append(delta)
            higher_nll_count += int(delta > 0)
            lower_nll_count += int(delta < 0)
            equal_nll_count += int(delta == 0)

        left_tokens = left.get("student_token_count")
        right_tokens = right.get("student_token_count")
        if left_tokens is not None and right_tokens is not None:
            token_deltas.append(float(left_tokens) - float(right_tokens))

        left_generated = left.get("num_generated_tokens")
        right_generated = right.get("num_generated_tokens")
        if left_generated is not None and right_generated is not None:
            generated_token_deltas.append(float(left_generated) - float(right_generated))

    nll_pair_count = len(nll_deltas)
    return {
        "paired_sample_count": len(shared_sample_ids),
        "paired_nll_count": nll_pair_count,
        "student_mean_nll_delta": summarize_numeric(nll_deltas),
        "higher_nll_rate": rate(higher_nll_count, nll_pair_count),
        "lower_nll_rate": rate(lower_nll_count, nll_pair_count),
        "equal_nll_rate": rate(equal_nll_count, nll_pair_count),
        "student_token_count_delta": summarize_numeric(token_deltas),
        "num_generated_tokens_delta": summarize_numeric(generated_token_deltas),
        "correct_count_delta": correctness_delta,
        "valid_count_delta": valid_delta,
    }


def read_json(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def build_data_summary(
    *,
    group_run_id: str,
    modes: list[str],
    scored_files: dict[str, Path],
    distill_summary_files: dict[str, Path],
) -> dict[str, Any]:
    records_by_mode = {mode: read_jsonl(scored_files[mode]) for mode in modes}
    distill_by_mode = {mode: read_json(distill_summary_files[mode]) for mode in modes}
    mode_summaries = {
        mode: summarize_mode(records_by_mode[mode], distill_summary=distill_by_mode.get(mode))
        for mode in modes
    }
    comparisons: dict[str, Any] = {}
    if "teacher_plain" in records_by_mode:
        for mode in modes:
            if mode == "teacher_plain":
                continue
            comparisons[f"{mode}_vs_teacher_plain"] = compare_modes(
                records_by_mode[mode],
                records_by_mode["teacher_plain"],
            )
    for left, right in (("teacher_token_hard", "teacher_token_soft"), ("teacher_token_soft", "teacher_token_hard")):
        if left in records_by_mode and right in records_by_mode:
            comparisons[f"{left}_vs_{right}"] = compare_modes(records_by_mode[left], records_by_mode[right])
    return {
        "group_run_id": group_run_id,
        "modes_order": modes,
        "scored_files": {mode: str(path) for mode, path in scored_files.items()},
        "distill_summary_files": {mode: str(path) for mode, path in distill_summary_files.items()},
        "modes": mode_summaries,
        "comparisons": comparisons,
    }


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any) -> str:
    if value is None:
        return "na"
    return f"{float(value) * 100:.2f}%"


def build_markdown_summary(summary: dict[str, Any]) -> str:
    modes_order = list(summary.get("modes_order", summary["modes"].keys()))
    lines = [
        "# Data Quality Summary",
        "",
        f"- group_run_id: `{summary['group_run_id']}`",
        "",
        "## Mode Metrics",
        "",
        "| mode | records | correct | valid | truncated | gen tokens mean | score tokens mean | student NLL mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes_order:
        item = summary["modes"][mode]
        lines.append(
            "| "
            f"{mode} | "
            f"{item['record_count']} | "
            f"{fmt_percent(item['correct_rate'])} | "
            f"{fmt_percent(item['valid_rate'])} | "
            f"{fmt_percent(item['generation_truncated_rate'])} | "
            f"{fmt_float(item['num_generated_tokens']['mean'], 2)} | "
            f"{fmt_float(item['student_token_count']['mean'], 2)} | "
            f"{fmt_float(item['student_mean_nll']['mean'], 4)} |"
        )
    lines.extend(["", "## NLL Comparisons", ""])
    if not summary["comparisons"]:
        lines.append("No pairwise comparisons available.")
    else:
        lines.extend(
            [
                "| comparison | paired | mean NLL delta | higher NLL rate | mean token delta | correct count delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, item in summary["comparisons"].items():
            lines.append(
                "| "
                f"{name} | "
                f"{item['paired_sample_count']} | "
                f"{fmt_float(item['student_mean_nll_delta']['mean'], 4)} | "
                f"{fmt_percent(item['higher_nll_rate'])} | "
                f"{fmt_float(item['student_token_count_delta']['mean'], 2)} | "
                f"{item['correct_count_delta']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_data_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_file = output_dir / "data_quality_summary.json"
    markdown_file = output_dir / "data_quality_summary.md"
    write_json(json_file, summary)
    write_text(markdown_file, build_markdown_summary(summary))
    return json_file, markdown_file
