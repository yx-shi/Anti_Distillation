from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import read_jsonl, write_jsonl


DEFAULT_INPUT_FILE = "result/pre_exp/candidates/smoke/scored_candidates.jsonl"
DEFAULT_OUTPUT_DIR = "result/pre_exp/datasets/smoke/selections"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select baseline/adversarial candidates from scored teacher outputs.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: int(item.get("candidate_id", 0)))


def first_valid_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid_candidates = [item for item in candidates if item.get("is_valid_candidate", False)]
    if not valid_candidates:
        raise ValueError("No valid candidate is available for this sample.")
    return sort_candidates(valid_candidates)[0]


def first_nonempty_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """在所有候选里退化选择“第一个非空回答”。

    128 条真实 smoke 数据里会遇到一种情况：
    - teacher 给出的回答看起来是合理的完整自然语言
    - 但没有落成当前 extractor 能识别的 `#### ...` / boxed / formula 形式
    - 于是它既不是 `is_valid_candidate`，也不是 `is_correct`

    如果这时直接报错，整条 smoke pipeline 会被少数格式问题卡死。
    因此这里提供一个更保守的兜底：
    至少保留“第一个非空候选”，让训练链路继续跑，同时通过 fallback_reason
    明确记录这是一条低置信度样本。
    """

    nonempty_candidates = [item for item in candidates if not item.get("is_empty", False)]
    if not nonempty_candidates:
        raise ValueError("No non-empty candidate is available for this sample.")
    return sort_candidates(nonempty_candidates)[0]


def build_selection_record(
    selected: dict[str, Any],
    *,
    selection_mode: str,
    teacher_candidate_count: int,
    fallback_reason: str,
) -> dict[str, Any]:
    """把被选中的候选转换成后续构建 distill JSONL 所需的结构。"""

    completion_text = str(selected.get("candidate_text_clean") or selected.get("candidate_text") or "").strip()
    return {
        "sample_id": selected["sample_id"],
        "question": selected["question"],
        "prompt_text": selected["prompt_text"],
        "completion": completion_text,
        "selection_mode": selection_mode,
        "selected_candidate_id": selected["candidate_id"],
        "teacher_candidate_count": teacher_candidate_count,
        "teacher_answer_correct": bool(selected.get("is_correct", False)),
        "student_mean_nll": selected.get("student_mean_nll"),
        "student_token_count": int(selected.get("student_token_count", 0)),
        "fallback_reason": fallback_reason,
    }


def choose_adversarial(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    correct_candidates = [item for item in candidates if item.get("is_correct", False)]
    if not correct_candidates:
        raise ValueError("No correct candidate is available for adversarial selection.")

    # 这里优先选择 student_mean_nll 最大的正确候选。
    # 如果 NLL 并列，再按 candidate_id 升序打破平局，保证结果可复现。
    return max(
        correct_candidates,
        key=lambda item: (
            float(item.get("student_mean_nll") if item.get("student_mean_nll") is not None else float("-inf")),
            -int(item.get("candidate_id", 0)),
        ),
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    records = read_jsonl(args.input_file)
    grouped_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_candidates[int(record["sample_id"])].append(record)

    baseline_records: list[dict[str, Any]] = []
    adversarial_records: list[dict[str, Any]] = []

    for sample_id in sorted(grouped_candidates):
        candidates = sort_candidates(grouped_candidates[sample_id])
        teacher_candidate_count = len(candidates)

        baseline_fallback_reason = ""
        correct_candidates = [item for item in candidates if item.get("is_correct", False)]
        if correct_candidates:
            baseline_selected = correct_candidates[0]
        else:
            try:
                baseline_selected = first_valid_candidate(candidates)
                baseline_fallback_reason = "no_correct_candidate"
            except ValueError:
                baseline_selected = first_nonempty_candidate(candidates)
                baseline_fallback_reason = "no_valid_candidate"

        baseline_records.append(
            build_selection_record(
                baseline_selected,
                selection_mode="teacher_baseline",
                teacher_candidate_count=teacher_candidate_count,
                fallback_reason=baseline_fallback_reason,
            )
        )

        if correct_candidates:
            adversarial_selected = choose_adversarial(candidates)
            adversarial_fallback_reason = ""
        else:
            adversarial_selected = baseline_selected
            adversarial_fallback_reason = baseline_fallback_reason or "no_correct_candidate"

        adversarial_records.append(
            build_selection_record(
                adversarial_selected,
                selection_mode="teacher_adversarial",
                teacher_candidate_count=teacher_candidate_count,
                fallback_reason=adversarial_fallback_reason,
            )
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "teacher_baseline.selected.jsonl", baseline_records)
    write_jsonl(output_dir / "teacher_adversarial.selected.jsonl", adversarial_records)

    print(
        "select_candidates_done "
        f"samples={len(grouped_candidates)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
