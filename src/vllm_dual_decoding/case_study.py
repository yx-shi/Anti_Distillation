from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_dual_decoding.analyze_generation_modes import parse_mode_files
from vllm_dual_decoding.common import read_jsonl, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact case study for token-level vLLM-dual smoke outputs."
    )
    parser.add_argument("--mode-file", action="append", default=[], metavar="MODE=PATH")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--json-output-file", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--snippet-chars", type=int, default=900)
    return parser


def index_records(mode_files: dict[str, Path]) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for mode, path in mode_files.items():
        indexed[mode] = {
            int(record["sample_id"]): record
            for record in read_jsonl(path)
            if "sample_id" in record
        }
    return indexed


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nll_gap(record: dict[str, Any], plain: dict[str, Any]) -> float | None:
    value = as_float(record.get("student_mean_nll"))
    plain_value = as_float(plain.get("student_mean_nll"))
    if value is None or plain_value is None:
        return None
    return value - plain_value


def clipped_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return text[:head_chars].rstrip() + "\n...\n" + text[-tail_chars:].lstrip()


def record_summary(record: dict[str, Any], plain: dict[str, Any] | None = None) -> dict[str, Any]:
    gap = nll_gap(record, plain) if plain is not None else None
    return {
        "mode": record.get("generation_mode"),
        "worker_cls": record.get("worker_cls"),
        "adversarial_mode": record.get("adversarial_mode"),
        "finish_reason": record.get("finish_reason"),
        "is_truncated": record.get("is_generation_truncated"),
        "is_extractable": record.get("is_extractable"),
        "is_correct": record.get("is_correct"),
        "extracted_answer": record.get("extracted_answer"),
        "student_mean_nll": record.get("student_mean_nll"),
        "student_nll_gap_vs_plain": gap,
        "num_generated_tokens": record.get("num_generated_tokens"),
    }


def score_case(sample_records: dict[str, dict[str, Any]]) -> float:
    plain = sample_records["teacher_plain"]
    scores: list[float] = []
    for mode in ("teacher_token_hard", "teacher_token_soft"):
        record = sample_records.get(mode)
        if record is None:
            continue
        gap = nll_gap(record, plain)
        if gap is not None:
            scores.append(abs(gap))
        if record.get("extracted_answer") != plain.get("extracted_answer"):
            scores.append(0.25)
        if record.get("is_correct") != plain.get("is_correct"):
            scores.append(0.5)
        if record.get("is_generation_truncated") != plain.get("is_generation_truncated"):
            scores.append(0.2)
    return sum(scores)


def select_cases(indexed: dict[str, dict[int, dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    required_modes = ("teacher_plain", "teacher_token_hard", "teacher_token_soft")
    common_ids = set(indexed.get(required_modes[0], {}))
    for mode in required_modes[1:]:
        common_ids &= set(indexed.get(mode, {}))

    cases: list[dict[str, Any]] = []
    for sample_id in sorted(common_ids):
        sample_records = {mode: indexed[mode][sample_id] for mode in required_modes}
        plain = sample_records["teacher_plain"]
        cases.append(
            {
                "sample_id": sample_id,
                "score": score_case(sample_records),
                "question": plain.get("question"),
                "gold_answer": plain.get("gold_answer"),
                "summaries": {
                    mode: record_summary(record, plain if mode != "teacher_plain" else None)
                    for mode, record in sample_records.items()
                },
                "texts": {
                    mode: record.get("candidate_text", "")
                    for mode, record in sample_records.items()
                },
            }
        )

    cases.sort(key=lambda item: (float(item["score"]), int(item["sample_id"])), reverse=True)
    return cases[:limit]


def markdown_for_cases(cases: list[dict[str, Any]], snippet_chars: int) -> str:
    lines = [
        "# vLLM-Dual Smoke Case Study",
        "",
        "Selected by large Student NLL gap, answer/correctness disagreement, or truncation disagreement between plain and adversarial decoding.",
        "",
    ]
    for idx, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## Case {idx}: sample_id={case['sample_id']}",
                "",
                f"- gold_answer: `{case.get('gold_answer')}`",
                f"- selection_score: `{case.get('score'):.4f}`",
                "",
                "### Question",
                "",
                str(case.get("question", "")).strip(),
                "",
                "### Mode Summary",
                "",
                "| mode | adv | worker | truncated | extractable | correct | answer | nll | gap_vs_plain | tokens |",
                "|---|---|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        for mode, summary in case["summaries"].items():
            lines.append(
                "| {mode} | {adv} | {worker} | {trunc} | {extractable} | {correct} | {answer} | {nll} | {gap} | {tokens} |".format(
                    mode=mode,
                    adv=summary.get("adversarial_mode"),
                    worker=summary.get("worker_cls"),
                    trunc=summary.get("is_truncated"),
                    extractable=summary.get("is_extractable"),
                    correct=summary.get("is_correct"),
                    answer=summary.get("extracted_answer"),
                    nll=summary.get("student_mean_nll"),
                    gap=summary.get("student_nll_gap_vs_plain"),
                    tokens=summary.get("num_generated_tokens"),
                )
            )
        for mode, text in case["texts"].items():
            lines.extend(
                [
                    "",
                    f"### {mode} Output Snippet",
                    "",
                    "```text",
                    textwrap.dedent(clipped_text(str(text), snippet_chars)),
                    "```",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()
    mode_files = parse_mode_files(args.mode_file)
    indexed = index_records(mode_files)
    cases = select_cases(indexed, args.limit)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_for_cases(cases, args.snippet_chars), encoding="utf-8")

    write_json(args.json_output_file, {"cases": cases})
    print(
        "vllm_dual_case_study_done "
        f"cases={len(cases)} output_file={args.output_file} json_output_file={args.json_output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
