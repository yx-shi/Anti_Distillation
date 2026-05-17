from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiment.data_summary import build_data_summary, build_markdown_summary
from src.vllm_dual_decoding.common import write_json, write_jsonl


def record(sample_id: int, mode: str, *, correct: bool, valid: bool, nll: float, tokens: int) -> dict:
    return {
        "sample_id": sample_id,
        "generation_mode": mode,
        "candidate_text": "x" * tokens,
        "is_correct": correct,
        "is_valid_candidate": valid,
        "is_extractable": valid,
        "is_empty": False,
        "is_generation_truncated": False,
        "score_truncated": False,
        "student_mean_nll": nll,
        "student_token_count": tokens,
        "num_generated_tokens": tokens + 1,
    }


class DataSummaryTest(unittest.TestCase):
    def test_build_data_summary_reports_quality_and_pairwise_nll(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mode_files = {}
            distill_summary_files = {}
            rows_by_mode = {
                "teacher_plain": [
                    record(1, "teacher_plain", correct=True, valid=True, nll=1.0, tokens=10),
                    record(2, "teacher_plain", correct=False, valid=True, nll=2.0, tokens=20),
                ],
                "teacher_token_hard": [
                    record(1, "teacher_token_hard", correct=True, valid=True, nll=1.5, tokens=11),
                    record(2, "teacher_token_hard", correct=True, valid=True, nll=2.5, tokens=21),
                ],
                "teacher_token_soft": [
                    record(1, "teacher_token_soft", correct=False, valid=False, nll=0.5, tokens=9),
                    record(2, "teacher_token_soft", correct=True, valid=True, nll=3.0, tokens=22),
                ],
            }
            for mode, rows in rows_by_mode.items():
                scored_file = root / f"{mode}.jsonl"
                distill_summary_file = root / f"{mode}.summary.json"
                write_jsonl(scored_file, rows)
                write_json(
                    distill_summary_file,
                    {
                        "mode": mode,
                        "distill_record_count": len(rows),
                        "teacher_correct_count": sum(1 for row in rows if row["is_correct"]),
                        "teacher_valid_count": sum(1 for row in rows if row["is_valid_candidate"]),
                    },
                )
                mode_files[mode] = scored_file
                distill_summary_files[mode] = distill_summary_file

            summary = build_data_summary(
                group_run_id="group",
                modes=list(rows_by_mode),
                scored_files=mode_files,
                distill_summary_files=distill_summary_files,
            )

        plain = summary["modes"]["teacher_plain"]
        hard = summary["modes"]["teacher_token_hard"]
        soft = summary["modes"]["teacher_token_soft"]
        self.assertEqual(plain["record_count"], 2)
        self.assertEqual(plain["correct_rate"], 0.5)
        self.assertEqual(hard["correct_rate"], 1.0)
        self.assertEqual(soft["valid_rate"], 0.5)
        self.assertEqual(plain["student_mean_nll"]["mean"], 1.5)
        self.assertEqual(plain["student_token_count"]["mean"], 15.0)
        self.assertEqual(summary["comparisons"]["teacher_token_hard_vs_teacher_plain"]["student_mean_nll_delta"]["mean"], 0.5)
        self.assertEqual(summary["comparisons"]["teacher_token_hard_vs_teacher_plain"]["higher_nll_rate"], 1.0)

    def test_markdown_summary_contains_mode_rows(self) -> None:
        summary = {
            "group_run_id": "group",
            "modes": {
                "teacher_plain": {
                    "record_count": 2,
                    "correct_rate": 0.5,
                    "valid_rate": 1.0,
                    "generation_truncated_rate": 0.0,
                    "num_generated_tokens": {"mean": 12.0},
                    "student_token_count": {"mean": 11.0},
                    "student_mean_nll": {"mean": 1.5},
                }
            },
            "comparisons": {},
        }

        markdown = build_markdown_summary(summary)

        self.assertIn("# Data Quality Summary", markdown)
        self.assertIn("teacher_plain", markdown)
        self.assertIn("50.00%", markdown)


if __name__ == "__main__":
    unittest.main()

