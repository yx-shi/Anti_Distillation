from __future__ import annotations

import unittest

from src.run_experiment import parse_stage_list


class RunExperimentCliTest(unittest.TestCase):
    def test_parse_stage_list_accepts_comma_separated_aliases(self) -> None:
        self.assertEqual(
            parse_stage_list("teacher_generate,students_score,build_distill,analyze_data"),
            ["teacher_generate", "student_score", "build_distill", "data_summary"],
        )

    def test_parse_stage_list_rejects_empty_items(self) -> None:
        with self.assertRaises(ValueError):
            parse_stage_list("teacher_generate,,build_distill")


if __name__ == "__main__":
    unittest.main()
