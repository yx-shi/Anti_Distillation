from __future__ import annotations

import unittest

from grading.extract_ans import extract_final_ans


class AnswerExtractionTest(unittest.TestCase):
    def test_extracts_last_boxed_answer(self) -> None:
        self.assertEqual(
            extract_final_ans(r"work \boxed{1} then final \boxed{\frac{3}{7}}"),
            r"\frac{3}{7}",
        )

    def test_extracts_final_answer_marker(self) -> None:
        self.assertEqual(extract_final_ans("reasoning\nFinal Answer: 42"), "42")

    def test_hash_marker_is_ignored_by_default(self) -> None:
        self.assertIsNone(extract_final_ans("reasoning\n#### 42"))

    def test_hash_marker_can_be_enabled_for_gsm8k_compatibility(self) -> None:
        self.assertEqual(
            extract_final_ans("reasoning\n#### 42", allow_hash_fallback=True),
            "42",
        )


if __name__ == "__main__":
    unittest.main()
