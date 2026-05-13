from __future__ import annotations

import unittest

from src.pre_exp.select_candidates import choose_adversarial


def candidate(candidate_id: int, *, is_correct: bool, nll: float | None) -> dict:
    return {
        "sample_id": 0,
        "question": "q",
        "prompt_text": "p",
        "candidate_id": candidate_id,
        "candidate_text": f"candidate-{candidate_id}",
        "is_correct": is_correct,
        "student_mean_nll": nll,
        "student_token_count": 8 if nll is not None else 0,
    }


class SelectCandidatesTest(unittest.TestCase):
    def test_adversarial_matches_correct_baseline_correctness_before_maximizing_nll(self) -> None:
        baseline = candidate(0, is_correct=True, nll=1.0)
        selected = choose_adversarial(
            [
                baseline,
                candidate(1, is_correct=False, nll=10.0),
                candidate(2, is_correct=True, nll=3.0),
            ],
            baseline_selected=baseline,
        )

        self.assertEqual(selected["candidate_id"], 2)

    def test_adversarial_matches_wrong_baseline_correctness_before_maximizing_nll(self) -> None:
        baseline = candidate(0, is_correct=False, nll=1.0)
        selected = choose_adversarial(
            [
                baseline,
                candidate(1, is_correct=True, nll=10.0),
                candidate(2, is_correct=False, nll=3.0),
            ],
            baseline_selected=baseline,
        )

        self.assertEqual(selected["candidate_id"], 2)


if __name__ == "__main__":
    unittest.main()
