from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.sft import rollout_eval
from grading.gold_answer import normalize_gold_answer


class GoldAnswerNormalizationTest(unittest.TestCase):
    def test_deepscaler_short_answer_passes_through(self) -> None:
        self.assertEqual(normalize_gold_answer("  \\frac{3}{7}  "), "\\frac{3}{7}")

    def test_legacy_hash_marker_is_normalized_at_ingestion(self) -> None:
        self.assertEqual(normalize_gold_answer("reasoning\n#### 42"), "42")


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.assert_chat_template_args(tokenize, add_generation_prompt, enable_thinking)
        return messages[0]["content"] + "\nassistant:"

    @staticmethod
    def assert_chat_template_args(
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> None:
        if tokenize or not add_generation_prompt or enable_thinking:
            raise AssertionError("Unexpected chat template arguments.")


class RolloutEvalSampleTest(unittest.TestCase):
    def test_build_rollout_eval_samples_uses_deepscaler_fields(self) -> None:
        dataset = [{"problem": "  What is 6*7?  ", "answer": " 42 "}]

        samples = rollout_eval.build_rollout_eval_samples(
            tokenizer=FakeTokenizer(),
            eval_source_dataset=dataset,
            max_samples=0,
            subset_seed=42,
            question_field="problem",
            answer_field="answer",
        )

        self.assertEqual(samples[0]["sample_id"], 0)
        self.assertEqual(samples[0]["question"], "What is 6*7?")
        self.assertEqual(samples[0]["gold_answer"], "42")
        self.assertIn("What is 6*7?", samples[0]["prompt_text"])


class MultiRolloutEvalTest(unittest.TestCase):
    def test_multi_rollout_metrics_use_per_sample_variance_propagation(self) -> None:
        samples = [
            {"sample_id": 1, "question": "q1", "gold_answer": "yes"},
            {"sample_id": 2, "question": "q2", "gold_answer": "yes"},
        ]
        generated_texts_by_sample = [
            ["yes", "yes", "no", "no"],
            ["yes", "no", "no", "no"],
        ]

        with patch.object(
            rollout_eval,
            "load_grading_functions",
            return_value=(lambda text: text, lambda predicted, gold: predicted == gold),
        ):
            metrics, records = rollout_eval.grade_multi_rollout_predictions(
                samples,
                generated_texts_by_sample,
            )

        self.assertEqual(metrics["num_rollouts"], 4.0)
        self.assertEqual(metrics["rollout_total"], 2.0)
        self.assertAlmostEqual(metrics["rollout_correct"], 0.75)
        self.assertAlmostEqual(metrics["rollout_acc"], 0.375)
        self.assertAlmostEqual(metrics["rollout_acc_variance"], 0.109375)
        self.assertAlmostEqual(metrics["rollout_acc_std"], math.sqrt(0.109375))
        self.assertAlmostEqual(metrics["mean_sample_rollout_acc_variance"], 0.21875)

        self.assertAlmostEqual(records[0]["sample_rollout_acc_mean"], 0.5)
        self.assertAlmostEqual(records[0]["sample_rollout_acc_variance"], 0.25)
        self.assertAlmostEqual(records[1]["sample_rollout_acc_mean"], 0.25)
        self.assertAlmostEqual(records[1]["sample_rollout_acc_variance"], 0.1875)
        self.assertEqual(len(records[0]["rollouts"]), 4)
        self.assertEqual(records[0]["generated_text"], "yes")
        self.assertTrue(records[0]["is_correct"])


if __name__ == "__main__":
    unittest.main()
