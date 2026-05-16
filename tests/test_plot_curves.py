from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pre_exp.common import write_json
from src.pre_exp.plot_curves import build_series, collect_curve_data, render_svg


class PlotCurvesTest(unittest.TestCase):
    def test_rollout_curve_data_keeps_std_for_step_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs"
            analysis_dir = root / "analysis"
            analysis_dir.mkdir(parents=True)
            write_json(
                analysis_dir / "checkpoint_eval_teacher_baseline_000000.json",
                {
                    "checkpoint_label": "000000",
                    "checkpoint_step": 0,
                    "model_name_or_path": "/models/teacher_baseline/base",
                    "metrics": {
                        "rollout_acc": 0.5,
                        "rollout_acc_variance": 0.01,
                        "rollout_acc_std": 0.1,
                        "rollout_correct": 1.0,
                        "rollout_total": 2.0,
                    },
                },
            )

            curve_data = collect_curve_data(run_dir, analysis_dir, ["teacher_baseline"])
            points = curve_data["mode_data"]["teacher_baseline"]["rollout_accuracy"]

        self.assertEqual(points[0]["step"], 0)
        self.assertEqual(points[0]["rollout_acc"], 0.5)
        self.assertEqual(points[0]["rollout_acc_std"], 0.1)
        self.assertEqual(points[0]["rollout_acc_variance"], 0.01)

    def test_rollout_svg_renders_standard_deviation_error_bars(self) -> None:
        series = [
            {
                "mode": "teacher_baseline",
                "points": [(0.0, 0.5, 0.1), (200.0, 0.6, 0.05)],
            }
        ]

        svg = render_svg("Rollout Accuracy", series, "rollout_acc")

        self.assertIn('class="errorbar"', svg)
        self.assertIn('class="errorcap"', svg)

    def test_build_series_includes_optional_rollout_std(self) -> None:
        mode_data = {
            "teacher_baseline": {
                "rollout_accuracy": [
                    {"step": 0, "rollout_acc": 0.5, "rollout_acc_std": 0.1}
                ]
            }
        }

        series = build_series(
            ["teacher_baseline"],
            mode_data,
            "rollout_accuracy",
            "rollout_acc",
            error_name="rollout_acc_std",
        )

        self.assertEqual(series[0]["points"], [(0.0, 0.5, 0.1)])

    def test_collect_curve_data_supports_explicit_token_level_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "smoke__ds-gsm8k__mode-teacher_plain__k-na__w-na__t-0.7__p-0.8__n-8__seed-123"
            mode_dir = root / "runs" / run_id
            analysis_file = root / "analysis" / run_id / "final_eval.json"
            mode_dir.mkdir(parents=True)
            analysis_file.parent.mkdir(parents=True)
            (mode_dir / "train.log").write_text(
                "\n".join(
                    [
                        "[train] step=1 train_loss=2.5",
                        "[eval] step=1 val_loss=2.0 val_ppl=7.4",
                        "training_done steps=2",
                    ]
                ),
                encoding="utf-8",
            )
            write_json(
                analysis_file,
                {
                    "checkpoint_label": "final_checkpoint",
                    "model_name_or_path": str(mode_dir / "final_checkpoint"),
                    "metrics": {
                        "rollout_acc": 0.25,
                        "rollout_acc_variance": 0.02,
                        "rollout_acc_std": 0.1414,
                        "rollout_correct": 2.0,
                        "rollout_total": 8.0,
                    },
                },
            )

            curve_data = collect_curve_data(
                run_dir=root / "unused_runs_root",
                analysis_dir=root / "unused_analysis_root",
                modes=["teacher_plain"],
                mode_dirs={"teacher_plain": mode_dir},
                analysis_files={"teacher_plain": analysis_file},
            )
            mode_data = curve_data["mode_data"]["teacher_plain"]

        self.assertEqual(mode_data["mode_dir"], str(mode_dir))
        self.assertEqual(mode_data["final_step"], 2)
        self.assertEqual(mode_data["train_loss"][0]["train_loss"], 2.5)
        self.assertEqual(mode_data["eval_metrics"][0]["val_loss"], 2.0)
        self.assertEqual(mode_data["rollout_accuracy"][0]["step"], 2)
        self.assertEqual(mode_data["rollout_accuracy"][0]["rollout_acc"], 0.25)


if __name__ == "__main__":
    unittest.main()
