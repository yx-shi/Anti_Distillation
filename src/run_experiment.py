from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
SRC_ROOT = CURRENT_FILE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.experiment.config import STAGE_ALIASES, load_experiment_config, normalize_stage
from src.experiment.launcher import ExperimentLauncher


GROUP_STAGES = {"data_summary", "final_summary"}


def parse_stage_list(raw_stage: str) -> list[str]:
    raw_items = raw_stage.split(",")
    stages: list[str] = []
    for raw_item in raw_items:
        if not raw_item.strip():
            raise ValueError(
                "Empty stage in --stage list. Use comma-separated stage names like "
                "teacher_generate,student_score,build_distill."
            )
        stages.append(normalize_stage(raw_item))
    return stages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-driven token-level experiments.")
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml.")
    parser.add_argument(
        "--stage",
        required=True,
        metavar="STAGE[,STAGE...]",
        help=(
            "Stage(s) to run: teacher_generate, student_score/students_score, "
            "build_distill, train, rollout_eval/checkpoint_eval, eval, plot/curves, "
            "data_summary, final_summary/result_summary. "
            f"Known aliases: {', '.join(sorted(STAGE_ALIASES))}."
        ),
    )
    parser.add_argument("--mode", default="", help="Optional mode; defaults to all modes in the YAML.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_experiment_config(args.config)
    launcher = ExperimentLauncher(config)
    stages = parse_stage_list(args.stage)
    modes = config.modes_for_cli(args.mode or None)

    for stage in stages:
        if stage in GROUP_STAGES:
            if stage == "data_summary":
                launcher.run_data_summary(
                    modes,
                    dry_run=args.dry_run,
                    allow_overwrite=args.allow_overwrite,
                )
            elif stage == "final_summary":
                launcher.run_final_summary(
                    modes,
                    dry_run=args.dry_run,
                    allow_overwrite=args.allow_overwrite,
                )
            else:
                raise ValueError(f"Unsupported group stage: {stage}")
        else:
            for mode in modes:
                launcher.run_stage(
                    stage,
                    mode,
                    dry_run=args.dry_run,
                    allow_overwrite=args.allow_overwrite,
                )


if __name__ == "__main__":
    main()
