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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-driven token-level experiments.")
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGE_ALIASES),
        help=(
            "Stage to run: teacher_generate, student_score/students_score, "
            "build_distill, train, eval, plot/curves."
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
    stage = normalize_stage(args.stage)

    for mode in config.modes_for_cli(args.mode or None):
        launcher.run_stage(
            stage,
            mode,
            dry_run=args.dry_run,
            allow_overwrite=args.allow_overwrite,
        )


if __name__ == "__main__":
    main()
