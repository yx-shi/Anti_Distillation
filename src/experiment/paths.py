from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig, normalize_mode


@dataclass(frozen=True)
class ExperimentPaths:
    run_id: str
    candidate_dir: Path
    shard_dir: Path
    candidate_file: Path
    scored_file: Path
    dataset_dir: Path
    distill_file: Path
    distill_summary_file: Path
    eval_file: Path
    eval_summary_file: Path
    run_dir: Path
    analysis_dir: Path
    log_dir: Path
    curve_dir: Path
    final_eval_file: Path


def paths_for_mode(config: ExperimentConfig, mode: str) -> ExperimentPaths:
    normalized_mode = normalize_mode(mode)
    run_id = config.run_id_for_mode(normalized_mode)
    result_root = Path(config.paths["result_root"])
    run_root = Path(config.paths["run_root"])
    candidate_dir = result_root / "candidates" / run_id
    dataset_dir = result_root / "datasets" / run_id
    analysis_dir = result_root / "analysis" / run_id
    return ExperimentPaths(
        run_id=run_id,
        candidate_dir=candidate_dir,
        shard_dir=candidate_dir / "shards",
        candidate_file=candidate_dir / "candidate_pool.jsonl",
        scored_file=candidate_dir / "scored_candidates.jsonl",
        dataset_dir=dataset_dir,
        distill_file=dataset_dir / "distill.jsonl",
        distill_summary_file=dataset_dir / "distill.summary.json",
        eval_file=dataset_dir / "eval.jsonl",
        eval_summary_file=dataset_dir / "eval.summary.json",
        run_dir=run_root / run_id,
        analysis_dir=analysis_dir,
        log_dir=analysis_dir / "logs",
        curve_dir=analysis_dir / "curves",
        final_eval_file=analysis_dir / "final_eval.json",
    )
