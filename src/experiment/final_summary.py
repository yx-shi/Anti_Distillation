from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pre_exp.plot_curves import collect_curve_data, write_outputs
from src.vllm_dual_decoding.common import write_json

from .config import ExperimentConfig, normalize_mode
from .data_summary import fmt_float, fmt_percent, read_json, write_text
from .paths import paths_for_mode


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _last_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    return items[-1] if items else {}


def _metric(payload: dict[str, Any], name: str) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _as_float(metrics.get(name))


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _checkpoint_eval_files(analysis_dir: Path) -> list[Path]:
    return sorted(path for path in analysis_dir.glob("checkpoint_eval_*.json") if path.is_file())


def _checkpoint_entry(payload: dict[str, Any], source_file: Path, final_step: int | None) -> dict[str, Any]:
    label = payload.get("checkpoint_label")
    raw_step = payload.get("checkpoint_step")
    step = raw_step
    if step is None and label == "final_checkpoint":
        step = final_step
    return {
        "checkpoint_label": label,
        "checkpoint_step": step,
        "raw_checkpoint_step": raw_step,
        "model_name_or_path": payload.get("model_name_or_path"),
        "source_file": str(source_file),
        "max_samples": payload.get("max_samples"),
        "eligible_sample_count": payload.get("eligible_sample_count"),
        "metrics": payload.get("metrics", {}),
    }


def _best_by_acc(entries: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [entry for entry in entries if _metric(entry, "rollout_acc") is not None]
    if not valid:
        return {}
    return max(valid, key=lambda entry: float(_metric(entry, "rollout_acc") or 0.0))


def _entry_with_label(entries: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for entry in entries:
        if entry.get("checkpoint_label") == label:
            return entry
    return {}


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mode_comparison(mode_item: dict[str, Any], baseline_item: dict[str, Any]) -> dict[str, Any]:
    final_eval = mode_item.get("final_eval", {})
    baseline_final_eval = baseline_item.get("final_eval", {})
    rollout_eval = mode_item.get("rollout_eval", {})
    baseline_rollout_eval = baseline_item.get("rollout_eval", {})
    data_quality = mode_item.get("data_quality", {})
    baseline_data_quality = baseline_item.get("data_quality", {})
    return {
        "final_eval_rollout_acc_delta": _delta(
            _metric(final_eval, "rollout_acc"),
            _metric(baseline_final_eval, "rollout_acc"),
        ),
        "checkpoint_final_rollout_acc_delta": _delta(
            _metric(rollout_eval.get("final_checkpoint", {}), "rollout_acc"),
            _metric(baseline_rollout_eval.get("final_checkpoint", {}), "rollout_acc"),
        ),
        "best_checkpoint_rollout_acc_delta": _delta(
            _metric(rollout_eval.get("best", {}), "rollout_acc"),
            _metric(baseline_rollout_eval.get("best", {}), "rollout_acc"),
        ),
        "data_correct_rate_delta": _delta(
            _as_float(data_quality.get("correct_rate")),
            _as_float(baseline_data_quality.get("correct_rate")),
        ),
        "data_valid_rate_delta": _delta(
            _as_float(data_quality.get("valid_rate")),
            _as_float(baseline_data_quality.get("valid_rate")),
        ),
        "data_student_mean_nll_delta": _delta(
            _as_float((data_quality.get("student_mean_nll") or {}).get("mean")),
            _as_float((baseline_data_quality.get("student_mean_nll") or {}).get("mean")),
        ),
    }


def _curve_analysis_files(config: ExperimentConfig, modes: list[str]) -> dict[str, list[Path]]:
    analysis_files: dict[str, list[Path]] = {}
    for mode in modes:
        paths = paths_for_mode(config, mode)
        analysis_files[mode] = _checkpoint_eval_files(paths.analysis_dir)
    return analysis_files


def build_group_curve_data(config: ExperimentConfig, modes: list[str]) -> dict[str, Any]:
    normalized_modes = [normalize_mode(mode) for mode in modes]
    mode_dirs = {
        mode: paths_for_mode(config, mode).run_dir
        for mode in normalized_modes
    }
    return collect_curve_data(
        run_dir=Path(config.paths["run_root"]),
        analysis_dir=Path(config.paths["result_root"]) / "analysis",
        modes=normalized_modes,
        mode_dirs=mode_dirs,
        analysis_files=_curve_analysis_files(config, normalized_modes),
        discover_analysis_dir=False,
    )


def build_final_result_summary(
    *,
    config: ExperimentConfig,
    modes: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_modes = [normalize_mode(mode) for mode in modes]
    group_run_id = config.group_run_id_for_modes(normalized_modes)
    curve_data = build_group_curve_data(config, normalized_modes)
    summary_dir = Path(config.paths.get("summary_root", "result/summary")) / group_run_id
    data_summary_file = summary_dir / "data_quality_summary.json"
    data_summary = read_json(data_summary_file)

    mode_summaries: dict[str, Any] = {}
    for mode in normalized_modes:
        paths = paths_for_mode(config, mode)
        mode_curve = curve_data["mode_data"][mode]
        final_step = mode_curve.get("final_step")
        checkpoint_entries = [
            _checkpoint_entry(_read_json_optional(path), path, final_step)
            for path in _checkpoint_eval_files(paths.analysis_dir)
        ]
        checkpoint_entries = [entry for entry in checkpoint_entries if entry.get("metrics")]
        initial = _entry_with_label(checkpoint_entries, "000000")
        final_checkpoint = _entry_with_label(checkpoint_entries, "final_checkpoint")
        final_eval_payload = _read_json_optional(paths.final_eval_file)
        final_eval = (
            _checkpoint_entry(final_eval_payload, paths.final_eval_file, final_step)
            if final_eval_payload.get("metrics")
            else {}
        )
        mode_summaries[mode] = {
            "run_id": paths.run_id,
            "paths": {
                "run_dir": str(paths.run_dir),
                "analysis_dir": str(paths.analysis_dir),
                "distill_summary_file": str(paths.distill_summary_file),
                "eval_summary_file": str(paths.eval_summary_file),
                "checkpoint_eval_summary_file": str(paths.rollout_eval_summary_file),
                "final_eval_file": str(paths.final_eval_file),
            },
            "data_quality": (data_summary.get("modes") or {}).get(mode, {}),
            "distill_summary": read_json(paths.distill_summary_file),
            "eval_dataset_summary": read_json(paths.eval_summary_file),
            "training": {
                "final_step": final_step,
                "last_train_loss": _last_item(mode_curve.get("train_loss", [])),
                "last_validation": _last_item(mode_curve.get("eval_metrics", [])),
                "train_points": len(mode_curve.get("train_loss", [])),
                "validation_points": len(mode_curve.get("eval_metrics", [])),
            },
            "rollout_eval": {
                "summary_file": str(paths.rollout_eval_summary_file),
                "checkpoint_count": len(checkpoint_entries),
                "initial": initial,
                "best": _best_by_acc(checkpoint_entries),
                "final_checkpoint": final_checkpoint,
                "checkpoints": checkpoint_entries,
            },
            "final_eval": final_eval,
        }

    comparisons: dict[str, Any] = {}
    if "teacher_plain" in mode_summaries:
        baseline = mode_summaries["teacher_plain"]
        for mode in normalized_modes:
            if mode == "teacher_plain":
                continue
            comparisons[f"{mode}_vs_teacher_plain"] = _mode_comparison(mode_summaries[mode], baseline)

    summary = {
        "group_run_id": group_run_id,
        "config_file": str(config.path),
        "dataset": {
            "key": config.dataset.key,
            "dataset_name": config.dataset.dataset_name,
            "dataset_config_name": config.dataset.dataset_config_name,
            "train_split": config.dataset.train_split,
            "eval_split": config.dataset.eval_split,
        },
        "modes_order": normalized_modes,
        "paths": {
            "summary_dir": str(summary_dir),
            "result_root": str(config.paths["result_root"]),
            "run_root": str(config.paths["run_root"]),
            "data_quality_summary_file": str(data_summary_file),
        },
        "figures": {
            "train_loss": str(summary_dir / "train_loss_curve.svg"),
            "validation_loss_ppl": str(summary_dir / "val_loss_ppl_curve.svg"),
            "rollout_accuracy": str(summary_dir / "rollout_accuracy_curve.svg"),
        },
        "modes": mode_summaries,
        "comparisons": comparisons,
    }
    return summary, curve_data


def _fmt_metric(entry: dict[str, Any], name: str, *, percent: bool = False, digits: int = 4) -> str:
    value = _metric(entry, name)
    if percent:
        return fmt_percent(value)
    return fmt_float(value, digits)


def _mode_label(mode: str) -> str:
    return {
        "teacher_plain": "plain",
        "teacher_token_hard": "hard",
        "teacher_token_soft": "soft",
    }.get(mode, mode)


def build_final_markdown_summary(summary: dict[str, Any]) -> str:
    modes_order = list(summary.get("modes_order", summary["modes"].keys()))
    dataset = summary["dataset"]
    lines = [
        "# Final Result Summary",
        "",
        f"- group_run_id: `{summary['group_run_id']}`",
        f"- dataset: `{dataset['dataset_name']}` / `{dataset['dataset_config_name']}`",
        f"- train split: `{dataset['train_split']}`",
        f"- eval split: `{dataset['eval_split']}`",
        "",
        "## Figures",
        "",
        "![Training Loss](train_loss_curve.svg)",
        "",
        "![Validation Loss and Perplexity](val_loss_ppl_curve.svg)",
        "",
        "![Rollout Accuracy](rollout_accuracy_curve.svg)",
        "",
        "## Data Quality",
        "",
        "| mode | generated | correct | valid | mean tokens | student NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes_order:
        item = summary["modes"][mode].get("data_quality", {})
        lines.append(
            "| "
            f"{_mode_label(mode)} | "
            f"{item.get('record_count', 'na')} | "
            f"{fmt_percent(item.get('correct_rate'))} | "
            f"{fmt_percent(item.get('valid_rate'))} | "
            f"{fmt_float((item.get('num_generated_tokens') or {}).get('mean'), 2)} | "
            f"{fmt_float((item.get('student_mean_nll') or {}).get('mean'), 4)} |"
        )

    lines.extend(
        [
            "",
            "## Training And Evaluation",
            "",
            "| mode | final step | train loss | val loss | val ppl | rollout init | rollout best | rollout final@ckpt | final eval |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in modes_order:
        item = summary["modes"][mode]
        training = item["training"]
        last_train = training.get("last_train_loss") or {}
        last_val = training.get("last_validation") or {}
        rollout_eval = item.get("rollout_eval", {})
        final_eval = item.get("final_eval", {})
        lines.append(
            "| "
            f"{_mode_label(mode)} | "
            f"{training.get('final_step', 'na')} | "
            f"{fmt_float(last_train.get('train_loss'), 4)} | "
            f"{fmt_float(last_val.get('val_loss'), 4)} | "
            f"{fmt_float(last_val.get('val_ppl'), 4)} | "
            f"{_fmt_metric(rollout_eval.get('initial', {}), 'rollout_acc', percent=True)} +/- "
            f"{_fmt_metric(rollout_eval.get('initial', {}), 'rollout_acc_std', percent=True)} | "
            f"{_fmt_metric(rollout_eval.get('best', {}), 'rollout_acc', percent=True)} +/- "
            f"{_fmt_metric(rollout_eval.get('best', {}), 'rollout_acc_std', percent=True)} | "
            f"{_fmt_metric(rollout_eval.get('final_checkpoint', {}), 'rollout_acc', percent=True)} +/- "
            f"{_fmt_metric(rollout_eval.get('final_checkpoint', {}), 'rollout_acc_std', percent=True)} | "
            f"{_fmt_metric(final_eval, 'rollout_acc', percent=True)} +/- "
            f"{_fmt_metric(final_eval, 'rollout_acc_std', percent=True)} |"
        )

    lines.extend(["", "## Comparisons To Plain", ""])
    if not summary.get("comparisons"):
        lines.append("No baseline comparison available.")
    else:
        lines.extend(
            [
                "| comparison | final eval delta | checkpoint final delta | best checkpoint delta | data NLL delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, item in summary["comparisons"].items():
            lines.append(
                "| "
                f"{name} | "
                f"{fmt_percent(item.get('final_eval_rollout_acc_delta'))} | "
                f"{fmt_percent(item.get('checkpoint_final_rollout_acc_delta'))} | "
                f"{fmt_percent(item.get('best_checkpoint_rollout_acc_delta'))} | "
                f"{fmt_float(item.get('data_student_mean_nll_delta'), 4)} |"
            )

    best_mode = None
    best_acc = None
    for mode in modes_order:
        acc = _metric(summary["modes"][mode].get("final_eval", {}), "rollout_acc")
        if acc is not None and (best_acc is None or acc > best_acc):
            best_mode = mode
            best_acc = acc
    if best_mode is not None:
        lines.extend(
            [
                "",
                "## Takeaway",
                "",
                f"- Best full final eval: `{_mode_label(best_mode)}` with {fmt_percent(best_acc)} rollout accuracy.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_final_result_summary(
    summary: dict[str, Any],
    curve_data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_outputs = write_outputs(curve_data, output_dir)
    json_file = output_dir / "final_result_summary.json"
    markdown_file = output_dir / "final_result_summary.md"
    write_json(json_file, summary)
    write_text(markdown_file, build_final_markdown_summary(summary))
    outputs = {
        "json_file": json_file,
        "markdown_file": markdown_file,
        "curve_data_file": output_dir / "curve_data.json",
    }
    for path in curve_outputs:
        if path.name == "train_loss_curve.svg":
            outputs["train_loss_curve"] = path
        elif path.name == "val_loss_ppl_curve.svg":
            outputs["val_loss_ppl_curve"] = path
        elif path.name == "rollout_accuracy_curve.svg":
            outputs["rollout_accuracy_curve"] = path
    return outputs
