from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_MODES = ("teacher_baseline", "teacher_adversarial")
TRAIN_LOSS_RE = re.compile(r"\bstep=(?P<step>\d+)\b.*\btrain_loss=(?P<value>[-+0-9.eE]+)")
EVAL_RE = re.compile(
    r"\bstep=(?P<step>\d+)\b.*\bval_loss=(?P<loss>[-+0-9.eE]+)\b.*\bval_ppl=(?P<ppl>[-+0-9.eE]+)"
)
TRAINING_DONE_RE = re.compile(r"\btraining_done\b.*\bsteps=(?P<step>\d+)\b")
CHECKPOINT_STEP_RE = re.compile(r"checkpoint-step-(?P<step>\d+)")
COLORS = {
    "teacher_baseline": "#2563eb",
    "teacher_adversarial": "#dc2626",
}
FALLBACK_COLORS = ("#059669", "#7c3aed", "#ea580c", "#0891b2")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect pre-experiment train/eval/rollout curves and emit JSON plus SVGs."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    return parser


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def checkpoint_step_from_path(path: Path) -> int | None:
    match = CHECKPOINT_STEP_RE.search(str(path))
    if not match:
        return None
    return int(match.group("step"))


def parse_train_log(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]], int | None]:
    train_loss: list[dict[str, float]] = []
    eval_metrics: list[dict[str, float]] = []
    final_step: int | None = None
    if not path.is_file():
        return train_loss, eval_metrics, final_step

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        done_match = TRAINING_DONE_RE.search(raw_line)
        if done_match:
            final_step = int(done_match.group("step"))

        train_match = TRAIN_LOSS_RE.search(raw_line)
        if train_match:
            value = as_float(train_match.group("value"))
            if value is not None:
                train_loss.append({"step": int(train_match.group("step")), "train_loss": value})

        eval_match = EVAL_RE.search(raw_line)
        if eval_match:
            val_loss = as_float(eval_match.group("loss"))
            val_ppl = as_float(eval_match.group("ppl"))
            if val_loss is not None and val_ppl is not None:
                eval_metrics.append(
                    {
                        "step": int(eval_match.group("step")),
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "source": "train.log",
                    }
                )

    return dedupe_points(train_loss, ("step",), "train_loss"), dedupe_points(
        eval_metrics,
        ("step",),
        "val_loss",
    ), final_step


def parse_train_config(path: Path) -> int | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    return as_int(payload.get("max_steps"))


def load_mode_checkpoint_metrics(mode_dir: Path, final_step: int | None) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for metrics_path in sorted(mode_dir.glob("checkpoint-step-*/metrics.json")):
        metrics = read_json(metrics_path)
        if not isinstance(metrics, dict):
            continue
        step = checkpoint_step_from_path(metrics_path)
        val_loss = as_float(metrics.get("val_loss", metrics.get("loss")))
        val_ppl = as_float(metrics.get("val_ppl", metrics.get("ppl")))
        if step is not None and (val_loss is not None or val_ppl is not None):
            points.append(
                {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "source": str(metrics_path),
                }
            )

    final_metrics_path = mode_dir / "final_metrics.json"
    if final_metrics_path.is_file():
        metrics = read_json(final_metrics_path)
        if isinstance(metrics, dict):
            val_loss = as_float(metrics.get("val_loss", metrics.get("loss")))
            val_ppl = as_float(metrics.get("val_ppl", metrics.get("ppl")))
            if final_step is not None and (val_loss is not None or val_ppl is not None):
                points.append(
                    {
                        "step": final_step,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "source": str(final_metrics_path),
                    }
                )
    return dedupe_points(points, ("step",), "val_loss")


def infer_mode_from_payload(path: Path, payload: dict[str, Any], modes: list[str]) -> str | None:
    haystack = " ".join(
        [
            path.name,
            str(payload.get("model_name_or_path", "")),
            str(payload.get("output_file", "")),
        ]
    )
    matches = [mode for mode in modes if mode in haystack]
    if len(matches) == 1:
        return matches[0]
    return None


def rollout_step(payload: dict[str, Any], path: Path, final_steps: dict[str, int | None], mode: str) -> int | None:
    step = as_int(payload.get("checkpoint_step"))
    if step is not None:
        return step
    label = str(payload.get("checkpoint_label", ""))
    if label and label != "None":
        label_step = as_int(label)
        if label_step is not None:
            return label_step
    path_step = checkpoint_step_from_path(path)
    if path_step is not None:
        return path_step
    if label == "final_checkpoint" or "final" in path.name:
        return final_steps.get(mode)
    return None


def load_rollout_metrics(
    analysis_dir: Path,
    modes: list[str],
    final_steps: dict[str, int | None],
) -> dict[str, list[dict[str, Any]]]:
    by_mode = {mode: [] for mode in modes}
    if not analysis_dir.is_dir():
        return by_mode

    for json_path in sorted(analysis_dir.glob("*.json")):
        try:
            payload = read_json(json_path)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            continue
        rollout_acc = as_float(payload["metrics"].get("rollout_acc"))
        if rollout_acc is None:
            continue
        mode = infer_mode_from_payload(json_path, payload, modes)
        if mode is None:
            continue
        step = rollout_step(payload, json_path, final_steps, mode)
        if step is None:
            continue
        by_mode[mode].append(
            {
                "step": step,
                "rollout_acc": rollout_acc,
                "rollout_acc_variance": as_float(payload["metrics"].get("rollout_acc_variance")),
                "rollout_acc_std": as_float(payload["metrics"].get("rollout_acc_std")),
                "rollout_correct": as_float(payload["metrics"].get("rollout_correct")),
                "rollout_total": as_float(payload["metrics"].get("rollout_total")),
                "source": str(json_path),
            }
        )

    for mode in modes:
        by_mode[mode] = dedupe_rollout_points(by_mode[mode])
    return by_mode


def dedupe_rollout_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for point in points:
        step = as_int(point.get("step"))
        if step is None or point.get("rollout_acc") is None:
            continue
        current = by_step.get(step)
        current_total = as_float(current.get("rollout_total")) if current is not None else None
        point_total = as_float(point.get("rollout_total"))
        if current is None or (point_total or 0.0) >= (current_total or 0.0):
            by_step[step] = point
    return sorted(by_step.values(), key=lambda item: item["step"])


def dedupe_points(
    points: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    value_field: str,
) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for point in points:
        if point.get(value_field) is None:
            continue
        key = tuple(point.get(field) for field in key_fields)
        seen[key] = point
    return sorted(seen.values(), key=lambda item: (item.get("step", 0), str(item.get("source", ""))))


def collect_curve_data(run_dir: Path, analysis_dir: Path, modes: list[str]) -> dict[str, Any]:
    mode_data: dict[str, Any] = {}
    final_steps: dict[str, int | None] = {}

    for mode in modes:
        mode_dir = run_dir / mode
        train_loss, log_eval, log_final_step = parse_train_log(mode_dir / "train.log")
        config_final_step = parse_train_config(mode_dir / "train_config.json")
        observed_steps = [as_int(point.get("step")) for point in train_loss + log_eval]
        observed_steps = [step for step in observed_steps if step is not None]
        final_step = log_final_step or config_final_step or (max(observed_steps) if observed_steps else None)
        final_steps[mode] = final_step
        checkpoint_eval = load_mode_checkpoint_metrics(mode_dir, final_step)
        val_metrics = merge_val_metrics(log_eval, checkpoint_eval)

        mode_data[mode] = {
            "mode_dir": str(mode_dir),
            "final_step": final_step,
            "train_loss": train_loss,
            "eval_metrics": val_metrics,
            "checkpoint_metrics": checkpoint_eval,
            "rollout_accuracy": [],
        }

    rollout_metrics = load_rollout_metrics(analysis_dir, modes, final_steps)
    for mode in modes:
        mode_data[mode]["rollout_accuracy"] = rollout_metrics[mode]

    return {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "modes": modes,
        "mode_data": mode_data,
    }


def merge_val_metrics(
    log_eval: list[dict[str, Any]],
    checkpoint_eval: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for point in log_eval + checkpoint_eval:
        step = as_int(point.get("step"))
        if step is None:
            continue
        current = by_step.get(step, {"step": step})
        if point.get("val_loss") is not None:
            current["val_loss"] = point["val_loss"]
        if point.get("val_ppl") is not None:
            current["val_ppl"] = point["val_ppl"]
        current["source"] = point.get("source", current.get("source", ""))
        by_step[step] = current
    return sorted(by_step.values(), key=lambda item: item["step"])


def color_for_mode(mode: str, index: int) -> str:
    return COLORS.get(mode, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def render_svg(
    title: str,
    series: list[dict[str, Any]],
    y_label: str,
    width: int = 920,
    height: int = 420,
) -> str:
    margin_left = 72
    margin_right = 28
    margin_top = 48
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_points = [point for item in series for point in item["points"]]

    if not all_points:
        return empty_svg(title, width, height)

    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    for point in all_points:
        error = point[2] if len(point) >= 3 else None
        if error is not None:
            y_values.extend([point[1] - error, point[1] + error])
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        pad = abs(y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    lines = svg_header(width, height)
    lines.append(f'<text x="{margin_left}" y="28" class="title">{escape(title)}</text>')
    lines.extend(axis_elements(margin_left, margin_top, plot_width, plot_height, x_min, x_max, y_min, y_max, y_label))
    legend_x = margin_left + 10
    legend_y = margin_top + 14

    for idx, item in enumerate(series):
        color = color_for_mode(item["mode"], idx)
        points_attr = " ".join(f"{sx(point[0]):.2f},{sy(point[1]):.2f}" for point in item["points"])
        if len(item["points"]) >= 2:
            lines.append(f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for point in item["points"]:
            x = point[0]
            y = point[1]
            error = point[2] if len(point) >= 3 else None
            if error is not None and error > 0:
                y_low = sy(y - error)
                y_high = sy(y + error)
                x_pos = sx(x)
                lines.append(
                    f'<line x1="{x_pos:.2f}" y1="{y_high:.2f}" x2="{x_pos:.2f}" y2="{y_low:.2f}" '
                    f'stroke="{color}" stroke-width="1.4" class="errorbar"/>'
                )
                lines.append(
                    f'<line x1="{x_pos - 4:.2f}" y1="{y_high:.2f}" x2="{x_pos + 4:.2f}" y2="{y_high:.2f}" '
                    f'stroke="{color}" stroke-width="1.4" class="errorcap"/>'
                )
                lines.append(
                    f'<line x1="{x_pos - 4:.2f}" y1="{y_low:.2f}" x2="{x_pos + 4:.2f}" y2="{y_low:.2f}" '
                    f'stroke="{color}" stroke-width="1.4" class="errorcap"/>'
                )
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.2" fill="{color}"/>')
        lines.append(f'<line x1="{legend_x}" y1="{legend_y + idx * 22}" x2="{legend_x + 24}" y2="{legend_y + idx * 22}" stroke="{color}" stroke-width="2.4"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{legend_y + idx * 22 + 4}" class="legend">{escape(item["mode"])}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_val_loss_ppl_svg(curve_data: dict[str, Any]) -> str:
    modes = curve_data["modes"]
    mode_data = curve_data["mode_data"]
    width = 920
    panel_height = 330
    height = panel_height * 2
    top_series = build_series(modes, mode_data, "eval_metrics", "val_loss")
    bottom_series = build_series(modes, mode_data, "eval_metrics", "val_ppl")
    top_svg = render_panel("Validation loss", top_series, "val_loss", 0, width, panel_height)
    bottom_svg = render_panel("Validation perplexity", bottom_series, "val_ppl", panel_height, width, panel_height)
    lines = svg_header(width, height)
    lines.append('<text x="72" y="28" class="title">Validation Loss and Perplexity</text>')
    lines.extend(top_svg)
    lines.extend(bottom_svg)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_panel(
    title: str,
    series: list[dict[str, Any]],
    y_label: str,
    y_offset: int,
    width: int,
    height: int,
) -> list[str]:
    margin_left = 72
    margin_right = 28
    margin_top = y_offset + 70
    margin_bottom = 46
    plot_width = width - margin_left - margin_right
    plot_height = height - 120
    all_points = [point for item in series for point in item["points"]]
    if not all_points:
        return [f'<text x="{margin_left}" y="{margin_top + 40}" class="empty">{escape(title)}: no data</text>']
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        pad = abs(y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [f'<text x="{margin_left}" y="{margin_top - 16}" class="subtitle">{escape(title)}</text>']
    lines.extend(axis_elements(margin_left, margin_top, plot_width, plot_height, x_min, x_max, y_min, y_max, y_label))
    for idx, item in enumerate(series):
        color = color_for_mode(item["mode"], idx)
        points_attr = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in item["points"])
        if len(item["points"]) >= 2:
            lines.append(f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in item["points"]:
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
    return lines


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".title{font-size:20px;font-weight:700}",
        ".subtitle{font-size:15px;font-weight:700}",
        ".label{font-size:12px;fill:#4b5563}",
        ".tick{font-size:11px;fill:#6b7280}",
        ".legend{font-size:12px;fill:#374151}",
        ".empty{font-size:14px;fill:#6b7280}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".axis{stroke:#374151;stroke-width:1.2}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]


def axis_elements(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_label: str,
) -> list[str]:
    lines: list[str] = []
    bottom = top + plot_height
    right = left + plot_width
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    for idx in range(5):
        frac = idx / 4
        x = left + frac * plot_width
        value = x_min + frac * (x_max - x_min)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" class="tick">{format_tick(value)}</text>')
        y = bottom - frac * plot_height
        y_value = y_min + frac * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{format_tick(y_value)}</text>')
    lines.append(f'<text x="{left + plot_width / 2:.2f}" y="{bottom + 44}" text-anchor="middle" class="label">step</text>')
    lines.append(
        f'<text x="18" y="{top + plot_height / 2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 18 {top + plot_height / 2:.2f})">{escape(y_label)}</text>'
    )
    return lines


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def empty_svg(title: str, width: int, height: int) -> str:
    lines = svg_header(width, height)
    lines.append(f'<text x="72" y="28" class="title">{escape(title)}</text>')
    lines.append('<text x="72" y="96" class="empty">No data found.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def build_series(
    modes: list[str],
    mode_data: dict[str, Any],
    collection_name: str,
    value_name: str,
    error_name: str | None = None,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for mode in modes:
        points = []
        for point in mode_data[mode][collection_name]:
            value = as_float(point.get(value_name))
            step = as_float(point.get("step"))
            if step is not None and value is not None:
                error = as_float(point.get(error_name)) if error_name else None
                if error is not None:
                    points.append((step, value, error))
                else:
                    points.append((step, value))
        series.append({"mode": mode, "points": points})
    return series


def write_outputs(curve_data: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = curve_data["modes"]
    mode_data = curve_data["mode_data"]
    outputs = [output_dir / "curve_data.json"]
    write_json(outputs[0], curve_data)

    train_svg = output_dir / "train_loss_curve.svg"
    write_text(
        train_svg,
        render_svg("Training Loss", build_series(modes, mode_data, "train_loss", "train_loss"), "train_loss"),
    )
    outputs.append(train_svg)

    val_svg = output_dir / "val_loss_ppl_curve.svg"
    write_text(val_svg, render_val_loss_ppl_svg(curve_data))
    outputs.append(val_svg)

    rollout_svg = output_dir / "rollout_accuracy_curve.svg"
    write_text(
        rollout_svg,
        render_svg(
            "Rollout Accuracy",
            build_series(
                modes,
                mode_data,
                "rollout_accuracy",
                "rollout_acc",
                error_name="rollout_acc_std",
            ),
            "rollout_acc",
        ),
    )
    outputs.append(rollout_svg)
    return outputs


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    modes = list(args.modes)

    curve_data = collect_curve_data(run_dir=run_dir, analysis_dir=analysis_dir, modes=modes)
    outputs = write_outputs(curve_data=curve_data, output_dir=output_dir)
    for path in outputs:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
