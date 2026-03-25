#!/usr/bin/env python3
"""Render a small SVG progress chart for an inference sweep session."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root / "runs_rlm_quality")
    parser.add_argument("--session-dir", type=Path, default=None, help="Explicit session dir; defaults to run-root/latest.")
    parser.add_argument("--metric", default="avg_quality_score")
    parser.add_argument("--direction", choices=["higher", "lower"], default="higher")
    parser.add_argument("--label-key", default="description")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def resolve_session(run_root: Path, session_dir: Path | None) -> Path:
    if session_dir is not None:
        return session_dir.resolve()
    latest = run_root / "latest"
    if latest.is_symlink():
        return (run_root / latest.readlink()).resolve()
    return latest.resolve()


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def collect_completed(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    completed = []
    for idx, row in enumerate(rows, start=1):
        status = row.get("status", "")
        if status not in {"keep", "discard"}:
            continue
        value = as_float(row, metric)
        if value is None:
            continue
        point = dict(row)
        point["_x"] = run_index(row.get("run", ""), idx)
        point["_y"] = value
        point["_p95"] = as_float(row, "p95_latency_ms") or math.inf
        completed.append(point)
    completed.sort(key=lambda row: (row["_x"], row.get("run", "")))
    return completed


def frontier_points(completed: list[dict[str, str]], direction: str) -> list[dict[str, str]]:
    best_so_far: tuple[float, float] | None = None
    keep_points: list[dict[str, str]] = []
    for row in completed:
        candidate = (row["_y"], row["_p95"])
        if is_better(candidate, best_so_far, direction):
            best_so_far = candidate
            keep_points.append(row)
    return keep_points


def run_index(name: str, fallback: int) -> int:
    match = re.match(r"(\d+)", name or "")
    if match is None:
        return fallback
    return int(match.group(1))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:
        return None
    return float(value)


def is_better(candidate: tuple[float, float], incumbent: tuple[float, float] | None, direction: str) -> bool:
    if incumbent is None:
        return True
    cand_metric, cand_tiebreak = candidate
    best_metric, best_tiebreak = incumbent
    if direction == "higher":
        if cand_metric > best_metric + 1e-12:
            return True
        if abs(cand_metric - best_metric) <= 1e-12 and cand_tiebreak < best_tiebreak:
            return True
        return False
    if cand_metric < best_metric - 1e-12:
        return True
    if abs(cand_metric - best_metric) <= 1e-12 and cand_tiebreak < best_tiebreak:
        return True
    return False


def clean_label(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value.replace("_", " ")


def render_svg(rows: list[dict[str, str]], *, metric: str, direction: str, label_key: str, title: str) -> str:
    width = 1500
    height = 760
    left = 88
    right = 34
    top = 56
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom

    completed = collect_completed(rows, metric)
    if not completed:
        raise SystemExit("No completed rows found to plot.")

    xs = [row["_x"] for row in completed]
    ys = [row["_y"] for row in completed]
    y_min = min(ys)
    y_max = max(ys)
    y_pad = max((y_max - y_min) * 0.12, max(abs(y_min), abs(y_max), 1.0) * 0.005)
    y0 = y_min - y_pad
    y1 = y_max + y_pad

    def px_x(value: int) -> float:
        if max(xs) == min(xs):
            return left + plot_w / 2
        return left + ((value - min(xs)) / (max(xs) - min(xs))) * plot_w

    def px_y(value: float) -> float:
        if abs(y1 - y0) < 1e-12:
            return top + plot_h / 2
        return top + (1.0 - ((value - y0) / (y1 - y0))) * plot_h

    keep_points = frontier_points(completed, direction)
    keep_lookup = {row.get("run", ""): row for row in keep_points}
    best_so_far: tuple[float, float] | None = None
    best_line: list[tuple[int, float]] = []
    for row in completed:
        candidate = (row["_y"], row["_p95"])
        if is_better(candidate, best_so_far, direction):
            best_so_far = candidate
        if best_so_far is not None:
            best_line.append((row["_x"], best_so_far[0]))

    grid = []
    for i in range(6):
        yv = y0 + ((y1 - y0) * i / 5)
        py = px_y(yv)
        label = f"{yv:.6f}" if abs(y1 - y0) < 0.1 else f"{yv:.3f}"
        grid.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>'
            f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" fill="#c8ced8" font-size="16">{html.escape(label)}</text>'
        )

    x_ticks = []
    for x_val in xs:
        px = px_x(x_val)
        x_ticks.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>')
    x_tick_labels = []
    step = max(1, len(xs) // 10)
    for x_val in xs[::step]:
        px = px_x(x_val)
        x_tick_labels.append(
            f'<text x="{px:.1f}" y="{height-bottom+28}" text-anchor="middle" fill="#c8ced8" font-size="16">{x_val}</text>'
        )

    path_parts = []
    prev_y = None
    for x_val, y_val in best_line:
        px = px_x(x_val)
        py = px_y(y_val)
        if not path_parts:
            path_parts.append(f"M {px:.1f} {py:.1f}")
        else:
            path_parts.append(f"L {px:.1f} {prev_y:.1f}")
            path_parts.append(f"L {px:.1f} {py:.1f}")
        prev_y = py
    best_path = " ".join(path_parts)

    discard_marks = []
    keep_marks = []
    labels = []
    for row in completed:
        px = px_x(row["_x"])
        py = px_y(row["_y"])
        is_keep = row.get("run", "") in keep_lookup
        if is_keep:
            keep_marks.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="#39d98a" stroke="#123521" stroke-width="1.5"/>'
            )
            label = clean_label(row.get(label_key, ""))
            if label:
                labels.append(
                    f'<text x="{px+8:.1f}" y="{py-10:.1f}" fill="#68d9a0" font-size="15" '
                    f'transform="rotate(-28 {px+8:.1f} {py-10:.1f})">{html.escape(label)}</text>'
                )
        else:
            discard_marks.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="#c7cbd1" fill-opacity="0.65"/>'
            )

    kept_count = len(keep_points)
    title_text = title or f"Sweep Progress: {len(completed)} Experiments, {kept_count} Kept Improvements"
    y_axis = f"{metric} ({'higher' if direction == 'higher' else 'lower'} is better)"

    legend_x = width - 196
    legend_y = 72
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#0b1016"/>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>
  {''.join(grid)}
  {''.join(x_ticks)}
  <path d="{best_path}" fill="none" stroke="#5fe0a1" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(discard_marks)}
  {''.join(keep_marks)}
  {''.join(labels)}
  <text x="{width/2:.1f}" y="32" text-anchor="middle" fill="#eef2f6" font-size="26" font-family="system-ui, sans-serif">{html.escape(title_text)}</text>
  <text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" fill="#c8ced8" font-size="20" font-family="system-ui, sans-serif">Experiment #</text>
  <g transform="translate(24 {height/2:.1f}) rotate(-90)">
    <text text-anchor="middle" fill="#c8ced8" font-size="20" font-family="system-ui, sans-serif">{html.escape(y_axis)}</text>
  </g>
  {''.join(x_tick_labels)}
  <g transform="translate({legend_x} {legend_y})">
    <rect x="0" y="0" width="170" height="96" rx="12" fill="rgba(11,16,22,0.88)" stroke="rgba(255,255,255,0.12)"/>
    <circle cx="18" cy="24" r="3.4" fill="#c7cbd1" fill-opacity="0.65"/>
    <text x="40" y="29" fill="#dbe1e8" font-size="16" font-family="system-ui, sans-serif">Discarded</text>
    <circle cx="18" cy="50" r="5.5" fill="#39d98a" stroke="#123521" stroke-width="1.5"/>
    <text x="40" y="55" fill="#dbe1e8" font-size="16" font-family="system-ui, sans-serif">Kept</text>
    <line x1="8" y1="76" x2="28" y2="76" stroke="#5fe0a1" stroke-width="3"/>
    <text x="40" y="81" fill="#dbe1e8" font-size="16" font-family="system-ui, sans-serif">Running best</text>
  </g>
</svg>
"""
    return svg


def render_png(rows: list[dict[str, str]], *, metric: str, direction: str, label_key: str, title: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = collect_completed(rows, metric)
    if not completed:
        raise SystemExit("No completed rows found to plot.")

    keep_points = frontier_points(completed, direction)
    keep_lookup = {row.get("run", ""): row for row in keep_points}
    xs = [row["_x"] for row in completed]
    ys = [row["_y"] for row in completed]

    running_best = []
    best_so_far: tuple[float, float] | None = None
    for row in completed:
        candidate = (row["_y"], row["_p95"])
        if is_better(candidate, best_so_far, direction):
            best_so_far = candidate
        running_best.append(best_so_far[0])

    fig, ax = plt.subplots(figsize=(15, 7.6), dpi=100)
    fig.patch.set_facecolor("#0b1016")
    ax.set_facecolor("#0b1016")
    ax.grid(True, axis="y", color="#ffffff", alpha=0.10, linewidth=1)
    ax.grid(True, axis="x", color="#ffffff", alpha=0.05, linewidth=1)

    discard_x = [row["_x"] for row in completed if row.get("run", "") not in keep_lookup]
    discard_y = [row["_y"] for row in completed if row.get("run", "") not in keep_lookup]
    keep_x = [row["_x"] for row in keep_points]
    keep_y = [row["_y"] for row in keep_points]

    if discard_x:
        ax.scatter(discard_x, discard_y, s=20, color="#c7cbd1", alpha=0.65, label="Discarded", zorder=2)
    ax.scatter(keep_x, keep_y, s=48, color="#39d98a", edgecolors="#123521", linewidths=1.2, label="Kept", zorder=3)
    ax.step(xs, running_best, where="post", color="#5fe0a1", linewidth=2.2, label="Running best", zorder=1)

    for row in keep_points:
        label = clean_label(row.get(label_key, ""))
        if not label:
            continue
        ax.annotate(
            label,
            xy=(row["_x"], row["_y"]),
            xytext=(6, 6),
            textcoords="offset points",
            rotation=28,
            color="#68d9a0",
            fontsize=10,
        )

    kept_count = len(keep_points)
    ax.set_title(title or f"Sweep Progress: {len(completed)} Experiments, {kept_count} Kept Improvements", color="#eef2f6", fontsize=20)
    ax.set_xlabel("Experiment #", color="#c8ced8", fontsize=14)
    ax.set_ylabel(f"{metric} ({'higher' if direction == 'higher' else 'lower'} is better)", color="#c8ced8", fontsize=14)
    ax.tick_params(colors="#c8ced8", labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_alpha(0.12)
    legend = ax.legend(facecolor="#0b1016", edgecolor="#ffffff", framealpha=0.12)
    for text in legend.get_texts():
        text.set_color("#dbe1e8")
    fig.tight_layout()
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    session_dir = resolve_session(args.run_root, args.session_dir)
    results = load_results(session_dir / "results.tsv")
    output = args.output or (session_dir / "progress.svg")
    title = args.title or f"RLM Sweep Progress: {session_dir.name}"
    if output.suffix.lower() == ".png":
        render_png(
            results,
            metric=args.metric,
            direction=args.direction,
            label_key=args.label_key,
            title=title,
            output=output,
        )
        print(output)
        return
    svg = render_svg(results, metric=args.metric, direction=args.direction, label_key=args.label_key, title=title)
    output.write_text(svg, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
