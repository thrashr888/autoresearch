#!/usr/bin/env python3
"""Summarize an RLM sweep session and highlight the current weakest examples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root / "runs_rlm_quality")
    parser.add_argument("--session-dir", type=Path, default=None, help="Explicit session dir; defaults to run-root/latest.")
    parser.add_argument("--top-runs", type=int, default=5)
    parser.add_argument("--top-worst", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def load_results(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return 0.0
    return float(value)


def resolve_session(run_root: Path, session_dir: Path | None) -> Path:
    if session_dir is not None:
        return session_dir.resolve()
    latest = run_root / "latest"
    if latest.is_symlink():
        return (run_root / latest.readlink()).resolve()
    return latest.resolve()


def examples_path_for_run(session_dir: Path, run_name: str) -> Path:
    return session_dir / run_name / "artifacts" / "bench" / "examples.jsonl"


def load_examples(session_dir: Path, run_name: str) -> list[dict[str, Any]]:
    path = examples_path_for_run(session_dir, run_name)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(session_dir: Path, top_runs: int, top_worst: int) -> dict[str, Any]:
    status = load_json(session_dir / "status.json") or {}
    best = load_json(session_dir / "best.json")
    results = load_results(session_dir / "results.tsv")

    completed = [row for row in results if row.get("status") in {"keep", "discard"}]
    failed = [row for row in results if row.get("status") not in {"keep", "discard"}]
    ranked_runs = sorted(
        completed,
        key=lambda row: (
            as_float(row, "avg_quality_score"),
            as_float(row, "avg_preserve_score"),
            as_float(row, "avg_structure_score"),
            -as_float(row, "p95_latency_ms"),
        ),
        reverse=True,
    )

    example_scores: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        run_name = row["run"]
        for example in load_examples(session_dir, run_name):
            example_scores.setdefault(example["id"], []).append(
                {
                    "run": run_name,
                    "description": row.get("description", ""),
                    "quality_score": float(example["quality_score"]),
                    "reference_score": float(example["reference_score"]),
                    "restraint_score": float(example["restraint_score"]),
                    "output": example["output"],
                }
            )

    weakest_examples: list[dict[str, Any]] = []
    spread_examples: list[dict[str, Any]] = []
    if best is not None:
        best_run_name = best["summary"]["run_name"]
        best_examples = load_examples(session_dir, best_run_name)
        best_examples.sort(key=lambda row: row["quality_score"])
        for row in best_examples[:top_worst]:
            weakest_examples.append(
                {
                    "id": row["id"],
                    "quality_score": round(float(row["quality_score"]), 6),
                    "reference_score": round(float(row["reference_score"]), 6),
                    "restraint_score": round(float(row["restraint_score"]), 6),
                    "output": row["output"],
                }
            )

    for example_id, scores in example_scores.items():
        ordered = sorted(scores, key=lambda item: item["quality_score"])
        if len(ordered) < 2:
            continue
        spread_examples.append(
            {
                "id": example_id,
                "min_quality": round(ordered[0]["quality_score"], 6),
                "max_quality": round(ordered[-1]["quality_score"], 6),
                "spread": round(ordered[-1]["quality_score"] - ordered[0]["quality_score"], 6),
                "best_run": ordered[-1]["run"],
                "worst_run": ordered[0]["run"],
            }
        )
    spread_examples.sort(key=lambda item: item["spread"], reverse=True)

    return {
        "session_dir": str(session_dir),
        "state": status.get("state", "unknown"),
        "current_run": status.get("run_name"),
        "completed_runs": len(completed),
        "failed_or_blocked_runs": len(failed),
        "best": best,
        "top_runs": ranked_runs[:top_runs],
        "weakest_examples_in_best_run": weakest_examples,
        "largest_example_spreads": spread_examples[:top_worst],
    }


def format_text(summary: dict[str, Any]) -> str:
    lines = [
        f"session: {summary['session_dir']}",
        f"state: {summary['state']}",
        f"current_run: {summary['current_run']}",
        f"completed_runs: {summary['completed_runs']}",
        f"failed_or_blocked_runs: {summary['failed_or_blocked_runs']}",
    ]
    best = summary.get("best")
    if best is not None:
        best_summary = best["summary"]
        lines.extend(
            [
                "",
                "best_run:",
                f"  {best_summary['run_name']} | quality={best_summary['avg_quality_score']:.6f} | "
                f"p95_ms={best_summary['p95_latency_ms']:.1f} | subcalls={best_summary['avg_subcalls']:.2f}",
            ]
        )
    if summary["top_runs"]:
        lines.extend(["", "top_runs:"])
        for row in summary["top_runs"]:
            lines.append(
                f"  {row['run']} | {row['status']} | q={as_float(row, 'avg_quality_score'):.6f} | "
                f"p95={as_float(row, 'p95_latency_ms'):.1f} | {row.get('description', '')}"
            )
    if summary["weakest_examples_in_best_run"]:
        lines.extend(["", "weakest_examples_in_best_run:"])
        for row in summary["weakest_examples_in_best_run"]:
            lines.append(
                f"  {row['id']} | q={row['quality_score']:.6f} | ref={row['reference_score']:.6f} | "
                f"restraint={row['restraint_score']:.6f}"
            )
            lines.append(f"    output: {row['output']}")
    if summary["largest_example_spreads"]:
        lines.extend(["", "largest_example_spreads:"])
        for row in summary["largest_example_spreads"]:
            lines.append(
                f"  {row['id']} | spread={row['spread']:.6f} | "
                f"{row['worst_run']} -> {row['best_run']}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    session_dir = resolve_session(args.run_root, args.session_dir)
    summary = summarize(session_dir, args.top_runs, args.top_worst)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(format_text(summary))


if __name__ == "__main__":
    main()
