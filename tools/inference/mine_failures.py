#!/usr/bin/env python3
"""Summarize benchmark misses/drift from a research_claw_lite session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = ROOT / "runs_research_claw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--session-dir", type=Path, default=None, help="Defaults to run-root/latest")
    return parser.parse_args()


def resolve_session(run_root: Path, session_dir: Path | None) -> Path:
    if session_dir is not None:
        return session_dir.resolve()
    latest = run_root / "latest"
    if latest.is_symlink():
        return (run_root / latest.readlink()).resolve()
    return latest.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    session_dir = resolve_session(args.run_root.resolve(), args.session_dir)
    report = load_json(session_dir / "report.json")

    lines: list[str] = []
    lines.append(f"# failure mining: {session_dir.name}")
    lines.append("")
    best = report.get("best") or {}
    lines.append(f"best_candidate: {best.get('candidate')}")
    lines.append("")

    for ranked in report.get("ranked", []):
        candidate = ranked["candidate"]
        examples_path = session_dir / candidate / "trial" / "examples.jsonl"
        if not examples_path.exists():
            continue
        rows = load_jsonl(examples_path)
        lines.append(f"## {candidate}")
        failures = [row for row in rows if float(row.get("pass_score", 0.0)) < 1.0]
        drifts = [
            row for row in rows
            if float(row.get("pass_score", 0.0)) >= 1.0 and float(row.get("reference_score", 0.0)) < 0.95
        ]
        lines.append(f"- failures: {len(failures)}")
        lines.append(f"- drift_cases: {len(drifts)}")
        if failures:
            lines.append("- failed_examples:")
            for row in failures:
                lines.append(f"  - {row['id']}: output={json.dumps(row['output'])}")
        if drifts:
            lines.append("- drift_examples:")
            for row in drifts:
                lines.append(
                    f"  - {row['id']}: ref_score={float(row['reference_score']):.4f} output={json.dumps(row['output'])}"
                )
        lines.append("")

    out = "\n".join(lines) + "\n"
    output_path = session_dir / "failure_summary.md"
    output_path.write_text(out, encoding="utf-8")
    print(out, end="")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
