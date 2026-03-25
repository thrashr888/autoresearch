#!/usr/bin/env python3
"""Materialize a grammar/text-fixer benchmark from seed cases and optional custom examples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REQUIRED_FIELDS = {"id", "instruction", "input", "reference"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{lineno}: each JSONL row must be an object")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-bench",
        type=Path,
        action="append",
        default=[],
        help="Seed benchmark JSONL. Can be passed multiple times. Defaults depend on --profile.",
    )
    parser.add_argument(
        "--profile",
        choices=["grammar", "rlm", "rlm_strict", "combined"],
        default="grammar",
        help="Which benchmark profile to materialize when --seed-bench is not supplied.",
    )
    parser.add_argument(
        "--custom-bench",
        type=Path,
        action="append",
        default=[],
        help="Optional JSONL file(s) with the same schema to append.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the merged benchmark JSONL. Defaults depend on --profile.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of examples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle examples before writing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffling.")
    return parser.parse_args()


def default_seed_benches(root: Path, profile: str) -> list[Path]:
    grammar_seed = root / "benchmarks" / "grammar_fixer_seed.jsonl"
    rlm_seed = root / "benchmarks" / "rlm_editor_seed.jsonl"
    rlm_strict_seed = root / "benchmarks" / "rlm_editor_strict_seed.jsonl"
    if profile == "grammar":
        return [grammar_seed]
    if profile == "rlm":
        return [rlm_seed]
    if profile == "rlm_strict":
        return [rlm_strict_seed]
    return [grammar_seed, rlm_seed]


def default_output_path(root: Path, profile: str) -> Path:
    if profile == "grammar":
        return root / "benchmarks" / "grammar_fixer_working.jsonl"
    if profile == "rlm":
        return root / "benchmarks" / "rlm_editor_working.jsonl"
    if profile == "rlm_strict":
        return root / "benchmarks" / "rlm_editor_strict_working.jsonl"
    return root / "benchmarks" / "editing_combined_working.jsonl"


def validate_rows(rows: list[dict], *, source: str) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"{source} row {index}: missing required fields: {', '.join(missing)}")
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{source} row {index}: id must be a non-empty string")
        if row_id in seen_ids:
            raise ValueError(f"{source} row {index}: duplicate id: {row_id}")
        seen_ids.add(row_id)
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError(f"{source} row {index}: instruction must be a non-empty string")
        if not isinstance(row["input"], str):
            raise ValueError(f"{source} row {index}: input must be a string")
        if not isinstance(row["reference"], str):
            raise ValueError(f"{source} row {index}: reference must be a string")
        if "memory" in row and not isinstance(row["memory"], list):
            raise ValueError(f"{source} row {index}: memory must be a list of strings")
        if "preserve_terms" in row and not isinstance(row["preserve_terms"], list):
            raise ValueError(f"{source} row {index}: preserve_terms must be a list of strings")
        if "required_terms" in row and not isinstance(row["required_terms"], list):
            raise ValueError(f"{source} row {index}: required_terms must be a list of strings")
        if "forbidden_terms" in row and not isinstance(row["forbidden_terms"], list):
            raise ValueError(f"{source} row {index}: forbidden_terms must be a list of strings")
        if "checks" in row and not isinstance(row["checks"], dict):
            raise ValueError(f"{source} row {index}: checks must be an object")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"{source} row {index}: metadata must be an object")
        for field_name in ("memory", "preserve_terms", "required_terms", "forbidden_terms"):
            if field_name in row and any(not isinstance(item, str) for item in row[field_name]):
                raise ValueError(f"{source} row {index}: {field_name} must contain only strings")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    seed_benches = args.seed_bench or default_seed_benches(root, args.profile)
    output_path = args.output or default_output_path(root, args.profile)

    rows: list[dict] = []
    for seed_path in seed_benches:
        rows.extend(load_jsonl(seed_path))
    for custom_path in args.custom_bench:
        rows.extend(load_jsonl(custom_path))

    validate_rows(rows, source=str(output_path))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
