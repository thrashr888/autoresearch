#!/usr/bin/env python3
"""Mine benchmark candidates from product logs, validate them, and promote useful cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_grammar_bench import Example, make_backend, run_controller, score_example, similarity
from studio_server import DEFAULT_RESEARCH_RUN_ROOT, load_promoted_configs


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
DEFAULT_LOG_DIR = ROOT / "studio" / "logs"
DEFAULT_RUN_ROOT = ROOT / "runs_case_curator"
DEFAULT_INBOX_PATH = ROOT / "benchmarks" / "private" / "curated_rlm_inbox.jsonl"
DEFAULT_PROMOTED_PATH = ROOT / "benchmarks" / "private" / "curated_rlm_promoted.jsonl"
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "rlm_editor_strict_working.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--research-run-root", type=Path, default=DEFAULT_RESEARCH_RUN_ROOT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--inbox-path", type=Path, default=DEFAULT_INBOX_PATH)
    parser.add_argument("--promoted-path", type=Path, default=DEFAULT_PROMOTED_PATH)
    parser.add_argument("--mode", choices=["any", "quick", "deep"], default="any")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on candidate rows after log ingestion.")
    parser.add_argument("--min-disagreement", type=float, default=0.03)
    parser.add_argument("--min-deep-gain", type=float, default=0.03)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-keep-alive", default="15m")
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--mlx-python-bin", default="python")
    return parser.parse_args()


def make_session_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    session_dir = run_root / time.strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    latest = run_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(session_dir.name)
    return session_dir


def split_lines_or_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def split_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_inline_code_terms(text: str) -> list[str]:
    return re.findall(r"`[^`]+`", text)


def infer_case_trap_forbidden_terms(preserve_terms: list[str]) -> list[str]:
    inferred: list[str] = []
    for term in preserve_terms:
        lowered = term.lower()
        if lowered != term:
            inferred.append(lowered)
    return unique_strings(inferred)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def case_key(case_row: dict[str, Any]) -> str:
    payload = {
        "instruction": case_row["instruction"],
        "input": case_row["input"],
        "reference": case_row["reference"],
        "memory": case_row["memory"],
        "preserve_terms": case_row["preserve_terms"],
        "required_terms": case_row["required_terms"],
        "forbidden_terms": case_row["forbidden_terms"],
        "checks": case_row["checks"],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_candidate_from_log(log_row: dict[str, Any], source_path: Path, *, mode_filter: str) -> dict[str, Any] | None:
    request = log_row.get("request") or {}
    response = log_row.get("response") or {}
    mode = str(request.get("mode") or response.get("mode") or "").strip().lower()
    if mode_filter != "any" and mode != mode_filter:
        return None

    text = str(request.get("text") or request.get("input") or "").strip()
    accepted_reference = str(request.get("accepted_output") or request.get("final_output") or "").strip()
    reference = accepted_reference or str(request.get("reference") or "").strip()
    if not text or not reference:
        return None
    if not accepted_reference and similarity(text, reference) < 0.15:
        return None

    instruction = str(request.get("instruction") or "Fix the text with minimal edits using the provided memory.").strip()
    memory = unique_strings(split_lines_or_list(request.get("memory")))
    preserve_terms = unique_strings(split_terms(request.get("preserve_terms")))
    inferred_code_terms = [
        term for term in extract_inline_code_terms(text) + extract_inline_code_terms(reference)
        if term not in preserve_terms
    ]
    preserve_terms = unique_strings([*preserve_terms, *inferred_code_terms])
    required_terms = unique_strings(split_terms(request.get("required_terms")) or preserve_terms)
    forbidden_terms = unique_strings(split_terms(request.get("forbidden_terms")) or infer_case_trap_forbidden_terms(preserve_terms))
    checks = request.get("checks", {}) or {}

    metadata = {
        "source": "studio_log",
        "category": str((request.get("metadata") or {}).get("category") or "product"),
        "difficulty": str((request.get("metadata") or {}).get("difficulty") or "unknown"),
        "log_ts": log_row.get("ts"),
        "mode": mode or "unknown",
        "source_path": str(source_path),
        "request_model": str((response.get("config") or {}).get("model") or ""),
    }
    digest = hashlib.sha256(
        json.dumps({"instruction": instruction, "input": text, "reference": reference}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": f"auto-{digest}",
        "instruction": instruction,
        "input": text,
        "reference": reference,
        "memory": memory,
        "preserve_terms": preserve_terms,
        "required_terms": required_terms,
        "forbidden_terms": forbidden_terms,
        "checks": checks,
        "metadata": metadata,
    }


def example_from_case(case_row: dict[str, Any]) -> Example:
    return Example(
        id=case_row["id"],
        instruction=case_row["instruction"],
        input=case_row["input"],
        reference=case_row["reference"],
        memory=case_row["memory"],
        preserve_terms=case_row["preserve_terms"],
        required_terms=case_row["required_terms"],
        forbidden_terms=case_row["forbidden_terms"],
        checks=case_row["checks"],
        metadata=case_row["metadata"],
    )


def runtime_args(config: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend=config["backend"],
        model=config["model"],
        controller=config["controller"],
        memory_top_k=int(config["memory_top_k"]),
        temperature=float(config["temperature"]),
        num_ctx=int(config["num_ctx"]),
        max_tokens=int(config["max_tokens"]),
        benchmark=args.benchmark,
        ollama_host=args.ollama_host,
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_think="auto",
        request_timeout_s=float(args.request_timeout_s),
        mlx_python_bin=args.mlx_python_bin,
    )


def evaluate_with_config(case_row: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    example = example_from_case(case_row)
    backend = make_backend(runtime_args(config, args))
    result, debug = run_controller(
        backend,
        example,
        controller=str(config["controller"]),
        max_tokens=int(config["max_tokens"]),
        temperature=float(config["temperature"]),
        num_ctx=int(config["num_ctx"]),
        memory_top_k=int(config["memory_top_k"]),
    )
    return {
        "config": config,
        "output": result.text,
        "metrics": {
            "latency_ms": result.latency_ms,
            "ttft_ms": result.ttft_ms,
            "decode_tok_per_s": result.decode_tok_per_s,
            "peak_memory_mb": result.peak_memory_mb,
            "subcalls": result.subcalls,
        },
        "scores": score_example(example, result.text),
        "debug": debug,
        "backend_details": result.backend_details,
    }


def decide_case_status(
    quick_eval: dict[str, Any],
    deep_eval: dict[str, Any],
    *,
    min_disagreement: float,
    min_deep_gain: float,
) -> str:
    quick_scores = quick_eval["scores"]
    deep_scores = deep_eval["scores"]
    disagreement = 1.0 - similarity(quick_eval["output"], deep_eval["output"])
    quick_pass = float(quick_scores["pass_score"]) >= 1.0
    deep_gain = float(deep_scores["strict_quality_score"]) - float(quick_scores["strict_quality_score"])

    if quick_pass and disagreement < min_disagreement and deep_gain < min_deep_gain:
        return "skip"
    if (not quick_pass) or deep_gain >= min_deep_gain or disagreement >= min_disagreement:
        return "promote"
    return "inbox"


def append_unique_rows(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = load_jsonl(path)
    existing_keys = {case_key(row): row for row in existing}
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        key = case_key(row)
        if key in existing_keys:
            continue
        existing_keys[key] = row
        existing.append(row)
        new_rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")
    return new_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")


def render_progress(processed_rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(1, len(processed_rows) + 1))
    cumulative: list[int] = []
    total = 0
    status_colors = {
        "promote": "#2fbf71",
        "inbox": "#ffb000",
        "skip": "#c8c8c8",
        "duplicate": "#9aa0a6",
    }
    ys: list[int] = []
    colors: list[str] = []
    for row in processed_rows:
        if row["status"] == "promote":
            total += 1
        cumulative.append(total)
        ys.append(total)
        colors.append(status_colors.get(row["status"], "#9aa0a6"))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, cumulative, color="#2fbf71", linewidth=2, label="Promoted total")
    ax.scatter(xs, ys, c=colors, edgecolors="#444444", linewidths=0.5, s=38, label="Cases")
    ax.set_title(title)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Cumulative promoted")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fafafa")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_results_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "candidate",
        "status",
        "quick_strict_quality",
        "deep_strict_quality",
        "deep_gain",
        "quick_pass_rate",
        "deep_pass_rate",
        "quick_p95_ms",
        "deep_p95_ms",
        "disagreement",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    str(row.get(column, ""))
                    for column in header
                ) + "\n"
            )


def main() -> None:
    args = parse_args()
    session_dir = make_session_dir(args.run_root.resolve())
    run_root = args.run_root.resolve()
    quick_config, deep_config, meta = load_promoted_configs(args.research_run_root.resolve())

    log_paths = sorted(args.log_dir.glob("*.jsonl"))
    if not log_paths:
        raise SystemExit(f"No log files found in {args.log_dir}")

    existing_keys = {
        case_key(row)
        for path in [args.inbox_path.resolve(), args.promoted_path.resolve()]
        for row in load_jsonl(path)
    }

    candidates: list[dict[str, Any]] = []
    for path in log_paths:
        for log_row in load_jsonl(path):
            candidate = build_candidate_from_log(log_row, path, mode_filter=args.mode)
            if candidate is None:
                continue
            candidates.append(candidate)

    deduped: list[dict[str, Any]] = []
    seen_this_run: set[str] = set()
    for candidate in candidates:
        key = case_key(candidate)
        if key in seen_this_run:
            continue
        seen_this_run.add(key)
        deduped.append(candidate)
    if args.limit > 0:
        deduped = deduped[: args.limit]

    processed_rows: list[dict[str, Any]] = []
    promoted_rows: list[dict[str, Any]] = []
    inbox_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(deduped, start=1):
        key = case_key(candidate)
        if key in existing_keys:
            processed_rows.append(
                {
                    "candidate": candidate["id"],
                    "status": "duplicate",
                    "quick_strict_quality": "",
                    "deep_strict_quality": "",
                    "deep_gain": "",
                    "quick_pass_rate": "",
                    "deep_pass_rate": "",
                    "quick_p95_ms": "",
                    "deep_p95_ms": "",
                    "disagreement": "",
                }
            )
            continue

        quick_eval = evaluate_with_config(candidate, quick_config, args)
        deep_eval = evaluate_with_config(candidate, deep_config, args)
        disagreement = 1.0 - similarity(quick_eval["output"], deep_eval["output"])
        deep_gain = float(deep_eval["scores"]["strict_quality_score"]) - float(quick_eval["scores"]["strict_quality_score"])
        status = decide_case_status(
            quick_eval,
            deep_eval,
            min_disagreement=args.min_disagreement,
            min_deep_gain=args.min_deep_gain,
        )

        enriched = dict(candidate)
        enriched["evaluation"] = {
            "quick": quick_eval,
            "deep": deep_eval,
            "disagreement": disagreement,
            "deep_gain": deep_gain,
            "status": status,
        }
        enriched_rows.append(enriched)
        processed_rows.append(
            {
                "candidate": candidate["id"],
                "status": status,
                "quick_strict_quality": round(float(quick_eval["scores"]["strict_quality_score"]), 6),
                "deep_strict_quality": round(float(deep_eval["scores"]["strict_quality_score"]), 6),
                "deep_gain": round(deep_gain, 6),
                "quick_pass_rate": float(quick_eval["scores"]["pass_score"]),
                "deep_pass_rate": float(deep_eval["scores"]["pass_score"]),
                "quick_p95_ms": round(float(quick_eval["metrics"]["latency_ms"]), 1),
                "deep_p95_ms": round(float(deep_eval["metrics"]["latency_ms"]), 1),
                "disagreement": round(disagreement, 6),
            }
        )
        if status == "promote":
            promoted_rows.append(candidate)
            existing_keys.add(key)
        elif status == "inbox":
            inbox_rows.append(candidate)
            existing_keys.add(key)

    new_promoted = append_unique_rows(args.promoted_path.resolve(), promoted_rows)
    new_inbox = append_unique_rows(args.inbox_path.resolve(), inbox_rows)

    write_jsonl(session_dir / "candidates.jsonl", enriched_rows)
    write_jsonl(session_dir / "promoted.jsonl", new_promoted)
    write_jsonl(session_dir / "inbox.jsonl", new_inbox)
    write_results_tsv(session_dir / "results.tsv", processed_rows)

    progress_session = session_dir / "progress.png"
    progress_root = run_root / "progress.png"
    progress_repo = REPO_ROOT / "progress.png"
    render_progress(processed_rows, progress_session, f"Case Curation Progress: {session_dir.name}")
    render_progress(processed_rows, progress_root, f"Case Curation Progress: {session_dir.name}")
    render_progress(processed_rows, progress_repo, f"Case Curation Progress: {session_dir.name}")

    summary = {
        "log_files": [str(path) for path in log_paths],
        "promotion_source": meta.get("source"),
        "quick_config": quick_config,
        "deep_config": deep_config,
        "processed": len(processed_rows),
        "promoted": len(new_promoted),
        "inbox": len(new_inbox),
        "duplicates": sum(1 for row in processed_rows if row["status"] == "duplicate"),
        "skipped": sum(1 for row in processed_rows if row["status"] == "skip"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    lines = []
    lines.append(f"# case curator: {session_dir.name}")
    lines.append("")
    lines.append(f"- log_files: {len(log_paths)}")
    lines.append(f"- processed: {summary['processed']}")
    lines.append(f"- promoted: {summary['promoted']}")
    lines.append(f"- inbox: {summary['inbox']}")
    lines.append(f"- duplicates: {summary['duplicates']}")
    lines.append(f"- skipped: {summary['skipped']}")
    lines.append(f"- quick_model: `{quick_config['model']}` / `{quick_config['controller']}`")
    lines.append(f"- deep_model: `{deep_config['model']}` / `{deep_config['controller']}`")
    lines.append(f"- progress_png: `{progress_session}`")
    lines.append("")
    lines.append("## Promotions")
    lines.append("")
    for row in new_promoted[:12]:
        evaluation = next(item["evaluation"] for item in enriched_rows if item["id"] == row["id"])
        lines.append(
            f"- {row['id']}: deep_gain={evaluation['deep_gain']:.4f}, "
            f"quick={evaluation['quick']['scores']['strict_quality_score']:.4f}, "
            f"deep={evaluation['deep']['scores']['strict_quality_score']:.4f}"
        )
    (session_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=False))
    print(f"report_md: {session_dir / 'report.md'}")
    print(f"progress_png: {progress_session}")


if __name__ == "__main__":
    main()
