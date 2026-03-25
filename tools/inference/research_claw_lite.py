#!/usr/bin/env python3
"""Run a small ARC-inspired experiment batch on top of the local inference harness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plot_sweep_progress import is_better, render_png


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
RUN_BENCH = ROOT / "run_grammar_bench.py"
DEFAULT_GRAMMAR_BENCH = ROOT / "benchmarks" / "grammar_fixer_working.jsonl"
DEFAULT_RLM_BENCH = ROOT / "benchmarks" / "rlm_editor_working.jsonl"
MODEL_PROFILES = {
    "m5max_local": ["qwen3:30b", "gpt-oss:120b", "qwen3.5:122b"],
    "m5max_local_extended": [
        "qwen3.5:122b",
        "gpt-oss:120b",
        "llama4:latest",
        "mixtral:8x22b",
        "llama3.1:70b",
        "qwen3:30b",
    ],
    "m5max_local_focus": [
        "qwen3.5:122b",
        "mixtral:8x22b",
        "llama3.1:70b",
        "qwen3:30b",
    ],
    "m5max_local_risky": ["deepseek-v2:236b"],
    "m5max_mlx_practical": [
        "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        "mlx-community/gpt-oss-120b-MXFP4-Q8",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    ],
    "m5max_mlx_quality": [
        "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "mlx-community/gpt-oss-120b-MXFP4-Q8",
    ],
    "m5max_mlx_speed": [
        "mlx-community/gpt-oss-120b-MXFP4-Q8",
        "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    ],
}


@dataclass
class Candidate:
    name: str
    hypothesis: str
    controller: str
    memory_top_k: int = 2
    temperature: float = 0.0
    num_ctx: int = 2048
    max_tokens: int = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--profile", choices=["grammar", "rlm"], default="rlm")
    parser.add_argument("--backend", choices=["mock", "ollama", "mlx"], default="ollama")
    parser.add_argument("--model", default="mistral:latest")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional list of model names. Comma-separated entries are also accepted.",
    )
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES),
        default="",
        help="Optional built-in multi-model shortlist.",
    )
    parser.add_argument("--examples-limit", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_research_claw")
    parser.add_argument("--session-name", default="")
    parser.add_argument("--objective", choices=["balanced", "quality_first"], default="quality_first")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--candidate-set",
        choices=["screen", "quick", "expanded"],
        default="quick",
        help="screen runs 2 candidates; quick runs 3; expanded runs 5.",
    )
    return parser.parse_args()


def default_benchmark(profile: str) -> Path:
    if profile == "grammar":
        return DEFAULT_GRAMMAR_BENCH
    return DEFAULT_RLM_BENCH


def resolve_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.model_profile:
        models.extend(MODEL_PROFILES[args.model_profile])
    for item in args.models:
        models.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not models:
        return [args.model]

    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def progress_row(result: dict[str, Any], index: int) -> dict[str, str]:
    row = {
        "run": f"{index:03d}_{model_slug(result['model'])}_{result['candidate']}",
        "model": str(result["model"]),
        "candidate": str(result["candidate"]),
        "controller": str(result["controller"]),
        "memory_top_k": str(result["memory_top_k"]),
        "temperature": str(result["temperature"]),
        "num_ctx": str(result["num_ctx"]),
        "max_tokens": str(result["max_tokens"]),
        "score": f"{float(result.get('score', -1e18)):.6f}",
        "description": f"{result['model']} {result['candidate']}",
        "status": "error",
        "avg_strict_quality_score": "",
        "avg_preserve_score": "",
        "pass_rate": "",
        "p95_latency_ms": "",
        "avg_subcalls": "",
    }
    summary = result.get("summary")
    if not summary:
        return row
    row["avg_strict_quality_score"] = f"{float(summary.get('avg_strict_quality_score', 0.0)):.6f}"
    row["avg_preserve_score"] = f"{float(summary.get('avg_preserve_score', 0.0)):.6f}"
    row["pass_rate"] = f"{float(summary.get('pass_rate', 0.0)):.6f}"
    row["p95_latency_ms"] = f"{float(summary.get('p95_latency_ms', 0.0)):.1f}"
    row["avg_subcalls"] = f"{float(summary.get('avg_subcalls', 0.0)):.3f}"
    return row


def write_results_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    header = [
        "run",
        "status",
        "model",
        "candidate",
        "controller",
        "memory_top_k",
        "temperature",
        "num_ctx",
        "max_tokens",
        "avg_strict_quality_score",
        "avg_preserve_score",
        "pass_rate",
        "p95_latency_ms",
        "avg_subcalls",
        "score",
        "description",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in header) + "\n")


def label_progress_rows(rows: list[dict[str, str]], *, metric_key: str, direction: str) -> None:
    best_so_far: tuple[float, float] | None = None
    for row in rows:
        metric = row.get(metric_key, "")
        latency = row.get("p95_latency_ms", "")
        if not metric or not latency:
            row["status"] = "error"
            continue
        candidate = (float(metric), float(latency))
        if is_better(candidate, best_so_far, direction):
            row["status"] = "keep"
            best_so_far = candidate
        else:
            row["status"] = "discard"


def refresh_progress_artifacts(session_dir: Path, rows: list[dict[str, str]], *, title: str) -> None:
    if not rows:
        return
    label_progress_rows(rows, metric_key="score", direction="higher")
    results_path = session_dir / "results.tsv"
    write_results_tsv(results_path, rows)
    plottable = [row for row in rows if row.get("status") in {"keep", "discard"}]
    if not plottable:
        return
    frontier_rows = [dict(row) for row in rows if row.get("status") == "keep"]
    progress_rows = frontier_rows or rows
    render_png(
        progress_rows,
        metric="score",
        direction="higher",
        label_key="description",
        title=f"{title} (frontier objective score)",
        output=session_dir / "progress.png",
    )
    render_png(
        progress_rows,
        metric="score",
        direction="higher",
        label_key="description",
        title=f"{title} (frontier objective score)",
        output=REPO_ROOT / "progress.png",
    )
    render_png(
        rows,
        metric="score",
        direction="higher",
        label_key="description",
        title=f"{title} (all objective scores)",
        output=session_dir / "all_progress.png",
    )
    quality_rows = [dict(row) for row in rows]
    label_progress_rows(quality_rows, metric_key="avg_strict_quality_score", direction="higher")
    render_png(
        quality_rows,
        metric="avg_strict_quality_score",
        direction="higher",
        label_key="description",
        title=f"{title} (strict quality)",
        output=session_dir / "quality_progress.png",
    )


def candidate_pool(profile: str, candidate_set: str) -> list[Candidate]:
    if profile == "grammar":
        base = [
            Candidate(
                name="direct_baseline",
                hypothesis="Plain direct prompting gives the latency floor and a useful baseline.",
                controller="direct",
                memory_top_k=0,
            ),
            Candidate(
                name="memory_flat_k2",
                hypothesis="Surfacing a small amount of memory should help preserve required terms with modest latency cost.",
                controller="memory_flat",
                memory_top_k=2,
            ),
            Candidate(
                name="rlm_lite_k3",
                hypothesis="A lightweight two-stage controller may improve quality on harder edits without full recursive overhead.",
                controller="rlm_lite",
                memory_top_k=3,
            ),
        ]
        if candidate_set == "screen":
            return base
        if candidate_set == "expanded":
            base.extend(
                [
                    Candidate(
                        name="rlm_recursive_k3",
                        hypothesis="Recursive verification may improve difficult cases if the extra latency is acceptable.",
                        controller="rlm_recursive",
                        memory_top_k=3,
                    ),
                    Candidate(
                        name="memory_flat_k4_hot",
                        hypothesis="More memory and a touch of temperature might help varied grammar rewrites, but may also drift.",
                        controller="memory_flat",
                        memory_top_k=4,
                        temperature=0.2,
                    ),
                ]
            )
        return base

    base = [
        Candidate(
            name="memory_flat_k2",
            hypothesis="A flat memory prompt is the cheapest memory-aware baseline for recall-heavy edits.",
            controller="memory_flat",
            memory_top_k=2,
        ),
        Candidate(
            name="rlm_lite_k3",
            hypothesis="Two-stage planning should improve constraint handling over flat memory with moderate overhead.",
            controller="rlm_lite",
            memory_top_k=3,
        ),
        Candidate(
            name="rlm_recursive_k3",
            hypothesis="Recursive memory note extraction and verification should help on hard recall cases.",
            controller="rlm_recursive",
            memory_top_k=3,
        ),
    ]
    if candidate_set == "screen":
        return base[:2]
    if candidate_set == "expanded":
        base.extend(
            [
                Candidate(
                    name="rlm_recursive_k4",
                    hypothesis="Allowing one more memory slot may recover missing constraints on harder examples.",
                    controller="rlm_recursive",
                    memory_top_k=4,
                ),
                Candidate(
                    name="rlm_adaptive_k3",
                    hypothesis="Adaptive escalation may approach recursive quality while avoiding the worst-case latency on easy rows.",
                    controller="rlm_adaptive",
                    memory_top_k=3,
                ),
            ]
        )
    return base


def ensure_benchmark(path: Path, profile: str, python_bin: str) -> None:
    if path.exists():
        return
    command = [python_bin, str(ROOT / "prepare_grammar_bench.py"), "--profile", profile]
    subprocess.run(command, check=True, cwd=ROOT.parent)


def make_session_dir(root: Path, session_name: str) -> Path:
    name = session_name or time.strftime("%Y%m%d-%H%M%S")
    session_dir = root / name
    session_dir.mkdir(parents=True, exist_ok=False)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(session_dir.name)
    return session_dir


def score_run(summary: dict[str, Any], objective: str) -> float:
    quality = float(summary["avg_strict_quality_score"])
    preserve = float(summary["avg_preserve_score"])
    latency = float(summary["p95_latency_ms"])
    if objective == "quality_first":
        return quality * 1000.0 + preserve * 100.0 - latency / 10000.0
    return quality * 1000.0 + preserve * 80.0 - latency / 2500.0


def run_candidate(
    session_dir: Path,
    benchmark: Path,
    candidate: Candidate,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_root = session_dir / candidate.name
    command = [
        args.python_bin,
        str(RUN_BENCH),
        "--benchmark",
        str(benchmark),
        "--backend",
        args.backend,
        "--model",
        model,
        "--controller",
        candidate.controller,
        "--memory-top-k",
        str(candidate.memory_top_k),
        "--temperature",
        str(candidate.temperature),
        "--num-ctx",
        str(candidate.num_ctx),
        "--max-tokens",
        str(candidate.max_tokens),
        "--examples-limit",
        str(args.examples_limit),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--output-dir",
        str(run_root),
        "--run-name",
        "trial",
    ]
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT.parent)
    elapsed_s = time.time() - started
    summary_path = run_root / "trial" / "summary.json"
    payload: dict[str, Any] = {
        "candidate": candidate.name,
        "model": model,
        "hypothesis": candidate.hypothesis,
        "controller": candidate.controller,
        "memory_top_k": candidate.memory_top_k,
        "temperature": candidate.temperature,
        "num_ctx": candidate.num_ctx,
        "max_tokens": candidate.max_tokens,
        "elapsed_s": elapsed_s,
        "command": command,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "error",
    }
    if proc.returncode == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["summary"] = summary
        payload["score"] = score_run(summary, args.objective)
    else:
        payload["summary"] = None
        payload["score"] = -1e18
    return payload


def config_from_result(result: dict[str, Any], backend: str, model: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "model": model,
        "controller": result["controller"],
        "memory_top_k": int(result["memory_top_k"]),
        "temperature": float(result["temperature"]),
        "num_ctx": int(result["num_ctx"]),
        "max_tokens": int(result["max_tokens"]),
    }


def select_promotions(results: list[dict[str, Any]], *, profile: str, backend: str, model: str) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("summary")]
    if not ok_rows:
        return {"quick": None, "deep": None}

    deep = max(ok_rows, key=lambda row: row["score"])
    quick_pool = ok_rows
    if profile == "rlm":
        preferred = [
            row for row in ok_rows
            if str(row.get("controller")) in {"rlm_adaptive", "rlm_lite", "memory_flat", "direct"}
        ]
        if preferred:
            quick_pool = preferred
    quick = min(
        quick_pool,
        key=lambda row: (
            float(row["summary"].get("p95_latency_ms", 1e18)),
            -float(row["summary"].get("avg_strict_quality_score", 0.0)),
        ),
    )

    def pack(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate": row["candidate"],
            "config": config_from_result(row, backend, model),
            "summary": row["summary"],
            "score": row["score"],
        }

    return {"quick": pack(quick), "deep": pack(deep)}


def write_model_report(
    session_dir: Path,
    benchmark: Path,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    model: str,
) -> dict[str, Any]:
    ranked = sorted(results, key=lambda row: row["score"], reverse=True)
    promotions = select_promotions(ranked, profile=args.profile, backend=args.backend, model=model)
    report = {
        "profile": args.profile,
        "benchmark": str(benchmark),
        "backend": args.backend,
        "model": model,
        "examples_limit": args.examples_limit,
        "objective": args.objective,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "promotions": promotions,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (session_dir / "promotions.json").write_text(json.dumps(promotions, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    lines = []
    lines.append(f"# research_claw_lite: {session_dir.name}")
    lines.append("")
    lines.append(f"- profile: {args.profile}")
    lines.append(f"- benchmark: `{benchmark}`")
    lines.append(f"- backend/model: `{args.backend}` / `{model}`")
    lines.append(f"- examples_limit: {args.examples_limit}")
    lines.append(f"- objective: {args.objective}")
    lines.append("")
    if promotions.get("quick"):
        lines.append("## Promotions")
        lines.append("")
        lines.append(f"- quick: {promotions['quick']['candidate']}")
        lines.append(f"- deep: {promotions['deep']['candidate']}")
        lines.append("")
    lines.append("## Ranking")
    lines.append("")
    for index, row in enumerate(ranked, start=1):
        lines.append(f"### {index}. {row['candidate']} [{row['status']}]")
        lines.append(f"- hypothesis: {row['hypothesis']}")
        if row["summary"]:
            summary = row["summary"]
            lines.append(f"- strict_quality: {summary['avg_strict_quality_score']:.4f}")
            lines.append(f"- drift: {summary.get('avg_drift_score', 1.0):.4f}")
            lines.append(f"- preserve: {summary['avg_preserve_score']:.4f}")
            lines.append(f"- pass_rate: {summary['pass_rate']:.4f}")
            lines.append(f"- p95_latency_ms: {summary['p95_latency_ms']:.1f}")
            lines.append(f"- avg_subcalls: {summary['avg_subcalls']:.2f}")
            lines.append(f"- score: {row['score']:.3f}")
        else:
            lines.append(f"- error: returncode={row['returncode']}")
        lines.append("")
    (session_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def select_global_promotions(model_reports: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    quick_pool = []
    for report in model_reports:
        promotions = report.get("promotions", {})
        for key in ("quick", "deep"):
            row = promotions.get(key)
            if row:
                quick_pool.append(row)
    deep_pool = [
        report["promotions"]["deep"]
        for report in model_reports
        if report.get("promotions", {}).get("deep")
    ]
    if not quick_pool and not deep_pool:
        return {"quick": None, "deep": None}

    if profile == "rlm":
        preferred = [
            row for row in quick_pool
            if str(row["config"].get("controller")) in {"rlm_adaptive", "rlm_lite", "memory_flat", "direct"}
        ]
        if preferred:
            quick_pool = preferred
    if quick_pool:
        best_quality = max(float(row["summary"].get("avg_strict_quality_score", 0.0)) for row in quick_pool)
        threshold = max(0.85, best_quality - 0.05)
        acceptable = [
            row for row in quick_pool
            if float(row["summary"].get("pass_rate", 0.0)) >= 0.5
            or float(row["summary"].get("avg_strict_quality_score", 0.0)) >= threshold
        ]
        if acceptable:
            quick_pool = acceptable

    quick = min(
        quick_pool,
        key=lambda row: (
            float(row["summary"].get("p95_latency_ms", 1e18)),
            -float(row["summary"].get("avg_strict_quality_score", 0.0)),
        ),
    ) if quick_pool else None
    deep = max(
        deep_pool,
        key=lambda row: (
            float(row.get("score", -1e18)),
            float(row["summary"].get("avg_strict_quality_score", 0.0)),
            -float(row["summary"].get("p95_latency_ms", 1e18)),
        ),
    ) if deep_pool else None
    return {"quick": quick, "deep": deep}


def write_multi_model_report(
    session_dir: Path,
    benchmark: Path,
    model_reports: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    leaderboard = []
    for report in model_reports:
        best = report.get("best")
        quick = report.get("promotions", {}).get("quick")
        deep = report.get("promotions", {}).get("deep")
        leaderboard.append(
            {
                "model": report["model"],
                "session_dir": report["session_dir"],
                "best_candidate": None if not best else best["candidate"],
                "best_score": None if not best else best["score"],
                "best_summary": None if not best else best["summary"],
                "quick": quick,
                "deep": deep,
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -float((row["best_summary"] or {}).get("avg_strict_quality_score", -1e18)),
            float((row["best_summary"] or {}).get("p95_latency_ms", 1e18)),
        ),
    )

    promotions = select_global_promotions(model_reports, profile=args.profile)
    report = {
        "profile": args.profile,
        "benchmark": str(benchmark),
        "backend": args.backend,
        "models": [report["model"] for report in model_reports],
        "examples_limit": args.examples_limit,
        "objective": args.objective,
        "leaderboard": leaderboard,
        "promotions": promotions,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (session_dir / "promotions.json").write_text(json.dumps(promotions, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    lines = []
    lines.append(f"# research_claw_lite: {session_dir.name}")
    lines.append("")
    lines.append(f"- profile: {args.profile}")
    lines.append(f"- benchmark: `{benchmark}`")
    lines.append(f"- backend: `{args.backend}`")
    lines.append(f"- models: `{', '.join(report['models'])}`")
    lines.append(f"- examples_limit: {args.examples_limit}")
    lines.append(f"- objective: {args.objective}")
    lines.append("")
    if promotions.get("quick") or promotions.get("deep"):
        lines.append("## Global Promotions")
        lines.append("")
        if promotions.get("quick"):
            lines.append(
                f"- quick: `{promotions['quick']['config']['model']}` / {promotions['quick']['candidate']}"
            )
        if promotions.get("deep"):
            lines.append(
                f"- deep: `{promotions['deep']['config']['model']}` / {promotions['deep']['candidate']}"
            )
        lines.append("")
    lines.append("## Model Ranking")
    lines.append("")
    for index, row in enumerate(leaderboard, start=1):
        lines.append(f"### {index}. {row['model']}")
        lines.append(f"- session_dir: `{row['session_dir']}`")
        lines.append(f"- best_candidate: {row['best_candidate']}")
        if row["best_summary"]:
            lines.append(f"- strict_quality: {row['best_summary']['avg_strict_quality_score']:.4f}")
            lines.append(f"- preserve: {row['best_summary']['avg_preserve_score']:.4f}")
            lines.append(f"- pass_rate: {row['best_summary']['pass_rate']:.4f}")
            lines.append(f"- p95_latency_ms: {row['best_summary']['p95_latency_ms']:.1f}")
        if row["quick"]:
            lines.append(
                f"- promoted_quick: {row['quick']['candidate']} "
                f"({row['quick']['summary']['p95_latency_ms']:.1f} ms)"
            )
        if row["deep"]:
            lines.append(
                f"- promoted_deep: {row['deep']['candidate']} "
                f"({row['deep']['summary']['avg_strict_quality_score']:.4f})"
            )
        lines.append("")
    (session_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark = (args.benchmark or default_benchmark(args.profile)).resolve()
    ensure_benchmark(benchmark, args.profile, args.python_bin)
    session_dir = make_session_dir(args.output_dir.resolve(), args.session_name)

    models = resolve_models(args)
    if len(models) == 1:
        model = models[0]
        results = []
        progress_rows: list[dict[str, str]] = []
        for candidate in candidate_pool(args.profile, args.candidate_set):
            result = run_candidate(session_dir, benchmark, candidate, model, args)
            result_path = session_dir / f"{candidate.name}.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
            print(f"[{result['status']}] {model} / {candidate.name}")
            if result["summary"]:
                summary = result["summary"]
                print(
                    f"  strict_quality={summary['avg_strict_quality_score']:.4f} "
                    f"preserve={summary['avg_preserve_score']:.4f} "
                    f"p95_latency_ms={summary['p95_latency_ms']:.1f}"
                )
            else:
                print(f"  returncode={result['returncode']}")
            results.append(result)
            progress_rows.append(progress_row(result, len(progress_rows) + 1))
            refresh_progress_artifacts(
                session_dir,
                progress_rows,
                title=f"Research Claw Progress: {session_dir.name}",
            )
        write_model_report(session_dir, benchmark, results, args, model=model)
    else:
        model_reports = []
        aggregate_rows: list[dict[str, str]] = []
        for model in models:
            model_dir = session_dir / model_slug(model)
            model_dir.mkdir(parents=True, exist_ok=False)
            print(f"== model: {model} ==")
            results = []
            model_progress_rows: list[dict[str, str]] = []
            for candidate in candidate_pool(args.profile, args.candidate_set):
                result = run_candidate(model_dir, benchmark, candidate, model, args)
                result_path = model_dir / f"{candidate.name}.json"
                result_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
                print(f"[{result['status']}] {model} / {candidate.name}")
                if result["summary"]:
                    summary = result["summary"]
                    print(
                        f"  strict_quality={summary['avg_strict_quality_score']:.4f} "
                        f"preserve={summary['avg_preserve_score']:.4f} "
                        f"p95_latency_ms={summary['p95_latency_ms']:.1f}"
                    )
                else:
                    print(f"  returncode={result['returncode']}")
                results.append(result)
                model_progress_rows.append(progress_row(result, len(model_progress_rows) + 1))
                refresh_progress_artifacts(
                    model_dir,
                    model_progress_rows,
                    title=f"Research Claw Progress: {model}",
                )
                aggregate_rows.append(progress_row(result, len(aggregate_rows) + 1))
                refresh_progress_artifacts(
                    session_dir,
                    aggregate_rows,
                    title=f"Research Claw Progress: {session_dir.name}",
                )
            report = write_model_report(model_dir, benchmark, results, args, model=model)
            report["session_dir"] = str(model_dir)
            model_reports.append(report)
        write_multi_model_report(session_dir, benchmark, model_reports, args)

    print(f"session_dir: {session_dir}")
    print(f"report_json: {session_dir / 'report.json'}")
    print(f"report_md: {session_dir / 'report.md'}")


if __name__ == "__main__":
    main()
