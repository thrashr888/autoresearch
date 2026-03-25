#!/usr/bin/env python3
"""Run a sequence of grammar/text-fixer benchmark experiments and keep the best config."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from plot_sweep_progress import load_results, render_png, render_svg


DEFAULT_PROFILE = {
    "backend": "ollama",
    "model": "mistral:latest",
    "controller": "direct",
    "memory_top_k": 2,
    "temperature": 0.0,
    "num_ctx": 2048,
    "max_tokens": 96,
}

SWEEP_PROFILES = {
    "default": {
        "search": {
            "controller": ["direct", "memory_flat", "rlm_adaptive", "rlm_lite", "rlm_recursive"],
            "memory_top_k": [1, 2, 3, 4],
            "temperature": [0.0, 0.1, 0.2],
            "num_ctx": [1024, 2048, 4096],
            "max_tokens": [64, 96, 128],
        },
        "seed_runs": [
            ("baseline", {}),
            ("memory_flat", {"controller": "memory_flat"}),
            ("rlm_adaptive", {"controller": "rlm_adaptive"}),
            ("rlm_lite", {"controller": "rlm_lite"}),
            ("rlm_recursive", {"controller": "rlm_recursive"}),
            ("ctx_1024", {"num_ctx": 1024}),
            ("shorter_output", {"max_tokens": 64}),
            ("rlm_lite_top3", {"controller": "rlm_lite", "memory_top_k": 3}),
            ("rlm_recursive_top4", {"controller": "rlm_recursive", "memory_top_k": 4}),
        ],
    },
    "grammar_focus": {
        "search": {
            "controller": ["direct", "memory_flat"],
            "memory_top_k": [1, 2, 3],
            "temperature": [0.0],
            "num_ctx": [1024, 2048, 4096],
            "max_tokens": [48, 64, 96],
        },
        "seed_runs": [
            ("baseline", {"controller": "direct"}),
            ("memory_flat", {"controller": "memory_flat"}),
            ("memory_flat_top3", {"controller": "memory_flat", "memory_top_k": 3}),
            ("ctx_1024", {"num_ctx": 1024}),
            ("shorter_output", {"max_tokens": 48}),
        ],
    },
    "rlm_focus": {
        "search": {
            "controller": ["memory_flat", "rlm_recursive"],
            "memory_top_k": [1, 2, 3],
            "temperature": [0.0],
            "num_ctx": [1024, 2048, 4096],
            "max_tokens": [64, 96],
        },
        "seed_runs": [
            ("memory_flat", {"controller": "memory_flat"}),
            ("rlm_recursive", {"controller": "rlm_recursive"}),
            ("rlm_recursive_top3", {"controller": "rlm_recursive", "memory_top_k": 3}),
            ("ctx_4096", {"num_ctx": 4096}),
            ("shorter_output", {"max_tokens": 64}),
        ],
    },
    "rlm_quality": {
        "search": {
            "controller": ["rlm_recursive"],
            "memory_top_k": [2, 3, 4],
            "temperature": [0.0],
            "num_ctx": [2048, 4096],
            "max_tokens": [64, 96, 128],
        },
        "seed_runs": [
            ("baseline_quality", {"controller": "rlm_recursive", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 2048}),
            ("shorter_output", {"controller": "rlm_recursive", "memory_top_k": 3, "max_tokens": 64, "num_ctx": 2048}),
            ("longer_output", {"controller": "rlm_recursive", "memory_top_k": 3, "max_tokens": 128, "num_ctx": 2048}),
            ("top2", {"controller": "rlm_recursive", "memory_top_k": 2, "max_tokens": 96, "num_ctx": 2048}),
            ("top4", {"controller": "rlm_recursive", "memory_top_k": 4, "max_tokens": 96, "num_ctx": 2048}),
            ("ctx_4096", {"controller": "rlm_recursive", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 4096}),
            ("top4_long", {"controller": "rlm_recursive", "memory_top_k": 4, "max_tokens": 128, "num_ctx": 2048}),
        ],
    },
    "rlm_qwen35_focus": {
        "search": {
            "controller": ["memory_flat", "rlm_lite", "rlm_recursive"],
            "memory_top_k": [2, 3, 4],
            "temperature": [0.0],
            "num_ctx": [1024, 2048, 4096],
            "max_tokens": [64, 96, 128],
        },
        "seed_runs": [
            ("memory_flat_k2", {"controller": "memory_flat", "memory_top_k": 2, "max_tokens": 96, "num_ctx": 2048}),
            ("rlm_lite_k3", {"controller": "rlm_lite", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 2048}),
            ("rlm_recursive_k3", {"controller": "rlm_recursive", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 2048}),
            ("rlm_lite_k2", {"controller": "rlm_lite", "memory_top_k": 2, "max_tokens": 96, "num_ctx": 2048}),
            ("rlm_lite_ctx1024", {"controller": "rlm_lite", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 1024}),
            ("rlm_lite_ctx4096", {"controller": "rlm_lite", "memory_top_k": 3, "max_tokens": 96, "num_ctx": 4096}),
            ("rlm_lite_short", {"controller": "rlm_lite", "memory_top_k": 3, "max_tokens": 64, "num_ctx": 2048}),
            ("rlm_lite_long", {"controller": "rlm_lite", "memory_top_k": 3, "max_tokens": 128, "num_ctx": 2048}),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=root / "benchmarks" / "grammar_fixer_working.jsonl",
        help="Benchmark JSONL file.",
    )
    parser.add_argument("--run-root", default=str(root / "runs"), help="Where sweep sessions are stored.")
    parser.add_argument("--max-runs", type=int, default=18)
    parser.add_argument("--sweep-profile", choices=sorted(SWEEP_PROFILES), default="default")
    parser.add_argument("--backend", choices=["ollama", "mlx", "mock"], default=DEFAULT_PROFILE["backend"])
    parser.add_argument("--model", default=DEFAULT_PROFILE["model"])
    parser.add_argument("--objective", choices=["balanced", "quality_first", "quality_max"], default="balanced")
    parser.add_argument("--score-key", choices=["avg_quality_score", "avg_strict_quality_score"], default="avg_quality_score")
    parser.add_argument("--quality-slack", type=float, default=0.02)
    parser.add_argument("--preserve-slack", type=float, default=0.05)
    parser.add_argument("--max-p95-latency-ms", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--backend-wait-seconds", type=float, default=30.0)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_session_dir(run_root: str) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / time.strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(session_dir.name)
    return session_dir


def config_key(config: dict[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value)) for key, value in sorted(config.items()))


def mutate_config(config: dict[str, object], search_space: dict[str, list[object]], rng: random.Random) -> dict[str, object]:
    mutable_keys = [
        key for key, values in search_space.items()
        if any(value != config.get(key) for value in values)
    ]
    if not mutable_keys:
        return dict(config)
    candidate = dict(config)
    rng.shuffle(mutable_keys)
    mutate_count = 1 if rng.random() < 0.60 or len(mutable_keys) == 1 else 2
    for key in mutable_keys[:mutate_count]:
        values = [value for value in search_space[key] if value != candidate.get(key)]
        if values:
            candidate[key] = rng.choice(values)
    return candidate


def ensure_backend_available(candidate: dict[str, object], args: argparse.Namespace) -> str | None:
    backend = str(candidate["backend"])
    if backend in {"mock", "mlx"}:
        return None
    if backend != "ollama":
        return f"Unsupported backend preflight: {backend}"

    probe_url = args.ollama_host.rstrip("/") + "/api/tags"
    deadline = time.time() + max(0.0, args.backend_wait_seconds)
    last_error: Exception | None = None
    while True:
        try:
            with urllib.request.urlopen(probe_url, timeout=5.0) as response:
                if response.status < 400:
                    return None
                last_error = RuntimeError(f"HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if time.time() >= deadline:
            break
        time.sleep(3.0)
    return f"Backend unavailable at {probe_url}: {last_error}"


def extract_failure_reason(log_path: Path, fallback: str) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return fallback
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:300]
    return fallback


def is_keep(summary: dict[str, object], baseline: dict[str, object], best: dict[str, object], args: argparse.Namespace) -> bool:
    score_key = args.score_key
    avg_quality = float(summary[score_key])
    avg_preserve = float(summary["avg_preserve_score"])
    if avg_quality < float(baseline[score_key]) - args.quality_slack:
        return False
    if avg_preserve < float(baseline["avg_preserve_score"]) - args.preserve_slack:
        return False

    best_quality = float(best[score_key])
    best_latency = float(best["p95_latency_ms"])
    latency = float(summary["p95_latency_ms"])
    if args.max_p95_latency_ms > 0 and latency > args.max_p95_latency_ms:
        return False

    if args.objective == "quality_max":
        if avg_quality > best_quality + 0.0005:
            return True
        if abs(avg_quality - best_quality) <= 0.0005 and avg_preserve > float(best["avg_preserve_score"]) + 0.01:
            return True
        if abs(avg_quality - best_quality) <= 0.0005 and avg_preserve >= float(best["avg_preserve_score"]) - 0.001:
            return latency < best_latency * 0.97
        return False

    if args.objective == "quality_first":
        if avg_quality > best_quality + 0.002:
            return True
        if avg_quality >= best_quality - 0.002 and avg_preserve > float(best["avg_preserve_score"]) + 0.01:
            return True
        if avg_quality >= best_quality - 0.002 and latency < best_latency * 0.97:
            return True
        return False

    if avg_quality > best_quality + 0.01:
        return True
    if avg_quality >= best_quality - 0.01 and latency < best_latency * 0.97:
        return True
    return False


def append_result(path: Path, row: dict[str, object]) -> None:
    default_header = [
        "run",
        "status",
        "avg_quality_score",
        "avg_strict_quality_score",
        "avg_preserve_score",
        "avg_required_score",
        "avg_forbidden_score",
        "avg_constraint_score",
        "avg_structure_score",
        "avg_drift_score",
        "pass_rate",
        "hard_avg_quality_score",
        "hard_avg_strict_quality_score",
        "hard_pass_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_ttft_ms",
        "avg_decode_tok_per_s",
        "avg_subcalls",
        "peak_memory_mb",
        "description",
        "error",
    ]
    write_header = not path.exists()
    header = default_header
    if not write_header:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if first_line:
            header = first_line.split("\t")
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("\t".join(header) + "\n")
        handle.write(
            "\t".join(str(row.get(column, "")) for column in header) + "\n"
        )


def refresh_progress_artifacts(results_path: Path, session_dir: Path, repo_root: Path, metric: str) -> None:
    try:
        rows = load_results(results_path)
        if not rows:
            return
        title = f"RLM Sweep Progress: {session_dir.name}"
        svg = render_svg(rows, metric=metric, direction="higher", label_key="description", title=title)
        (session_dir / "progress.svg").write_text(svg, encoding="utf-8")
        render_png(
            rows,
            metric=metric,
            direction="higher",
            label_key="description",
            title=title,
            output=session_dir / "progress.png",
        )
        render_png(
            rows,
            metric=metric,
            direction="higher",
            label_key="description",
            title=title,
            output=repo_root / "progress.png",
        )
    except Exception as exc:  # pragma: no cover - best-effort artifact update
        print(f"[progress] warning: {exc}", flush=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent
    os.chdir(repo_root)
    session_dir = make_session_dir(args.run_root)
    results_path = session_dir / "results.tsv"
    status_path = session_dir / "status.json"
    best_path = session_dir / "best.json"
    rng = random.Random(args.seed)

    best_config = dict(DEFAULT_PROFILE)
    best_config.update({"backend": args.backend, "model": args.model})
    baseline_summary: dict[str, object] | None = None
    best_summary: dict[str, object] | None = None
    seen = set()
    sweep_profile = SWEEP_PROFILES[args.sweep_profile]
    search_space = sweep_profile["search"]
    seed_runs = sweep_profile["seed_runs"]
    session_state = "complete"
    session_error = ""

    for run_index in range(1, args.max_runs + 1):
        if run_index <= len(seed_runs):
            run_label, overrides = seed_runs[run_index - 1]
            candidate = dict(best_config)
            candidate.update(overrides)
        else:
            if best_summary is None:
                session_state = "failed"
                session_error = "No successful seed runs; aborting search."
                status_payload = {
                    "state": session_state,
                    "error": session_error,
                    "session_dir": str(session_dir),
                    "results_path": str(results_path),
                }
                status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(session_error, flush=True)
                break
            candidate = mutate_config(best_config, search_space, rng)
            run_label = ",".join(
                f"{key}={value}"
                for key, value in sorted(candidate.items())
                if best_config.get(key) != value
            ) or "baseline"
        if config_key(candidate) in seen:
            continue
        seen.add(config_key(candidate))

        run_name = f"{run_index:03d}_{run_label.replace(',', '-').replace('=', '-')}"
        run_dir = session_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        latest_log = session_dir / "latest.log"
        if latest_log.is_symlink() or latest_log.exists():
            latest_log.unlink()
        latest_log.symlink_to(f"{run_name}/run.log")

        status_payload = {
            "state": "running",
            "run": run_index,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "description": run_label,
            "config": candidate,
        }
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        backend_error = ensure_backend_available(candidate, args)
        if backend_error is not None:
            status = "blocked"
            session_state = "blocked"
            session_error = backend_error
            failure_summary = {
                "backend": candidate["backend"],
                "model": candidate["model"],
                "controller": candidate["controller"],
                "wall_seconds": 0.0,
                "run_name": run_name,
                "description": run_label,
                "error": backend_error,
            }
            (run_dir / "summary.json").write_text(
                json.dumps(failure_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            append_result(
                results_path,
                {
                    "run": run_name,
                    "status": status,
                    "description": run_label,
                    "error": backend_error,
                },
            )
            refresh_progress_artifacts(results_path, session_dir, repo_root, args.score_key)
            status_payload = {
                "state": session_state,
                "last_run": run_name,
                "last_status": status,
                "error": session_error,
                "best": None if best_summary is None else {"config": best_config, "summary": best_summary},
            }
            status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[run {run_index:03d}] blocked | {run_label} | {backend_error}", flush=True)
            break

        log_path = run_dir / "run.log"
        summary_run_dir = run_dir / "artifacts"
        cmd = [
            sys.executable,
            str(root / "run_grammar_bench.py"),
            "--benchmark",
            str(args.benchmark),
            "--backend",
            str(candidate["backend"]),
            "--model",
            str(candidate["model"]),
            "--controller",
            str(candidate["controller"]),
            "--memory-top-k",
            str(candidate["memory_top_k"]),
            "--temperature",
            str(candidate["temperature"]),
            "--num-ctx",
            str(candidate["num_ctx"]),
            "--max-tokens",
            str(candidate["max_tokens"]),
            "--output-dir",
            str(summary_run_dir),
            "--run-name",
            "bench",
            "--ollama-host",
            args.ollama_host,
            "--request-timeout-s",
            str(args.request_timeout_s),
        ]
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# run_name: {run_name}\n")
            handle.write(f"# description: {run_label}\n")
            handle.flush()
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        wall_seconds = round(time.time() - started, 1)
        summary_path = summary_run_dir / "bench" / "summary.json"
        if proc.returncode != 0 or not summary_path.exists():
            status = "failed"
            error_message = extract_failure_reason(
                log_path,
                f"benchmark subprocess failed with exit code {proc.returncode}",
            )
            failure_summary = {
                "backend": candidate["backend"],
                "model": candidate["model"],
                "controller": candidate["controller"],
                "wall_seconds": wall_seconds,
                "run_name": run_name,
                "description": run_label,
                "error": error_message,
            }
            (run_dir / "summary.json").write_text(
                json.dumps(failure_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            append_result(
                results_path,
                {
                    "run": run_name,
                    "status": status,
                    "description": run_label,
                    "error": error_message,
                },
            )
            refresh_progress_artifacts(results_path, session_dir, repo_root, args.score_key)
            status_payload = {
                "state": "idle",
                "last_run": run_name,
                "last_status": status,
                "best": None if best_summary is None else {"config": best_config, "summary": best_summary},
            }
            status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[run {run_index:03d}] failed | {run_label} | {error_message}", flush=True)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["wall_seconds"] = wall_seconds
        summary["run_name"] = run_name
        summary["description"] = run_label
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if baseline_summary is None:
            baseline_summary = dict(summary)
            best_summary = dict(summary)
            best_config = dict(candidate)
            status = "keep"
            best_path.write_text(json.dumps({"config": best_config, "summary": best_summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif is_keep(summary, baseline_summary, best_summary, args):
            best_summary = dict(summary)
            best_config = dict(candidate)
            status = "keep"
            best_path.write_text(json.dumps({"config": best_config, "summary": best_summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            status = "discard"

        append_result(
            results_path,
            {
                "run": run_name,
                "status": status,
                "avg_quality_score": round(float(summary["avg_quality_score"]), 6),
                "avg_strict_quality_score": round(float(summary["avg_strict_quality_score"]), 6),
                "avg_preserve_score": round(float(summary["avg_preserve_score"]), 6),
                "avg_required_score": round(float(summary["avg_required_score"]), 6),
                "avg_forbidden_score": round(float(summary["avg_forbidden_score"]), 6),
                "avg_constraint_score": round(float(summary["avg_constraint_score"]), 6),
                "avg_structure_score": round(float(summary["avg_structure_score"]), 6),
                "pass_rate": round(float(summary["pass_rate"]), 6),
                "hard_avg_quality_score": (
                    "" if summary["hard_avg_quality_score"] is None else round(float(summary["hard_avg_quality_score"]), 6)
                ),
                "hard_avg_strict_quality_score": (
                    ""
                    if summary["hard_avg_strict_quality_score"] is None
                    else round(float(summary["hard_avg_strict_quality_score"]), 6)
                ),
                "hard_pass_rate": "" if summary["hard_pass_rate"] is None else round(float(summary["hard_pass_rate"]), 6),
                "avg_latency_ms": round(float(summary["avg_latency_ms"]), 1),
                "p95_latency_ms": round(float(summary["p95_latency_ms"]), 1),
                "avg_ttft_ms": "" if summary["avg_ttft_ms"] is None else round(float(summary["avg_ttft_ms"]), 1),
                "avg_decode_tok_per_s": "" if summary["avg_decode_tok_per_s"] is None else round(float(summary["avg_decode_tok_per_s"]), 1),
                "avg_subcalls": round(float(summary["avg_subcalls"]), 2),
                "peak_memory_mb": "" if summary["peak_memory_mb"] is None else round(float(summary["peak_memory_mb"]), 1),
                "description": run_label,
                "error": "",
            },
        )
        refresh_progress_artifacts(results_path, session_dir, repo_root, args.score_key)

        status_payload = {
            "state": "idle",
            "last_run": run_name,
            "last_status": status,
            "best": None if best_summary is None else {"config": best_config, "summary": best_summary},
        }
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[run {run_index:03d}] {status} | score={summary[args.score_key]:.4f} | "
            f"p95_ms={summary['p95_latency_ms']:.1f} | {run_label}",
            flush=True,
        )

    final_status = {
        "state": session_state,
        "error": session_error,
        "best": None if best_summary is None else {"config": best_config, "summary": best_summary},
        "session_dir": str(session_dir),
        "results_path": str(results_path),
    }
    status_path.write_text(json.dumps(final_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Sweep {session_state}.", flush=True)


if __name__ == "__main__":
    main()
