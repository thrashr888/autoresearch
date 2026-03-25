#!/usr/bin/env python3
"""Run a sequence of MLX experiments and keep mutating the current best config."""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BASE_CONFIG = {
    "ASPECT_RATIO": 64,
    "HEAD_DIM": 128,
    "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 2**16,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "ADAM_BETAS": "0.8,0.95",
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    "DEPTH": 4,
    "DEVICE_BATCH_SIZE": 16,
    "FINAL_EVAL_BATCH_SIZE": 256,
    "STARTUP_EXCLUDE_STEPS": 1,
}

DEFAULT_SEARCH_SPACE = {
    "DEPTH": [2, 3, 4, 5],
    "ASPECT_RATIO": [48, 56, 64, 72],
    "HEAD_DIM": [64, 128],
    "TOTAL_BATCH_SIZE": [2**14, 2**15, 2**16],
    "DEVICE_BATCH_SIZE": [8, 16, 32],
    "MATRIX_LR": [0.03, 0.04, 0.05, 0.06],
    "WEIGHT_DECAY": [0.1, 0.15, 0.2, 0.25, 0.3],
    "WARMUP_RATIO": [0.0, 0.02, 0.03],
    "WARMDOWN_RATIO": [0.25, 0.5, 0.75],
    "FINAL_LR_FRAC": [0.0, 0.05, 0.1],
    "EMBEDDING_LR": [0.4, 0.6, 0.8],
    "UNEMBEDDING_LR": [0.003, 0.004, 0.006],
    "SCALAR_LR": [0.4, 0.5, 0.6],
    "ADAM_BETAS": ["0.8,0.95", "0.85,0.97"],
    "WINDOW_PATTERN": ["SSSL", "SLSL"],
}

DEFAULT_SEED_EXPERIMENTS = [
    ("baseline", {}),
    ("device_batch_32", {"DEVICE_BATCH_SIZE": 32}),
    ("smaller_batch_2pow15", {"TOTAL_BATCH_SIZE": 2**15, "DEVICE_BATCH_SIZE": 16}),
    (
        "smaller_batch_tail",
        {
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
    (
        "smaller_batch_warmup",
        {
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "WARMUP_RATIO": 0.02,
            "WARMDOWN_RATIO": 0.25,
            "FINAL_LR_FRAC": 0.15,
        },
    ),
    (
        "depth3_tail",
        {
            "DEPTH": 3,
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
    (
        "depth2_fast",
        {
            "DEPTH": 2,
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "MATRIX_LR": 0.05,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
    (
        "head64_width48",
        {
            "ASPECT_RATIO": 48,
            "HEAD_DIM": 64,
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
    (
        "aggressive_opt",
        {
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "EMBEDDING_LR": 0.75,
            "UNEMBEDDING_LR": 0.006,
            "MATRIX_LR": 0.05,
            "SCALAR_LR": 0.6,
            "ADAM_BETAS": "0.85,0.97",
            "WEIGHT_DECAY": 0.12,
            "WARMUP_RATIO": 0.02,
            "WARMDOWN_RATIO": 0.25,
            "FINAL_LR_FRAC": 0.12,
        },
    ),
    (
        "conservative_opt",
        {
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "EMBEDDING_LR": 0.45,
            "UNEMBEDDING_LR": 0.003,
            "MATRIX_LR": 0.03,
            "SCALAR_LR": 0.4,
            "WEIGHT_DECAY": 0.18,
            "WARMDOWN_RATIO": 0.35,
            "FINAL_LR_FRAC": 0.08,
        },
    ),
    (
        "window_slsl",
        {
            "WINDOW_PATTERN": "SLSL",
            "TOTAL_BATCH_SIZE": 2**15,
            "DEVICE_BATCH_SIZE": 16,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
]

SAFE_BASE_CONFIG = {
    "ASPECT_RATIO": 48,
    "HEAD_DIM": 64,
    "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 2**14,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.15,
    "ADAM_BETAS": "0.8,0.95",
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.05,
    "DEPTH": 2,
    "DEVICE_BATCH_SIZE": 4,
    "FINAL_EVAL_BATCH_SIZE": 32,
    "STARTUP_EXCLUDE_STEPS": 1,
}

SAFE_SEARCH_SPACE = {
    "DEPTH": [2, 3],
    "ASPECT_RATIO": [32, 48],
    "HEAD_DIM": [64],
    "TOTAL_BATCH_SIZE": [2**13, 2**14, 2**15],
    "DEVICE_BATCH_SIZE": [4, 8],
    "MATRIX_LR": [0.03, 0.04, 0.05],
    "WEIGHT_DECAY": [0.1, 0.15, 0.2],
    "WARMUP_RATIO": [0.0, 0.02],
    "WARMDOWN_RATIO": [0.25, 0.5],
    "FINAL_LR_FRAC": [0.05, 0.1],
    "EMBEDDING_LR": [0.4, 0.6],
    "UNEMBEDDING_LR": [0.003, 0.004, 0.006],
    "SCALAR_LR": [0.4, 0.5],
    "ADAM_BETAS": ["0.8,0.95", "0.85,0.97"],
    "WINDOW_PATTERN": ["SSSL", "SLSL"],
}

SAFE_SEED_EXPERIMENTS = [
    ("baseline", {}),
    ("batch_2pow13", {"TOTAL_BATCH_SIZE": 2**13, "DEVICE_BATCH_SIZE": 4}),
    ("device_batch_8", {"TOTAL_BATCH_SIZE": 2**14, "DEVICE_BATCH_SIZE": 8}),
    (
        "depth3_tail",
        {
            "DEPTH": 3,
            "TOTAL_BATCH_SIZE": 2**14,
            "DEVICE_BATCH_SIZE": 4,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
    (
        "width32",
        {
            "ASPECT_RATIO": 32,
            "TOTAL_BATCH_SIZE": 2**14,
            "DEVICE_BATCH_SIZE": 4,
        },
    ),
    (
        "higher_matrix_lr",
        {
            "TOTAL_BATCH_SIZE": 2**14,
            "DEVICE_BATCH_SIZE": 4,
            "MATRIX_LR": 0.05,
            "WARMDOWN_RATIO": 0.30,
            "FINAL_LR_FRAC": 0.10,
        },
    ),
]

PROFILES = {
    "default": {
        "base_config": DEFAULT_BASE_CONFIG,
        "search_space": DEFAULT_SEARCH_SPACE,
        "seed_experiments": DEFAULT_SEED_EXPERIMENTS,
    },
    "safe": {
        "base_config": SAFE_BASE_CONFIG,
        "search_space": SAFE_SEARCH_SPACE,
        "seed_experiments": SAFE_SEED_EXPERIMENTS,
    },
}

METRIC_RE = re.compile(r"^([a-z_]+):\s+(.+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run an overnight MLX experiment sweep.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="default", help="Search profile.")
    parser.add_argument("--max-runs", type=int, default=24, help="Maximum number of experiments to run.")
    parser.add_argument("--run-root", default="runs", help="Directory for sweep logs and summaries.")
    parser.add_argument("--time-budget-seconds", type=int, default=300, help="Per-run training budget.")
    parser.add_argument("--heartbeat-secs", type=float, default=30.0, help="How often train.py emits heartbeat lines.")
    parser.add_argument(
        "--quick-eval-interval-secs",
        type=float,
        default=90.0,
        help="How often to run a small validation pass during training. Set 0 to disable.",
    )
    parser.add_argument(
        "--quick-eval-tokens",
        type=int,
        default=65536,
        help="Validation token budget for quick eval heartbeats.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate generation.")
    return parser.parse_args()


def make_session_dir(root):
    root_path = Path(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    session_dir = root_path / stamp
    session_dir.mkdir(parents=True, exist_ok=False)
    latest = root_path / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(session_dir.name)
    return session_dir


def config_key(config):
    items = sorted(config.items())
    return tuple((key, str(value)) for key, value in items)


def is_valid_config(config):
    tokens_per_fwdbwd = int(config["DEVICE_BATCH_SIZE"]) * 2048
    total_batch_size = int(config["TOTAL_BATCH_SIZE"])
    if total_batch_size % tokens_per_fwdbwd != 0:
        return False
    head_dim = int(config["HEAD_DIM"])
    model_dim = ((int(config["DEPTH"]) * int(config["ASPECT_RATIO"]) + head_dim - 1) // head_dim) * head_dim
    return model_dim >= head_dim


def mutate_config(best_config, rng, search_space):
    keys = list(search_space)
    while True:
        candidate = dict(best_config)
        rng.shuffle(keys)
        num_mutations = 1 if rng.random() < 0.55 else 2 if rng.random() < 0.85 else 3
        for key in keys[:num_mutations]:
            values = [value for value in search_space[key] if value != candidate[key]]
            candidate[key] = rng.choice(values)
        if is_valid_config(candidate):
            return candidate


def format_overrides(base, candidate):
    overrides = {key: value for key, value in candidate.items() if base[key] != value}
    if not overrides:
        return "baseline"
    parts = []
    for key, value in sorted(overrides.items()):
        parts.append(f"{key.lower()}={value}")
    return ",".join(parts)


def slugify(label):
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip()).strip("-").lower()
    return slug or "run"


def parse_metrics(log_path):
    metrics = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = METRIC_RE.match(line.strip())
            if not match:
                continue
            key, raw_value = match.groups()
            raw_value = raw_value.strip()
            try:
                value = float(raw_value)
                if value.is_integer():
                    value = int(value)
                metrics[key] = value
            except ValueError:
                metrics[key] = raw_value
    return metrics


def append_result(results_path, row):
    write_header = not results_path.exists()
    with results_path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(
                "\t".join(
                    [
                        "run",
                        "status",
                        "val_bpb",
                        "training_seconds",
                        "total_seconds",
                        "peak_vram_mb",
                        "num_steps",
                        "description",
                    ]
                )
                + "\n"
            )
        handle.write(
            "\t".join(
                [
                    str(row["run"]),
                    row["status"],
                    str(row.get("val_bpb", "")),
                    str(row.get("training_seconds", "")),
                    str(row.get("total_seconds", "")),
                    str(row.get("peak_vram_mb", "")),
                    str(row.get("num_steps", "")),
                    row["description"],
                ]
            )
            + "\n"
        )


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    profile = PROFILES[args.profile]
    session_dir = make_session_dir(args.run_root)
    results_path = session_dir / "results.tsv"
    summary_path = session_dir / "best.json"
    status_path = session_dir / "status.json"

    rng = random.Random(args.seed)
    best_config = dict(profile["base_config"])
    search_space = profile["search_space"]
    seed_experiments = profile["seed_experiments"]
    best_metrics = None
    seen = set()

    print(f"Session directory: {session_dir}")
    print(f"Results file: {results_path}")
    print(f"Status file: {status_path}")
    print(f"Profile: {args.profile}")
    print(f"Quick eval: every {args.quick_eval_interval_secs}s over {args.quick_eval_tokens} tokens")
    print()

    for run_index in range(1, args.max_runs + 1):
        if run_index <= len(seed_experiments):
            run_label, overrides = seed_experiments[run_index - 1]
            candidate = dict(best_config)
            candidate.update(overrides)
            if not is_valid_config(candidate):
                raise ValueError(f"Invalid seeded config for {run_label}: {candidate}")
        else:
            for _ in range(50):
                candidate = mutate_config(best_config, rng, search_space)
                if config_key(candidate) not in seen:
                    break
            else:
                print("Search space exhausted.")
                break
            run_label = format_overrides(best_config, candidate)

        seen.add(config_key(candidate))

        run_name = f"{run_index:03d}_{slugify(run_label)}"
        run_dir = session_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "config.json", candidate)

        env = os.environ.copy()
        env.update({key: str(value) for key, value in candidate.items()})
        env.update(
            {
                "RUN_NAME": run_name,
                "RUN_DIR": str(run_dir),
                "TIME_BUDGET_SECONDS": str(args.time_budget_seconds),
                "HEARTBEAT_SECS": str(args.heartbeat_secs),
                "QUICK_EVAL_INTERVAL_SECS": str(args.quick_eval_interval_secs),
                "QUICK_EVAL_TOKENS": str(args.quick_eval_tokens),
                "QUICK_EVAL_BATCH_SIZE": "128",
            }
        )

        latest_log = session_dir / "latest.log"
        if latest_log.exists() or latest_log.is_symlink():
            latest_log.unlink()
        latest_log.symlink_to(f"{run_name}/train.log")

        status_payload = {
            "state": "running",
            "run": run_index,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "description": run_label,
            "config": candidate,
        }
        write_json(status_path, status_payload)

        log_path = run_dir / "train.log"
        cmd = [sys.executable, "train.py"]
        started = time.time()
        print(f"[run {run_index:03d}] start | {run_label}")
        print(f"[run {run_index:03d}] log   | {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"# run_name: {run_name}\n")
            log_file.write(f"# description: {run_label}\n")
            log_file.flush()
            proc = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        wall_seconds = time.time() - started

        metrics = parse_metrics(log_path)
        metrics["wall_seconds"] = round(wall_seconds, 1)
        metrics["exit_code"] = proc.returncode
        write_json(run_dir / "metrics.json", metrics)

        if proc.returncode != 0:
            status = "failed"
            print(f"[run {run_index:03d}] fail  | exit_code={proc.returncode}")
        else:
            val_bpb = metrics.get("val_bpb")
            improved = best_metrics is None or (isinstance(val_bpb, (int, float)) and val_bpb < best_metrics["val_bpb"])
            status = "keep" if improved else "discard"
            if improved:
                best_config = dict(candidate)
                best_metrics = {
                    "val_bpb": val_bpb,
                    "run": run_index,
                    "run_name": run_name,
                    "description": run_label,
                    "config": candidate,
                    "metrics": metrics,
                }
                write_json(summary_path, best_metrics)
            print(
                f"[run {run_index:03d}] done  | status={status} | "
                f"val_bpb={metrics.get('val_bpb')} | total_seconds={metrics.get('total_seconds')}"
            )

        append_result(
            results_path,
            {
                "run": run_name,
                "status": status,
                "val_bpb": metrics.get("val_bpb", ""),
                "training_seconds": metrics.get("training_seconds", ""),
                "total_seconds": metrics.get("total_seconds", ""),
                "peak_vram_mb": metrics.get("peak_vram_mb", ""),
                "num_steps": metrics.get("num_steps", ""),
                "description": run_label,
            },
        )

        status_payload = {
            "state": "idle",
            "last_run": run_name,
            "last_status": status,
            "best": best_metrics,
        }
        write_json(status_path, status_payload)
        print()

    final_payload = {
        "state": "complete",
        "best": best_metrics,
        "session_dir": str(session_dir),
        "results_path": str(results_path),
    }
    write_json(status_path, final_payload)
    print("Sweep complete.")
    if best_metrics is not None:
        print(f"Best run: {best_metrics['run_name']} | val_bpb={best_metrics['val_bpb']}")
        print(f"Best config: {best_metrics['config']}")


if __name__ == "__main__":
    main()
