#!/usr/bin/env python3
"""Serve a small local UI for manual RLM text-fixing experiments."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_grammar_bench import Example, load_examples, make_backend, run_controller, score_example


ROOT = Path(__file__).resolve().parent
STUDIO_DIR = ROOT / "studio"
LOG_DIR = STUDIO_DIR / "logs"
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "rlm_editor_working.jsonl"
DEFAULT_RUN_ROOT = ROOT / "runs_rlm_quality"
DEFAULT_RESEARCH_RUN_ROOT = ROOT / "runs_research_claw"
MODEL_MATRIX_PATH = ROOT / "mlx-bench" / "mlx_model_matrix.json"
DEFAULT_CONFIG = {
    "backend": "mlx",
    "model": "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
    "controller": "rlm_recursive",
    "memory_top_k": 3,
    "temperature": 0.0,
    "num_ctx": 2048,
    "max_tokens": 96,
}
PRODUCT_CONFIG = {
    "backend": "mlx",
    "model": "mlx-community/gpt-oss-120b-MXFP4-Q8",
    "controller": "memory_flat",
    "memory_top_k": 2,
    "temperature": 0.0,
    "num_ctx": 1536,
    "max_tokens": 72,
}


def load_model_matrix() -> dict[str, Any]:
    if not MODEL_MATRIX_PATH.exists():
        return {}
    return json.loads(MODEL_MATRIX_PATH.read_text(encoding="utf-8"))


def preset_configs_from_matrix(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for key, payload in (matrix or {}).items():
        config = dict((payload or {}).get("config") or {})
        if config:
            config["preset"] = key
            presets[key] = config
    return presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--research-run-root", type=Path, default=DEFAULT_RESEARCH_RUN_ROOT)
    parser.add_argument("--backend", choices=["mock", "ollama", "mlx"], default=DEFAULT_CONFIG["backend"])
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--controller", default=DEFAULT_CONFIG["controller"])
    parser.add_argument("--memory-top-k", type=int, default=DEFAULT_CONFIG["memory_top_k"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_CONFIG["num_ctx"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CONFIG["max_tokens"])
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-keep-alive", default="15m")
    parser.add_argument("--ollama-think", default="auto")
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--mlx-python-bin", default="/Users/thrashr888/.venvs/mlx-bench-qwen/bin/python")
    return parser.parse_args()


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


def load_best_config(run_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    latest = run_root / "latest" / "best.json"
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8"))
        return payload["config"], payload.get("summary")
    best_files = sorted(run_root.glob("*/best.json"))
    if best_files:
        payload = json.loads(best_files[-1].read_text(encoding="utf-8"))
        return payload["config"], payload.get("summary")
    return dict(DEFAULT_CONFIG), None


def load_promoted_configs(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    latest = run_root / "latest" / "promotions.json"
    payload = None
    source = None
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8"))
        source = latest
    else:
        promotion_files = sorted(run_root.glob("*/promotions.json"))
        if promotion_files:
            source = promotion_files[-1]
            payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload:
        return dict(PRODUCT_CONFIG), dict(DEFAULT_CONFIG), {"source": None}

    quick = dict(PRODUCT_CONFIG)
    quick.update((payload.get("quick") or {}).get("config", {}))
    deep = dict(DEFAULT_CONFIG)
    deep.update((payload.get("deep") or {}).get("config", {}))
    return quick, deep, {"source": str(source) if source else None, "promotions": payload}


def runtime_namespace(config: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return SimpleNamespace(
        backend=merged["backend"],
        model=merged["model"],
        controller=merged["controller"],
        memory_top_k=int(merged["memory_top_k"]),
        temperature=float(merged["temperature"]),
        num_ctx=int(merged["num_ctx"]),
        max_tokens=int(merged["max_tokens"]),
        benchmark=args.benchmark,
        ollama_host=args.ollama_host,
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_think=args.ollama_think,
        request_timeout_s=float(args.request_timeout_s),
        mlx_python_bin=args.mlx_python_bin,
    )


def build_product_config(nightly_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    product = dict(PRODUCT_CONFIG)
    product["backend"] = nightly_config.get("backend", args.backend)
    product["model"] = nightly_config.get("model", args.model)
    return product


def load_benchmark_samples(path: Path, limit: int = 12) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples = load_examples(path)[:limit]
    out: list[dict[str, Any]] = []
    for example in examples:
        out.append(
            {
                "id": example.id,
                "instruction": example.instruction,
                "input": example.input,
                "reference": example.reference,
                "memory": example.memory,
                "preserve_terms": example.preserve_terms,
                "required_terms": example.required_terms,
                "forbidden_terms": example.forbidden_terms,
                "checks": example.checks,
                "metadata": example.metadata,
            }
        )
    return out


def make_example(payload: dict[str, Any]) -> Example:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("`text` is required.")
    instruction = str(payload.get("instruction", "")).strip() or "Fix the text with minimal edits using the provided memory."
    reference = str(payload.get("reference", "")).strip() or text
    return Example(
        id=str(payload.get("id", f"manual-{uuid.uuid4().hex[:8]}")),
        instruction=instruction,
        input=text,
        reference=reference,
        memory=split_lines_or_list(payload.get("memory")),
        preserve_terms=split_terms(payload.get("preserve_terms")),
        required_terms=split_terms(payload.get("required_terms")),
        forbidden_terms=split_terms(payload.get("forbidden_terms")),
        checks=payload.get("checks", {}) or {},
        metadata=payload.get("metadata", {}) or {},
    )


def log_request(log_dir: Path, row: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{time.strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class StudioHandler(SimpleHTTPRequestHandler):
    server_version = "RLMStudio/0.3"

    def __init__(self, *args: Any, directory: str, app_state: dict[str, Any], **kwargs: Any) -> None:
        self.app_state = app_state
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/config":
            self._send_json(
                {
                    "default_config": self.app_state["product_config"],
                    "product_config": self.app_state["product_config"],
                    "nightly_config": self.app_state["nightly_config"],
                    "best_summary": self.app_state["best_summary"],
                    "presets": self.app_state["presets"],
                    "model_matrix": self.app_state["model_matrix"],
                    "benchmark": str(self.app_state["args"].benchmark),
                }
            )
            return
        if self.path == "/api/examples":
            self._send_json({"examples": self.app_state["examples"]})
            return
        if self.path == "/api/health":
            self._send_json({"ok": True})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/fix":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            example = make_example(payload)
            requested = dict(payload.get("config", {}) or {})
            mode = str(payload.get("mode") or requested.pop("mode", "quick")).strip().lower() or "quick"
            config = self.resolve_config(mode, example, requested)
            runtime_args = runtime_namespace(config, self.app_state["args"])
            backend = make_backend(runtime_args)
            result, debug = run_controller(
                backend,
                example,
                controller=str(config["controller"]),
                max_tokens=int(config["max_tokens"]),
                temperature=float(config["temperature"]),
                num_ctx=int(config["num_ctx"]),
                memory_top_k=int(config["memory_top_k"]),
            )
            scores = score_example(example, result.text) if payload.get("reference") else None
            response = {
                "ok": True,
                "mode": mode,
                "config": config,
                "example_id": example.id,
                "output": result.text,
                "metrics": {
                    "latency_ms": result.latency_ms,
                    "ttft_ms": result.ttft_ms,
                    "decode_tok_per_s": result.decode_tok_per_s,
                    "peak_memory_mb": result.peak_memory_mb,
                    "subcalls": result.subcalls,
                },
                "scores": scores,
                "debug": debug,
                "backend_details": result.backend_details,
            }
            log_request(
                LOG_DIR,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "request": payload,
                    "response": response,
                },
            )
            self._send_json(response)
        except Exception as exc:  # pragma: no cover - manual tool path
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def resolve_config(self, mode: str, example: Example, requested: dict[str, Any]) -> dict[str, Any]:
        nightly = dict(self.app_state["nightly_config"])
        product = dict(self.app_state["product_config"])
        preset_name = str(requested.pop("preset", "") or "").strip().lower()
        preset_configs = self.app_state.get("preset_configs") or {}
        selected_preset = dict(preset_configs.get(preset_name, {})) if preset_name else {}
        if requested.get("controller"):
            config = dict(product)
            config.update(selected_preset)
            config.update(requested)
            config["mode"] = "custom"
            return config

        if mode == "deep":
            config = dict(nightly)
            config.update(selected_preset)
            config["mode"] = "deep"
            return config

        config = dict(product)
        config.update(selected_preset)
        memory_count = len(example.memory)
        if memory_count == 0:
            config["controller"] = "direct"
            config["memory_top_k"] = 0
            config["max_tokens"] = 64
        else:
            config["controller"] = "memory_flat"
            config["memory_top_k"] = min(memory_count, int(config["memory_top_k"]))
            if len(example.input.split()) <= 16 and memory_count <= 1:
                config["max_tokens"] = 56
        config["mode"] = "quick"
        return config


def main() -> None:
    args = parse_args()
    promoted_product_config, promoted_nightly_config, promotion_meta = load_promoted_configs(args.research_run_root)
    nightly_config, best_summary = load_best_config(args.run_root)
    model_matrix = load_model_matrix()
    preset_configs = preset_configs_from_matrix(model_matrix)
    if promotion_meta.get("source"):
        product_config = build_product_config(promoted_nightly_config, args)
        product_config.update(promoted_product_config)
        nightly_config = dict(nightly_config)
        nightly_config.update(promoted_nightly_config)
    else:
        nightly_config.setdefault("backend", args.backend)
        nightly_config.setdefault("model", args.model)
        product_config = build_product_config(nightly_config, args)
    app_state = {
        "args": args,
        "nightly_config": nightly_config,
        "product_config": product_config,
        "best_summary": best_summary,
        "promotion_meta": promotion_meta,
        "model_matrix": model_matrix,
        "presets": sorted(preset_configs),
        "preset_configs": preset_configs,
        "examples": load_benchmark_samples(args.benchmark),
    }
    handler = partial(StudioHandler, directory=str(STUDIO_DIR), app_state=app_state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"RLM Studio listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
