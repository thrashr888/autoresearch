from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "tools" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


research_mod = load_module("research_claw_lite", INFERENCE_DIR / "research_claw_lite.py")
studio_mod = load_module("studio_server", INFERENCE_DIR / "studio_server.py")
bench_mod = load_module("run_grammar_bench", INFERENCE_DIR / "run_grammar_bench.py")


def test_select_promotions_picks_fast_and_deep_candidates() -> None:
    results = [
        {
            "candidate": "memory_flat_k2",
            "controller": "memory_flat",
            "memory_top_k": 2,
            "temperature": 0.0,
            "num_ctx": 2048,
            "max_tokens": 96,
            "summary": {
                "avg_strict_quality_score": 0.90,
                "avg_drift_score": 0.72,
                "p95_latency_ms": 3200.0,
            },
            "score": 900.0,
            "status": "ok",
        },
        {
            "candidate": "rlm_adaptive_k3",
            "controller": "rlm_adaptive",
            "memory_top_k": 3,
            "temperature": 0.0,
            "num_ctx": 2048,
            "max_tokens": 96,
            "summary": {
                "avg_strict_quality_score": 0.90,
                "avg_drift_score": 0.72,
                "p95_latency_ms": 1300.0,
            },
            "score": 905.0,
            "status": "ok",
        },
        {
            "candidate": "rlm_recursive_k4",
            "controller": "rlm_recursive",
            "memory_top_k": 4,
            "temperature": 0.0,
            "num_ctx": 2048,
            "max_tokens": 96,
            "summary": {
                "avg_strict_quality_score": 0.9915,
                "avg_drift_score": 1.0,
                "p95_latency_ms": 8858.8,
            },
            "score": 991.0,
            "status": "ok",
        },
    ]

    promotions = research_mod.select_promotions(results, profile="rlm", backend="ollama", model="mistral:latest")

    assert promotions["quick"]["candidate"] == "rlm_adaptive_k3"
    assert promotions["deep"]["candidate"] == "rlm_recursive_k4"
    assert promotions["quick"]["config"]["controller"] == "rlm_adaptive"
    assert promotions["deep"]["config"]["controller"] == "rlm_recursive"


def test_resolve_models_dedupes_profile_and_explicit_models() -> None:
    args = type(
        "Args",
        (),
        {
            "model": "mistral:latest",
            "models": ["qwen3:30b,qwen3.5:122b", "qwen3:30b"],
            "model_profile": "m5max_local",
        },
    )()

    models = research_mod.resolve_models(args)

    assert models == ["qwen3:30b", "gpt-oss:120b", "qwen3.5:122b"]


def test_resolve_models_extended_profile_keeps_curated_order() -> None:
    args = type(
        "Args",
        (),
        {
            "model": "mistral:latest",
            "models": [],
            "model_profile": "m5max_local_extended",
        },
    )()

    models = research_mod.resolve_models(args)

    assert models == [
        "qwen3.5:122b",
        "gpt-oss:120b",
        "llama4:latest",
        "mixtral:8x22b",
        "llama3.1:70b",
        "qwen3:30b",
    ]


def test_resolve_models_focus_profile_keeps_quality_shortlist_order() -> None:
    args = type(
        "Args",
        (),
        {
            "model": "mistral:latest",
            "models": [],
            "model_profile": "m5max_local_focus",
        },
    )()

    models = research_mod.resolve_models(args)

    assert models == [
        "qwen3.5:122b",
        "mixtral:8x22b",
        "llama3.1:70b",
        "qwen3:30b",
    ]


def test_candidate_pool_screen_skips_recursive_rlm_variant() -> None:
    candidates = research_mod.candidate_pool("rlm", "screen")

    assert [candidate.name for candidate in candidates] == ["memory_flat_k2", "rlm_lite_k3"]


def test_select_global_promotions_prefers_fast_quick_and_best_deep() -> None:
    model_reports = [
        {
            "model": "qwen3:30b",
            "promotions": {
                "quick": {
                    "candidate": "memory_flat_k2",
                    "config": {"backend": "ollama", "model": "qwen3:30b", "controller": "memory_flat"},
                    "summary": {"avg_strict_quality_score": 0.92, "p95_latency_ms": 2400.0},
                    "score": 920.0,
                },
                "deep": {
                    "candidate": "rlm_recursive_k3",
                    "config": {"backend": "ollama", "model": "qwen3:30b", "controller": "rlm_recursive"},
                    "summary": {"avg_strict_quality_score": 0.95, "p95_latency_ms": 8200.0},
                    "score": 950.0,
                },
            },
        },
        {
            "model": "qwen3.5:122b",
            "promotions": {
                "quick": {
                    "candidate": "memory_flat_k2",
                    "config": {"backend": "ollama", "model": "qwen3.5:122b", "controller": "memory_flat"},
                    "summary": {"avg_strict_quality_score": 0.94, "p95_latency_ms": 3600.0},
                    "score": 940.0,
                },
                "deep": {
                    "candidate": "rlm_recursive_k3",
                    "config": {"backend": "ollama", "model": "qwen3.5:122b", "controller": "rlm_recursive"},
                    "summary": {"avg_strict_quality_score": 0.98, "p95_latency_ms": 12800.0},
                    "score": 980.0,
                },
            },
        },
    ]

    promotions = research_mod.select_global_promotions(model_reports, profile="rlm")

    assert promotions["quick"]["config"]["model"] == "qwen3:30b"
    assert promotions["deep"]["config"]["model"] == "qwen3.5:122b"


def test_ollama_think_auto_matches_new_local_models() -> None:
    qwen = bench_mod.OllamaBackend(
        host="http://localhost:11434",
        model="qwen3.5:122b",
        keep_alive="15m",
        request_timeout_s=120.0,
        think="auto",
    )
    gpt_oss = bench_mod.OllamaBackend(
        host="http://localhost:11434",
        model="gpt-oss:120b",
        keep_alive="15m",
        request_timeout_s=120.0,
        think="auto",
    )

    assert qwen.resolve_think() is False
    assert gpt_oss.resolve_think() == "low"


def test_strict_scoring_respects_exact_preserve_terms_and_case_trap_forbidden_terms() -> None:
    example = bench_mod.Example(
        id="case-sensitive-glossary",
        instruction="Fix the sentence and keep glossary terms exact.",
        input="ane runs on apple silicon",
        reference="ANE runs on Apple Silicon.",
        memory=[],
        preserve_terms=["ANE", "Apple Silicon"],
        required_terms=["ANE", "Apple Silicon"],
        forbidden_terms=["ane", "apple silicon"],
    )

    correct = bench_mod.score_example(example, "ANE runs on Apple Silicon.")
    wrong_case = bench_mod.score_example(example, "ane runs on apple silicon.")

    assert correct["preserve_score"] == 1.0
    assert correct["required_score"] == 1.0
    assert correct["forbidden_score"] == 1.0
    assert wrong_case["preserve_score"] == 0.0
    assert wrong_case["required_score"] == 0.0
    assert wrong_case["forbidden_score"] == 0.0


def test_load_promoted_configs_prefers_promotions_json(tmp_path: Path) -> None:
    run_root = tmp_path / "runs_research_claw"
    session_dir = run_root / "session-a"
    session_dir.mkdir(parents=True)
    latest = run_root / "latest"
    latest.symlink_to(session_dir.name)
    promotions = {
        "quick": {
            "candidate": "rlm_adaptive_k3",
            "config": {"backend": "ollama", "model": "mistral:latest", "controller": "rlm_adaptive", "memory_top_k": 3},
            "summary": {"avg_strict_quality_score": 0.90},
        },
        "deep": {
            "candidate": "rlm_recursive_k4",
            "config": {"backend": "ollama", "model": "mistral:latest", "controller": "rlm_recursive", "memory_top_k": 4},
            "summary": {"avg_strict_quality_score": 0.99},
        },
    }
    (session_dir / "promotions.json").write_text(json.dumps(promotions), encoding="utf-8")

    quick, deep, meta = studio_mod.load_promoted_configs(run_root)

    assert quick["controller"] == "rlm_adaptive"
    assert deep["controller"] == "rlm_recursive"
    assert meta["source"].endswith("promotions.json")
