from __future__ import annotations

import importlib.util
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


curator_mod = load_module("curate_product_cases", INFERENCE_DIR / "curate_product_cases.py")


def test_build_candidate_from_log_infers_constraints() -> None:
    log_row = {
        "ts": "2026-03-19T12:00:00Z",
        "request": {
            "mode": "deep",
            "instruction": "Fix the note and keep glossary terms exact.",
            "text": "the `ANEBridge` helper works on apple silicon",
            "reference": "The `ANEBridge` helper works on Apple Silicon.",
            "memory": ["Use the glossary term Apple Silicon."],
            "preserve_terms": "Apple Silicon",
            "metadata": {"category": "ethertext", "difficulty": "hard"},
        },
        "response": {
            "config": {"model": "qwen3.5:122b"},
        },
    }

    candidate = curator_mod.build_candidate_from_log(
        log_row,
        Path("/tmp/studio-log.jsonl"),
        mode_filter="any",
    )

    assert candidate is not None
    assert candidate["instruction"] == "Fix the note and keep glossary terms exact."
    assert candidate["required_terms"] == ["Apple Silicon", "`ANEBridge`"]
    assert candidate["forbidden_terms"] == ["apple silicon", "`anebridge`"]
    assert candidate["memory"] == ["Use the glossary term Apple Silicon."]
    assert candidate["metadata"]["request_model"] == "qwen3.5:122b"


def test_decide_case_status_promotes_when_deep_materially_beats_quick() -> None:
    quick_eval = {
        "output": "quick output",
        "scores": {
            "pass_score": 0.0,
            "strict_quality_score": 0.70,
        },
    }
    deep_eval = {
        "output": "deep output",
        "scores": {
            "pass_score": 1.0,
            "strict_quality_score": 0.92,
        },
    }

    status = curator_mod.decide_case_status(
        quick_eval,
        deep_eval,
        min_disagreement=0.03,
        min_deep_gain=0.03,
    )

    assert status == "promote"


def test_build_candidate_from_log_skips_untrusted_reference_without_acceptance() -> None:
    log_row = {
        "request": {
            "mode": "deep",
            "text": "i'm just checking in to see if you got my last note",
            "reference": "The new ANE bridge makes autoresearch on Apple Silicon feel way more useful.",
        },
        "response": {
            "ok": True,
            "output": "I'm just checking in to see whether you got my last note.",
        },
    }

    candidate = curator_mod.build_candidate_from_log(
        log_row,
        Path("/tmp/studio-log.jsonl"),
        mode_filter="any",
    )

    assert candidate is None
