from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "inference" / "run_grammar_bench.py"
spec = importlib.util.spec_from_file_location("run_grammar_bench", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def make_example(input_text: str, reference: str) -> object:
    return mod.Example(
        id="ex",
        instruction="Fix grammar with minimal edits.",
        input=input_text,
        reference=reference,
        memory=[],
        preserve_terms=[],
        required_terms=[],
        forbidden_terms=[],
        checks={},
        metadata={},
    )


def test_score_example_penalizes_unnecessary_style_drift() -> None:
    example = make_example(
        "i'm just checking in to see if you got my last note",
        "I'm just checking in to see whether you got my last note.",
    )

    aligned = mod.score_example(example, "I'm just checking in to see whether you got my last note.")
    drifted = mod.score_example(example, "I'm just checking in to see whether you received my last note.")

    assert "drift_score" in aligned
    assert aligned["drift_score"] > drifted["drift_score"]
    assert aligned["strict_quality_score"] > drifted["strict_quality_score"]


def test_score_example_penalizes_non_reference_synonym_swap() -> None:
    example = make_example(
        "the new ane bridge make autoresearch on apple silicon feel way more usefull.",
        "The new ANE bridge makes autoresearch on Apple Silicon feel way more useful.",
    )

    aligned = mod.score_example(example, "The new ANE bridge makes autoresearch on Apple Silicon feel way more useful.")
    drifted = mod.score_example(example, "The new ANE bridge makes autoresearch on Apple Silicon feel much more useful.")

    assert aligned["drift_score"] > drifted["drift_score"]
    assert aligned["strict_quality_score"] > drifted["strict_quality_score"]
