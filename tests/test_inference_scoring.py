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


def test_select_memory_prioritizes_required_style_constraints() -> None:
    example = mod.Example(
        id="style-pref",
        instruction="Fix the sentence using the stored style preference and preserve glossary terms exactly.",
        input="im not sure if autoresearch should default to ane on this mac yet but its worth testing",
        reference="I'm not sure whether autoresearch should default to ANE on this Mac yet, but it's worth testing.",
        memory=[
            "User style preference: prefer 'whether' over 'if' in formal writing.",
            "Preserve glossary terms exactly: autoresearch, ANE, Mac.",
            "Keep the uncertainty in the sentence.",
        ],
        preserve_terms=["autoresearch", "ANE", "Mac"],
        required_terms=["whether", "autoresearch", "ANE", "Mac"],
        forbidden_terms=["definitely", "certainly"],
        checks={},
        metadata={"difficulty": "hard"},
    )

    selected = mod.select_memory(example, top_k=1)

    assert selected == ["User style preference: prefer 'whether' over 'if' in formal writing."]


def test_build_direct_prompt_includes_required_and_forbidden_terms() -> None:
    example = mod.Example(
        id="prompt-constraints",
        instruction="Fix the sentence using the stored style preference and preserve glossary terms exactly.",
        input="im not sure if autoresearch should default to ane on this mac yet but its worth testing",
        reference="I'm not sure whether autoresearch should default to ANE on this Mac yet, but it's worth testing.",
        memory=[],
        preserve_terms=["autoresearch", "ANE", "Mac"],
        required_terms=["whether", "autoresearch", "ANE", "Mac"],
        forbidden_terms=["definitely", "certainly"],
        checks={},
        metadata={},
    )

    prompt = mod.build_direct_prompt(example)

    assert "Terms to preserve exactly:" in prompt
    assert "Required terms to include:" in prompt
    assert "Terms to avoid:" in prompt
    assert "whether, autoresearch, ANE, Mac" in prompt
    assert "definitely, certainly" in prompt
