#!/usr/bin/env python3
"""Run a grammar/text-fixer benchmark against a local backend and log quality/runtime metrics."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are Ethertext's local writing assistant. Fix grammar, spelling, and punctuation with minimal edits. "
    "Preserve meaning, preserve formatting when asked, preserve named entities and glossary terms exactly, "
    "and return only the corrected text. Do not paraphrase, add politeness, add explanations, or make the text "
    "more formal unless the instruction explicitly requires it. Preserve sentence mood and point of view: keep "
    "questions as questions, keep requests as requests, keep terse notes terse, and do not replace words with "
    "synonyms unless grammar requires it."
)

MLX_QWEN_TEXT_MARKERS = ("qwen3.5", "text-")


def mlx_should_ignore_chat_template(model: str) -> bool:
    lowered = (model or "").lower()
    return all(marker in lowered for marker in MLX_QWEN_TEXT_MARKERS)


def extract_mlx_text(stdout: str) -> str:
    match = re.search(r"=+\n(.*?)\n=+\nPrompt:", stdout or "", re.S)
    text = match.group(1).strip() if match else (stdout or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    final_match = re.search(r'(?im)^FINAL:\s*(.+)$', text)
    if final_match:
        return final_match.group(1).strip()
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


@dataclass
class Example:
    id: str
    instruction: str
    input: str
    reference: str
    memory: list[str]
    preserve_terms: list[str]
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    ttft_ms: float | None = None
    decode_tok_per_s: float | None = None
    peak_memory_mb: float | None = None
    subcalls: int = 1
    backend_details: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def similarity(lhs: str, rhs: str) -> float:
    return SequenceMatcher(None, normalize_text(lhs), normalize_text(rhs)).ratio()


def token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z0-9_']+", normalize_text(text).lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def positive_delta(lhs: dict[str, int], rhs: dict[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for token in lhs.keys() | rhs.keys():
        value = lhs.get(token, 0) - rhs.get(token, 0)
        if value > 0:
            delta[token] = value
    return delta


def score_drift(example: Example, output: str) -> float:
    input_counts = token_counts(example.input)
    reference_counts = token_counts(example.reference)
    output_counts = token_counts(output)

    removed_ref = positive_delta(input_counts, reference_counts)
    added_ref = positive_delta(reference_counts, input_counts)
    removed_out = positive_delta(input_counts, output_counts)
    added_out = positive_delta(output_counts, input_counts)

    unexpected_remove = sum(max(0, removed_out.get(token, 0) - removed_ref.get(token, 0)) for token in removed_out)
    unexpected_add = sum(max(0, added_out.get(token, 0) - added_ref.get(token, 0)) for token in added_out)
    missed_remove = sum(max(0, removed_ref.get(token, 0) - removed_out.get(token, 0)) for token in removed_ref)
    missed_add = sum(max(0, added_ref.get(token, 0) - added_out.get(token, 0)) for token in added_ref)

    penalty_units = unexpected_remove + unexpected_add + (0.5 * missed_remove) + (0.5 * missed_add)
    baseline_units = sum(removed_ref.values()) + sum(added_ref.values())
    if penalty_units <= 0:
        return 1.0
    scale = max(2.0, float(baseline_units + unexpected_remove + unexpected_add))
    return max(0.0, 1.0 - (penalty_units / scale))


def load_examples(path: Path) -> list[Example]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    id=row["id"],
                    instruction=row["instruction"],
                    input=row["input"],
                    reference=row["reference"],
                    memory=row.get("memory", []),
                    preserve_terms=row.get("preserve_terms", []),
                    required_terms=row.get("required_terms", []),
                    forbidden_terms=row.get("forbidden_terms", []),
                    checks=row.get("checks", {}),
                    metadata=row.get("metadata", {}),
                )
            )
    return examples


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def contains_term(output: str, term: str, *, case_sensitive: bool) -> bool:
    needle = term.strip()
    if not needle:
        return False
    if case_sensitive:
        return needle in output
    return needle.lower() in output.lower()


def preserve_term_present(output: str, term: str) -> bool:
    return contains_term(output, term, case_sensitive=True)


def required_term_present(example: Example, output: str, term: str) -> bool:
    normalized = term.strip().lower()
    exact = any(normalized == preserve.strip().lower() for preserve in example.preserve_terms)
    return contains_term(output, term, case_sensitive=exact)


def forbidden_term_present(example: Example, output: str, term: str) -> bool:
    normalized = term.strip().lower()
    related_required = any(normalized == required.strip().lower() for required in example.required_terms)
    related_preserve = any(normalized == preserve.strip().lower() for preserve in example.preserve_terms)
    exact = related_required or related_preserve
    return contains_term(output, term, case_sensitive=exact)


def select_memory(example: Example, top_k: int) -> list[str]:
    if not example.memory:
        return []
    checks = example.checks or {}
    source_terms = set(re.findall(r"[A-Za-z0-9_]+", example.input.lower()))
    source_terms.update(re.findall(r"[A-Za-z0-9_]+", example.instruction.lower()))
    source_terms.update(term.lower().strip("`") for term in example.preserve_terms)
    source_terms.update(term.lower().strip("`") for term in example.required_terms)
    has_code_constraint = checks.get("preserve_inline_code") or bool(expected_inline_code_spans(example))
    has_bullet_constraint = checks.get("preserve_bullets") or any(is_bullet_line(line) for line in example.input.splitlines())
    has_line_break_constraint = checks.get("preserve_line_breaks")
    weekday_terms = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    input_terms = set(re.findall(r"[A-Za-z0-9_]+", example.input.lower()))
    required_terms = {term.lower().strip("`") for term in example.required_terms}
    preserve_terms = {term.lower().strip("`") for term in example.preserve_terms}
    style_required_terms = {term for term in required_terms if term and term not in preserve_terms}
    ranked = []
    for item in example.memory:
        lowered = item.lower()
        mem_terms = set(re.findall(r"[A-Za-z0-9_]+", lowered))
        overlap = len(source_terms & mem_terms)
        preserve_hits = sum(1 for term in example.preserve_terms if term.lower().strip("`") in lowered)
        required_hits = sum(1 for term in required_terms if term and term in lowered)
        style_required_hits = sum(1 for term in style_required_terms if term in lowered)
        score = overlap + (2 * preserve_hits) + (3 * required_hits) + (8 * style_required_hits)
        if lowered.startswith("preserve "):
            score += 3
        if "style preference" in lowered or lowered.startswith("style:"):
            score += 3
            if style_required_hits:
                score += 4
        if "capitalize" in lowered:
            score += 2
        if "unrelated note" in lowered:
            score -= 8
        if has_code_constraint and "inline code" in lowered:
            score += 5
        if has_bullet_constraint and "bullet" in lowered:
            score += 5
        if has_line_break_constraint and "line break" in lowered:
            score += 4
        if example.preserve_terms and "name" in lowered:
            score += 3
        if "whether" in lowered and ("if" in input_terms or "whether" in required_terms):
            score += 6
        if ("calendar" in lowered or "weekday" in lowered) and (weekday_terms & input_terms):
            score += 3
        ranked.append((score, item))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = [item for _, item in ranked[:top_k]]
    return chosen or example.memory[:top_k]


def extract_inline_code_spans(text: str) -> list[str]:
    return re.findall(r"`[^`]+`", text)


def nonempty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def is_bullet_line(text: str) -> bool:
    return bool(re.match(r"^\s*[-*]\s+", text))


def bullet_marker(text: str) -> str | None:
    match = re.match(r"^\s*([-*])\s+", text)
    return None if match is None else match.group(1)


def expected_inline_code_spans(example: Example) -> list[str]:
    preserve_code_terms = [term for term in example.preserve_terms if term.startswith("`") and term.endswith("`")]
    if preserve_code_terms:
        return preserve_code_terms
    reference_code_spans = extract_inline_code_spans(example.reference)
    if reference_code_spans:
        return reference_code_spans
    return extract_inline_code_spans(example.input)


def formatting_guardrails(example: Example) -> list[str]:
    checks = example.checks or {}
    guards: list[str] = []
    nonempty_input_lines = nonempty_lines(example.input)
    bullet_lines = [line for line in example.input.splitlines() if is_bullet_line(line)]
    paragraph_count = len([block for block in example.input.split("\n\n") if block.strip()])

    if checks.get("preserve_line_breaks"):
        guards.append(
            f"Formatting rule: keep the same number of paragraphs as the input ({paragraph_count}) and do not collapse the text into fewer paragraphs."
        )
        guards.append(
            f"Formatting rule: keep the same number of non-empty lines as the input ({len(nonempty_input_lines)}) unless a line only changes by punctuation or capitalization."
        )
    if checks.get("preserve_bullets") or bullet_lines:
        guards.append(
            f"Formatting rule: preserve the bullet list and keep all {len(bullet_lines)} bullet items in the output. Do not drop any bullets."
        )
    if (checks.get("preserve_bullets") or checks.get("preserve_line_breaks")) and nonempty_input_lines:
        guards.append(
            "Formatting rule: return the full corrected text, not a summary and not a partial excerpt from the input."
        )
    return guards


def needs_linewise_format_preservation(example: Example) -> bool:
    checks = example.checks or {}
    return bool(checks.get("preserve_bullets") or checks.get("preserve_line_breaks"))


def build_line_edit_prompt(
    example: Example,
    line: str,
    memory: list[str] | None = None,
    plan: str | None = None,
) -> str:
    sections = [
        SYSTEM_PROMPT,
        f"Task: {example.instruction}",
        "You are correcting exactly one line from a larger formatted text.",
        "Return the full corrected line and nothing else.",
        "Do not drop content from the line.",
    ]
    if is_bullet_line(line):
        marker = bullet_marker(line) or "-"
        sections.append(f"Formatting rule: preserve the bullet marker `{marker}` and keep this as a bullet line.")
    else:
        sections.append("Formatting rule: keep this as a non-bullet line.")
    if memory:
        sections.append("Relevant memory:")
        sections.extend(f"- {item}" for item in memory)
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if plan:
        sections.append("Editing plan:")
        sections.append(plan.strip())
    sections.append("Line:")
    sections.append(line)
    sections.append("Corrected line:")
    return "\n".join(sections)


def run_linewise_format_controller(
    backend: Backend,
    example: Example,
    *,
    memory: list[str],
    plan: str | None,
    max_tokens: int,
    temperature: float,
    num_ctx: int,
) -> GenerationResult:
    calls: list[GenerationResult] = []
    output_lines: list[str] = []
    for line in example.input.splitlines():
        if not line.strip():
            output_lines.append("")
            continue
        result = backend.generate(
            build_line_edit_prompt(example, line, memory=memory, plan=plan),
            max_tokens=max_tokens,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        calls.append(result)
        output_lines.append(result.text.strip())
    return combine_results("\n".join(output_lines), calls)


def build_direct_prompt(example: Example, memory: list[str] | None = None, plan: str | None = None) -> str:
    sections = [SYSTEM_PROMPT]
    sections.append(f"Task: {example.instruction}")
    sections.append(
        "Editing rule: prefer the smallest possible correction. Keep terse text terse. "
        "Do not turn questions into statements, do not turn requests into commands, and do not expand short words like "
        "'docs' into longer synonyms unless required."
    )
    sections.extend(formatting_guardrails(example))
    if memory:
        sections.append("Relevant memory:")
        sections.extend(f"- {item}" for item in memory)
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if example.required_terms:
        sections.append("Required terms to include:")
        sections.append(", ".join(example.required_terms))
    if example.forbidden_terms:
        sections.append("Terms to avoid:")
        sections.append(", ".join(example.forbidden_terms))
    if plan:
        sections.append("Editing plan:")
        sections.append(plan.strip())
    sections.append("Text:")
    sections.append(example.input)
    sections.append("Corrected text:")
    return "\n".join(sections)


def build_plan_prompt(example: Example, memory: list[str]) -> str:
    sections = [
        SYSTEM_PROMPT,
        "You are in planning mode. Summarize the editing constraints that matter for this text.",
        "Return at most three short bullet points and nothing else.",
        f"Task: {example.instruction}",
        "Relevant memory:",
    ]
    sections.extend(f"- {item}" for item in memory)
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if example.required_terms:
        sections.append("Required terms to include:")
        sections.append(", ".join(example.required_terms))
    if example.forbidden_terms:
        sections.append("Terms to avoid:")
        sections.append(", ".join(example.forbidden_terms))
    sections.append("Text:")
    sections.append(example.input)
    sections.append("Relevant editing constraints:")
    return "\n".join(sections)


def build_memory_note_prompt(example: Example, memory_item: str) -> str:
    sections = [
        SYSTEM_PROMPT,
        "You are reviewing one memory snippet for a local writing assistant.",
        'If the memory affects the edit, return exactly one short constraint prefixed with "preserve:", "style:", "format:", or "rewrite:".',
        "Do not rewrite the user's text. Do not quote the sentence. Do not explain.",
        'If it does not matter, return exactly "irrelevant".',
        f"Task: {example.instruction}",
        "Memory snippet:",
        memory_item,
    ]
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if example.required_terms:
        sections.append("Required terms to include:")
        sections.append(", ".join(example.required_terms))
    if example.forbidden_terms:
        sections.append("Terms to avoid:")
        sections.append(", ".join(example.forbidden_terms))
    sections.append("Text:")
    sections.append(example.input)
    sections.append("Relevant constraint:")
    return "\n".join(sections)


def build_merge_plan_prompt(example: Example, notes: list[str]) -> str:
    sections = [
        SYSTEM_PROMPT,
        "You are merging memory notes for a local writing assistant.",
        "Return at most three short bullet points and nothing else.",
        "Each bullet must be a constraint, not a rewritten sentence.",
        f"Task: {example.instruction}",
        "Memory notes:",
    ]
    sections.extend(f"- {note}" for note in notes)
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if example.required_terms:
        sections.append("Required terms to include:")
        sections.append(", ".join(example.required_terms))
    if example.forbidden_terms:
        sections.append("Terms to avoid:")
        sections.append(", ".join(example.forbidden_terms))
    sections.append("Text:")
    sections.append(example.input)
    sections.append("Merged editing constraints:")
    return "\n".join(sections)


def seems_memory_heavy(example: Example, memory: list[str], issues: list[str]) -> bool:
    metadata = example.metadata or {}
    difficulty = str(metadata.get("difficulty", "")).lower()
    lowered_memory = [item.lower() for item in memory]
    checks = example.checks or {}

    if difficulty == "hard":
        return True
    if checks.get("preserve_bullets") or checks.get("preserve_line_breaks"):
        return True
    if issues:
        return True
    if any("unrelated note" in item for item in lowered_memory):
        return True
    if len(memory) >= 4:
        return True
    if len(memory) >= 3 and len(example.input.split()) >= 18:
        return True
    if len(memory) >= 3 and any("style preference" in item for item in lowered_memory) and example.preserve_terms:
        return True
    return False


def build_revision_prompt(example: Example, draft: str, memory: list[str], plan: str, issues: list[str]) -> str:
    sections = [
        SYSTEM_PROMPT,
        "Revise the draft so it satisfies the listed issues with minimal edits.",
        "Return only the revised text and nothing else.",
        f"Task: {example.instruction}",
    ]
    sections.extend(formatting_guardrails(example))
    if memory:
        sections.append("Relevant memory:")
        sections.extend(f"- {item}" for item in memory)
    if example.preserve_terms:
        sections.append("Terms to preserve exactly:")
        sections.append(", ".join(example.preserve_terms))
    if plan:
        sections.append("Editing plan:")
        sections.append(plan.strip())
    sections.append("Issues to fix:")
    sections.extend(f"- {issue}" for issue in issues)
    sections.append("Original text:")
    sections.append(example.input)
    sections.append("Current draft:")
    sections.append(draft)
    sections.append("Revised text:")
    return "\n".join(sections)


class Backend:
    def generate(self, prompt: str, *, max_tokens: int, temperature: float, num_ctx: int) -> GenerationResult:
        raise NotImplementedError


def combine_results(text: str, calls: list[GenerationResult]) -> GenerationResult:
    ttft_values = [call.ttft_ms for call in calls if call.ttft_ms is not None]
    peak_values = [call.peak_memory_mb for call in calls if call.peak_memory_mb is not None]
    decode_tok_per_s = None
    for call in reversed(calls):
        if call.decode_tok_per_s is not None:
            decode_tok_per_s = call.decode_tok_per_s
            break
    return GenerationResult(
        text=text,
        latency_ms=sum(call.latency_ms for call in calls),
        ttft_ms=sum(ttft_values) if len(ttft_values) == len(calls) else None,
        decode_tok_per_s=decode_tok_per_s,
        peak_memory_mb=max(peak_values) if peak_values else None,
        subcalls=sum(call.subcalls for call in calls),
        backend_details={"calls": [call.backend_details for call in calls if call.backend_details]},
    )


def looks_like_meta_response(text: str) -> bool:
    lowered = normalize_text(text).lower()
    markers = [
        "the revised text",
        "here is the revised text",
        "here's the revised text",
        "no changes were made",
        "the corrected text",
        "issues to be fixed",
    ]
    return any(marker in lowered for marker in markers)


class MockBackend(Backend):
    RULES = {
        r"\bi has\b": "I have",
        r"\bi\b": "I",
        r"\byesturday\b": "yesterday",
        r"\bbuyed\b": "bought",
        r"\bteh\b": "the",
        r"\bwere looking\b": "we're looking",
        r"\bill\b": "I'll",
        r"\bim\b": "I'm",
        r"\bcant\b": "can't",
        r"\bdont\b": "don't",
        r"\bsynnc\b": "sync",
    }

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, num_ctx: int) -> GenerationResult:
        text = prompt.split("Text:\n", 1)[-1].split("\nCorrected text:", 1)[0].strip()
        fixed = text
        for pattern, replacement in self.RULES.items():
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s+", " ", fixed).strip()
        if fixed and fixed[0].islower():
            fixed = fixed[0].upper() + fixed[1:]
        if fixed and fixed[-1] not in ".!?`":
            fixed += "."
        return GenerationResult(text=fixed, latency_ms=1.0, ttft_ms=0.5, decode_tok_per_s=10_000.0)


class OllamaBackend(Backend):
    def __init__(self, host: str, model: str, keep_alive: str, request_timeout_s: float, think: str):
        self.url = host.rstrip("/") + "/api/generate"
        self.model = model
        self.keep_alive = keep_alive
        self.request_timeout_s = request_timeout_s
        self.think = think

    def resolve_think(self) -> bool | str | None:
        mode = self.think.strip().lower()
        if mode in {"", "auto"}:
            lowered = self.model.lower()
            if lowered.startswith("gpt-oss"):
                return "low"
            if lowered.startswith("qwen3") or lowered.startswith("qwen3.5"):
                return False
            return None
        if mode in {"false", "off", "disabled"}:
            return False
        if mode in {"true", "on", "enabled"}:
            return True
        if mode in {"low", "medium", "high"}:
            return mode
        return None

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, num_ctx: int) -> GenerationResult:
        think_mode = self.resolve_think()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
        }
        if think_mode is not None:
            payload["think"] = think_mode
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
            except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError(f"Could not reach Ollama at {self.url}: {exc}") from exc
                time.sleep(1.0 + attempt)
        latency_ms = (time.perf_counter() - started) * 1000
        load_duration = data.get("load_duration") or 0
        prompt_eval_duration = data.get("prompt_eval_duration") or 0
        eval_duration = data.get("eval_duration") or 0
        eval_count = data.get("eval_count") or 0
        ttft_ms = None
        if load_duration or prompt_eval_duration:
            ttft_ms = (load_duration + prompt_eval_duration) / 1_000_000
        decode_tok_per_s = None
        if eval_duration and eval_count:
            decode_tok_per_s = eval_count / (eval_duration / 1_000_000_000)
        thinking = data.get("thinking") or ""
        return GenerationResult(
            text=(data.get("response") or "").strip(),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            decode_tok_per_s=decode_tok_per_s,
            backend_details={
                "think_mode": think_mode,
                "thinking_chars": len(thinking),
                "done_reason": data.get("done_reason"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns": data.get("load_duration"),
            },
        )


class MlxBackend(Backend):
    def __init__(self, model: str, python_bin: str):
        self.model = model
        self.python_bin = python_bin

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, num_ctx: int) -> GenerationResult:
        command = [
            "/usr/bin/time",
            "-l",
            self.python_bin,
            "-m",
            "mlx_lm.generate",
            "--model",
            self.model,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--max-kv-size",
            str(num_ctx),
        ]
        if mlx_should_ignore_chat_template(self.model):
            command.append("--ignore-chat-template")
        started = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        latency_ms = (time.perf_counter() - started) * 1000
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "mlx_lm.generate failed")
        peak_memory_mb = None
        match = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
        if match:
            peak_memory_mb = int(match.group(1)) / 1024 / 1024
        text = extract_mlx_text(proc.stdout)
        return GenerationResult(text=text, latency_ms=latency_ms, peak_memory_mb=peak_memory_mb)


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "ollama":
        return OllamaBackend(
            host=args.ollama_host,
            model=args.model,
            keep_alive=args.ollama_keep_alive,
            request_timeout_s=args.request_timeout_s,
            think=args.ollama_think,
        )
    if args.backend == "mlx":
        return MlxBackend(model=args.model, python_bin=args.mlx_python_bin)
    raise ValueError(f"Unsupported backend: {args.backend}")


def score_structure(example: Example, output: str) -> tuple[float, list[str]]:
    checks = example.checks or {}
    scores: list[float] = []
    issues: list[str] = []

    code_spans = expected_inline_code_spans(example)
    if checks.get("preserve_inline_code") or code_spans:
        if code_spans:
            hits = sum(1 for span in code_spans if span in output)
            score = hits / len(code_spans)
            scores.append(score)
            if hits != len(code_spans):
                missing = [span for span in code_spans if span not in output]
                issues.append(f"Preserve inline code exactly: {', '.join(missing)}")

    input_lines = nonempty_lines(example.input)
    output_lines = nonempty_lines(output)
    input_bullets = [line for line in input_lines if is_bullet_line(line)]
    output_bullets = [line for line in output_lines if is_bullet_line(line)]
    if checks.get("preserve_bullets") or input_bullets:
        if input_bullets:
            count_delta = abs(len(output_bullets) - len(input_bullets))
            marker_matches = 0
            for input_line, output_line in zip(input_bullets, output_bullets):
                if bullet_marker(input_line) == bullet_marker(output_line):
                    marker_matches += 1
            baseline = len(input_bullets)
            score = max(0.0, (marker_matches - count_delta) / baseline)
            scores.append(score)
            if count_delta or marker_matches != min(len(input_bullets), len(output_bullets)):
                issues.append("Preserve the bullet layout, bullet count, and bullet markers.")

    if checks.get("preserve_line_breaks"):
        baseline = max(1, len(input_lines))
        delta = abs(len(output_lines) - len(input_lines))
        score = max(0.0, 1.0 - (delta / baseline))
        scores.append(score)
        if delta:
            issues.append("Preserve the original line breaks.")

    if not scores:
        return 1.0, issues
    return statistics.fmean(scores), issues


def score_example(example: Example, output: str) -> dict[str, float]:
    reference_score = similarity(output, example.reference)
    preserve_score = 1.0
    if example.preserve_terms:
        hits = sum(1 for term in example.preserve_terms if preserve_term_present(output, term))
        preserve_score = hits / len(example.preserve_terms)
    required_score = 1.0
    if example.required_terms:
        hits = sum(1 for term in example.required_terms if required_term_present(example, output, term))
        required_score = hits / len(example.required_terms)
    forbidden_score = 1.0
    if example.forbidden_terms:
        violations = sum(1 for term in example.forbidden_terms if forbidden_term_present(example, output, term))
        forbidden_score = max(0.0, 1.0 - (violations / len(example.forbidden_terms)))
    edit_ref = 1.0 - similarity(example.input, example.reference)
    edit_out = 1.0 - similarity(example.input, output)
    excess = max(0.0, edit_out - edit_ref - 0.10)
    restraint_score = max(0.0, 1.0 - (excess / 0.60))
    structure_score, _ = score_structure(example, output)
    drift_score = score_drift(example, output)
    quality_score = (
        (0.55 * reference_score)
        + (0.20 * preserve_score)
        + (0.10 * restraint_score)
        + (0.10 * structure_score)
        + (0.05 * drift_score)
    )
    constraint_score = statistics.fmean([preserve_score, required_score, forbidden_score, structure_score, drift_score])
    strict_quality_score = quality_score * constraint_score
    pass_score = (
        preserve_score == 1.0
        and required_score == 1.0
        and forbidden_score == 1.0
        and structure_score == 1.0
        and reference_score >= 0.90
        and restraint_score >= 0.80
    )
    return {
        "reference_score": reference_score,
        "preserve_score": preserve_score,
        "required_score": required_score,
        "forbidden_score": forbidden_score,
        "restraint_score": restraint_score,
        "structure_score": structure_score,
        "drift_score": drift_score,
        "constraint_score": constraint_score,
        "quality_score": quality_score,
        "strict_quality_score": strict_quality_score,
        "pass_score": 1.0 if pass_score else 0.0,
    }


def detect_revision_issues(example: Example, output: str) -> list[str]:
    issues: list[str] = []
    if looks_like_meta_response(output):
        issues.append("Return only the corrected text, not an explanation or meta-commentary.")
    missing_terms = [term for term in example.preserve_terms if not preserve_term_present(output, term)]
    if missing_terms:
        issues.append(f"Preserve these exact terms: {', '.join(missing_terms)}")
    missing_required = [term for term in example.required_terms if not required_term_present(example, output, term)]
    if missing_required:
        issues.append(f"Make sure the final text still includes: {', '.join(missing_required)}")
    forbidden_hits = [term for term in example.forbidden_terms if forbidden_term_present(example, output, term)]
    if forbidden_hits:
        issues.append(f"Do not introduce distractor content such as: {', '.join(forbidden_hits)}")
    _, structure_issues = score_structure(example, output)
    issues.extend(structure_issues)
    if score_example(example, output)["restraint_score"] < 0.75:
        issues.append("Do not over-rewrite the text; keep edits minimal.")
    deduped: list[str] = []
    seen = set()
    for issue in issues:
        if issue in seen:
            continue
        seen.add(issue)
        deduped.append(issue)
    return deduped


def run_controller(
    backend: Backend,
    example: Example,
    *,
    controller: str,
    max_tokens: int,
    temperature: float,
    num_ctx: int,
    memory_top_k: int,
) -> tuple[GenerationResult, dict[str, Any]]:
    selected_memory = select_memory(example, memory_top_k)
    debug: dict[str, Any] = {"selected_memory": selected_memory}
    if controller == "direct":
        if needs_linewise_format_preservation(example):
            debug["route"] = "linewise_direct"
            return run_linewise_format_controller(
                backend,
                example,
                memory=[],
                plan=None,
                max_tokens=max_tokens,
                temperature=temperature,
                num_ctx=num_ctx,
            ), debug
        prompt = build_direct_prompt(example)
        return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, num_ctx=num_ctx), debug
    if controller == "memory_flat":
        if needs_linewise_format_preservation(example):
            debug["route"] = "linewise_memory_flat"
            return run_linewise_format_controller(
                backend,
                example,
                memory=selected_memory,
                plan=None,
                max_tokens=max_tokens,
                temperature=temperature,
                num_ctx=num_ctx,
            ), debug
        prompt = build_direct_prompt(example, memory=selected_memory)
        return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, num_ctx=num_ctx), debug
    if controller == "rlm_adaptive":
        if not selected_memory:
            prompt = build_direct_prompt(example)
            debug["route"] = "direct"
            return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, num_ctx=num_ctx), debug

        flat_prompt = build_direct_prompt(example, memory=selected_memory)
        flat_result = backend.generate(
            flat_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        flat_issues = detect_revision_issues(example, flat_result.text)
        debug["flat_issues"] = flat_issues
        if not seems_memory_heavy(example, selected_memory, flat_issues):
            debug["route"] = "flat"
            return flat_result, debug

        debug["route"] = "recursive"
        note_calls: list[GenerationResult] = []
        memory_notes: list[str] = []
        for memory_item in selected_memory:
            note_result = backend.generate(
                build_memory_note_prompt(example, memory_item),
                max_tokens=min(40, max_tokens),
                temperature=0.0,
                num_ctx=num_ctx,
            )
            note_calls.append(note_result)
            note_text = note_result.text.strip()
            if note_text and note_text.lower() != "irrelevant":
                memory_notes.append(note_text)
        debug["memory_notes"] = memory_notes

        plan_source = memory_notes or selected_memory
        plan_result = backend.generate(
            build_merge_plan_prompt(example, plan_source),
            max_tokens=min(72, max_tokens),
            temperature=0.0,
            num_ctx=num_ctx,
        )
        debug["plan"] = plan_result.text

        draft_result = backend.generate(
            build_direct_prompt(example, memory=selected_memory, plan=plan_result.text),
            max_tokens=max_tokens,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        issues = detect_revision_issues(example, draft_result.text)
        debug["revision_issues"] = issues
        calls = [flat_result, *note_calls, plan_result, draft_result]
        final_text = draft_result.text

        if issues:
            revision_result = backend.generate(
                build_revision_prompt(example, draft_result.text, selected_memory, plan_result.text, issues),
                max_tokens=max_tokens,
                temperature=0.0,
                num_ctx=num_ctx,
            )
            calls.append(revision_result)
            final_text = revision_result.text

        return combine_results(final_text, calls), debug
    if controller == "rlm_lite":
        if not selected_memory:
            prompt = build_direct_prompt(example)
            return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, num_ctx=num_ctx), debug
        plan_prompt = build_plan_prompt(example, selected_memory)
        plan_result = backend.generate(
            plan_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            num_ctx=num_ctx,
        )
        debug["plan"] = plan_result.text
        if needs_linewise_format_preservation(example):
            debug["route"] = "linewise_rlm_lite"
            final_result = run_linewise_format_controller(
                backend,
                example,
                memory=selected_memory,
                plan=plan_result.text,
                max_tokens=max_tokens,
                temperature=temperature,
                num_ctx=num_ctx,
            )
            final_result.latency_ms += plan_result.latency_ms
            final_result.subcalls = plan_result.subcalls + final_result.subcalls
            if plan_result.ttft_ms is not None and final_result.ttft_ms is not None:
                final_result.ttft_ms += plan_result.ttft_ms
            if plan_result.peak_memory_mb and final_result.peak_memory_mb:
                final_result.peak_memory_mb = max(plan_result.peak_memory_mb, final_result.peak_memory_mb)
            return final_result, debug
        final_prompt = build_direct_prompt(example, memory=selected_memory, plan=plan_result.text)
        final_result = backend.generate(
            final_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        final_result.latency_ms += plan_result.latency_ms
        final_result.subcalls = plan_result.subcalls + final_result.subcalls
        if plan_result.ttft_ms is not None and final_result.ttft_ms is not None:
            final_result.ttft_ms += plan_result.ttft_ms
        if plan_result.peak_memory_mb and final_result.peak_memory_mb:
            final_result.peak_memory_mb = max(plan_result.peak_memory_mb, final_result.peak_memory_mb)
        return final_result, debug
    if controller == "rlm_recursive":
        if not selected_memory:
            prompt = build_direct_prompt(example)
            return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, num_ctx=num_ctx), debug

        note_calls: list[GenerationResult] = []
        memory_notes: list[str] = []
        for memory_item in selected_memory:
            note_result = backend.generate(
                build_memory_note_prompt(example, memory_item),
                max_tokens=min(48, max_tokens),
                temperature=0.0,
                num_ctx=num_ctx,
            )
            note_calls.append(note_result)
            note_text = note_result.text.strip()
            if note_text and note_text.lower() != "irrelevant":
                memory_notes.append(note_text)
        debug["memory_notes"] = memory_notes

        plan_source = memory_notes or selected_memory
        plan_result = backend.generate(
            build_merge_plan_prompt(example, plan_source),
            max_tokens=min(96, max_tokens),
            temperature=0.0,
            num_ctx=num_ctx,
        )
        debug["plan"] = plan_result.text

        draft_result = backend.generate(
            build_direct_prompt(example, memory=selected_memory, plan=plan_result.text),
            max_tokens=max_tokens,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        issues = detect_revision_issues(example, draft_result.text)
        debug["revision_issues"] = issues
        calls = [*note_calls, plan_result, draft_result]
        final_text = draft_result.text

        if issues:
            revision_result = backend.generate(
                build_revision_prompt(example, draft_result.text, selected_memory, plan_result.text, issues),
                max_tokens=max_tokens,
                temperature=0.0,
                num_ctx=num_ctx,
            )
            calls.append(revision_result)
            final_text = revision_result.text

        return combine_results(final_text, calls), debug
    raise ValueError(f"Unsupported controller: {controller}")


def summarize(run_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    latency_values = [row["latency_ms"] for row in run_rows]
    ttft_values = [row["ttft_ms"] for row in run_rows if row["ttft_ms"] is not None]
    decode_values = [row["decode_tok_per_s"] for row in run_rows if row["decode_tok_per_s"] is not None]
    peak_values = [row["peak_memory_mb"] for row in run_rows if row["peak_memory_mb"] is not None]
    summary = {
        "backend": args.backend,
        "model": args.model,
        "controller": args.controller,
        "num_examples": len(run_rows),
        "avg_quality_score": statistics.fmean(row["quality_score"] for row in run_rows),
        "avg_strict_quality_score": statistics.fmean(row["strict_quality_score"] for row in run_rows),
        "avg_reference_score": statistics.fmean(row["reference_score"] for row in run_rows),
        "avg_preserve_score": statistics.fmean(row["preserve_score"] for row in run_rows),
        "avg_required_score": statistics.fmean(row["required_score"] for row in run_rows),
        "avg_forbidden_score": statistics.fmean(row["forbidden_score"] for row in run_rows),
        "avg_restraint_score": statistics.fmean(row["restraint_score"] for row in run_rows),
        "avg_structure_score": statistics.fmean(row["structure_score"] for row in run_rows),
        "avg_drift_score": statistics.fmean(row["drift_score"] for row in run_rows),
        "avg_constraint_score": statistics.fmean(row["constraint_score"] for row in run_rows),
        "pass_rate": statistics.fmean(row["pass_score"] for row in run_rows),
        "avg_latency_ms": statistics.fmean(latency_values),
        "p95_latency_ms": percentile(latency_values, 0.95),
        "avg_ttft_ms": statistics.fmean(ttft_values) if ttft_values else None,
        "avg_decode_tok_per_s": statistics.fmean(decode_values) if decode_values else None,
        "peak_memory_mb": max(peak_values) if peak_values else None,
        "avg_subcalls": statistics.fmean(row["subcalls"] for row in run_rows),
        "benchmark_path": str(args.benchmark),
    }
    hard_rows = [row for row in run_rows if str(row.get("metadata", {}).get("difficulty", "")).lower() == "hard"]
    summary["hard_num_examples"] = len(hard_rows)
    summary["hard_avg_quality_score"] = statistics.fmean(row["quality_score"] for row in hard_rows) if hard_rows else None
    summary["hard_avg_strict_quality_score"] = (
        statistics.fmean(row["strict_quality_score"] for row in hard_rows) if hard_rows else None
    )
    summary["hard_pass_rate"] = statistics.fmean(row["pass_score"] for row in hard_rows) if hard_rows else None
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=root / "benchmarks" / "grammar_fixer_working.jsonl",
        help="Benchmark JSONL file.",
    )
    parser.add_argument("--backend", choices=["mock", "ollama", "mlx"], default="mock")
    parser.add_argument("--model", default="mistral:latest", help="Model name for the selected backend.")
    parser.add_argument(
        "--controller",
        choices=["direct", "memory_flat", "rlm_adaptive", "rlm_lite", "rlm_recursive"],
        default="direct",
    )
    parser.add_argument("--memory-top-k", type=int, default=2, help="How many memory snippets to surface.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--examples-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "manual")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-keep-alive", default="15m")
    parser.add_argument(
        "--ollama-think",
        default="auto",
        help="Thinking mode for Ollama models: auto, off/on, or low/medium/high when supported.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--mlx-python-bin", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.benchmark.exists():
        raise SystemExit(f"Benchmark file not found: {args.benchmark}")
    examples = load_examples(args.benchmark)
    if args.examples_limit > 0:
        examples = examples[: args.examples_limit]
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    backend = make_backend(args)
    rows: list[dict[str, Any]] = []
    for example in examples:
        result, debug = run_controller(
            backend,
            example,
            controller=args.controller,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            memory_top_k=args.memory_top_k,
        )
        scores = score_example(example, result.text)
        row = {
            "id": example.id,
            "input": example.input,
            "reference": example.reference,
            "output": result.text,
            "metadata": example.metadata,
            "controller": args.controller,
            "latency_ms": result.latency_ms,
            "ttft_ms": result.ttft_ms,
            "decode_tok_per_s": result.decode_tok_per_s,
            "peak_memory_mb": result.peak_memory_mb,
            "subcalls": result.subcalls,
            **scores,
            "debug": debug,
            "backend_details": result.backend_details,
        }
        rows.append(row)

    summary = summarize(rows, args)
    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "examples.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_path: {summary_path}")
    print(f"examples_path: {rows_path}")


if __name__ == "__main__":
    main()
