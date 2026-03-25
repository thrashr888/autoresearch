# autoresearch

This repo is now centered on inference research for local editing workflows.

The active story is:
- grammar autotuning
- RLM editing research
- product case curation / mining
- Studio as the wrapper over the whole stack

Older training-oriented paths were removed from the repo to simplify focus.

## Active paths

1. Grammar autotuning
- Purpose: short Ethertext-style cleanup/edit benchmarks
- Key files:
  - `tools/inference/prepare_grammar_bench.py`
  - `tools/inference/run_grammar_bench.py`
  - `tools/inference/overnight_grammar.py`

2. RLM editing research
- Purpose: memory-aware editing with glossary/context/distractor constraints
- Includes stricter recursive-controller experiments and Research Claw Lite comparisons
- Key files:
  - `tools/inference/research_claw_lite.py`
  - `tools/inference/analyze_rlm_sweep.py`
  - benchmark files under `tools/inference/benchmarks/`

3. Product case curation / mining
- Purpose: curate real examples, mine failures, and turn them into better benchmarks
- Key files:
  - `tools/inference/curate_product_cases.py`
  - `tools/inference/mine_failures.py`
  - tests under `tests/`

4. Studio wrapper
- Purpose: try the current best product/deep-memory configs on real text
- Key files:
  - `tools/inference/studio_server.py`
  - `tools/inference/studio/`

In short: the repo is functionally centered on `tools/inference/`.

## Last major thing

The last major expansion here was Research Claw Lite in `tools/inference/`, i.e. the compact candidate/model comparison and promotion workflow for inference experiments.

## Inference autotuning quick start

Install dependencies:

```bash
uv sync
```

Prepare the grammar benchmark:

```bash
uv run tools/inference/prepare_grammar_bench.py
```

Run a single benchmark:

```bash
uv run tools/inference/run_grammar_bench.py \
  --backend ollama \
  --model mistral:latest \
  --controller rlm_lite \
  --benchmark tools/inference/benchmarks/grammar_fixer_working.jsonl
```

Start Studio:

```bash
uv run tools/inference/studio_server.py
```

For deeper usage, see:
- `tools/inference/README.md`

## Repo shape

```text
README.md
pyproject.toml
uv.lock
tests/
tools/inference/
```

## Notes

- MLX-heavy local runs should be done conservatively.
- Run one heavy local model at a time.
- The main product/research surface is now inference, not training.

## License

MIT
