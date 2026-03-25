# Inference Autotuning

This directory adapts the `autoresearch` loop to local inference tuning on Apple Silicon.

It now supports two related but separate tracks:

- `Grammar autotuning` - product-focused Ethertext-style cleanup for short edits.
- `RLM editing research` - memory-aware editing where the controller must use saved context, glossary rules, or distractor memories.

## Files

- `benchmarks/grammar_fixer_seed.jsonl` - hand-written seed benchmark for the grammar/text-fixer product track.
- `benchmarks/rlm_editor_seed.jsonl` - memory-heavy seed benchmark for recursive editing experiments.
- `benchmarks/rlm_editor_strict_seed.jsonl` - stricter RLM benchmark with explicit must-keep and must-not-leak checks.
- `benchmarks/private/` - templates and instructions for private real-world examples that stay out of git.
- `prepare_grammar_bench.py` - merges the seed benchmark with optional custom examples.
- `run_grammar_bench.py` - runs one backend/model/config and logs quality + latency metrics.
- `overnight_grammar.py` - autonomous search loop for inference configs.
- `run_grammar_sweep.sh` - unbuffered runner wrapper.
- `launch_grammar_sweep.sh` - launchd wrapper for canary/full unattended runs.
- `studio_server.py` - local browser/API wrapper for trying a fast product default or the current best deep-memory controller on real text.
- `studio/` - static UI assets and request-log docs for the dark-mode `RLM Studio` workbench.
- `research_claw_lite.py` - compact candidate/model comparison runner that promotes the best quick/deep configs.

## Current controllers

- `direct` - single prompt, no memory.
- `memory_flat` - single prompt with all selected memory snippets.
- `rlm_adaptive` - start with `memory_flat`, then escalate to a recursive pass only on harder examples.
- `rlm_lite` - two-stage controller: select memory, summarize editing constraints, then generate the final correction.
- `rlm_recursive` - per-memory note extraction, merged plan, draft generation, and a verification/revision pass when constraints are violated.

`rlm_lite` is intentionally lightweight. It is not a reproduction of the recursive training procedure from the RLM paper; it is an inference-time controller that gives us a concrete research path toward memory-aware local editing.
`rlm_adaptive` is still experimental and is currently meant for manual trials, not the focused unattended RLM sweep.

## Sweep objectives

- Grammar sweeps use the default `balanced` objective: quality must improve without paying a large latency penalty.
- RLM sweeps use a `quality_first` objective with a looser latency cap, because the research question is whether recursive memory handling buys enough quality to justify slower responses.
- `run_rlm_quality_sweep.sh` uses `quality_max` to chase the highest-scoring recursive config under a hard latency cap.
- If Ollama is unavailable, unattended sweeps now stop in a `blocked` state instead of churning failed runs.

## Quality metrics

Each run logs:

- `avg_quality_score` - weighted combination of reference similarity, preserve-term accuracy, and rewrite restraint.
- `avg_strict_quality_score` - `avg_quality_score` with extra penalty when required terms are missing or distractor terms leak in.
- `avg_preserve_score` - how often required terms survive unchanged.
- `avg_required_score` - how often explicitly required phrases still appear.
- `avg_forbidden_score` - how often distractor or forbidden phrases stay out of the output.
- `avg_structure_score` - whether code spans, bullets, and line breaks survive when the benchmark marks them as required.
- `pass_rate` - share of examples that fully satisfy the hard constraints.
- `hard_pass_rate` - the same pass metric, but only on examples tagged `difficulty=hard`.
- `avg_latency_ms` and `p95_latency_ms`
- `avg_ttft_ms` and `avg_decode_tok_per_s` when the backend exposes them
- `peak_memory_mb` when the backend exposes it

For newer local thinking models, `run_grammar_bench.py` now defaults `--ollama-think auto`. That disables thinking for `qwen3*` / `qwen3.5*` models and uses `low` for `gpt-oss*`, which avoids blank benchmark outputs caused by reasoning traces consuming the token budget.

## Quick start

Prepare the grammar benchmark:

```bash
uv run tools/inference/prepare_grammar_bench.py
```

Prepare the RLM benchmark:

```bash
uv run tools/inference/prepare_grammar_bench.py --profile rlm
```

Prepare the stricter RLM benchmark:

```bash
uv run tools/inference/prepare_grammar_bench.py --profile rlm_strict
```

Prepare a grammar benchmark with your own private examples:

```bash
uv run tools/inference/prepare_grammar_bench.py \
  --profile grammar \
  --custom-bench tools/inference/benchmarks/private/ethertext_grammar_private.jsonl
```

Prepare an RLM benchmark with your own private examples:

```bash
uv run tools/inference/prepare_grammar_bench.py \
  --profile rlm \
  --custom-bench tools/inference/benchmarks/private/ethertext_rlm_private.jsonl
```

Run a single mock benchmark:

```bash
uv run tools/inference/run_grammar_bench.py \
  --backend mock \
  --controller direct \
  --benchmark tools/inference/benchmarks/grammar_fixer_working.jsonl
```

Run a single Ollama benchmark:

```bash
uv run tools/inference/run_grammar_bench.py \
  --backend ollama \
  --model mistral:latest \
  --controller rlm_lite \
  --benchmark tools/inference/benchmarks/grammar_fixer_working.jsonl
```

Run a single RLM-focused Ollama benchmark:

```bash
uv run tools/inference/run_grammar_bench.py \
  --backend ollama \
  --model mistral:latest \
  --controller rlm_recursive \
  --benchmark tools/inference/benchmarks/rlm_editor_working.jsonl
```

Run the stricter RLM benchmark:

```bash
uv run tools/inference/run_grammar_bench.py \
  --backend ollama \
  --model mistral:latest \
  --controller rlm_recursive \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl
```

For autonomous strict sweeps, point `overnight_grammar.py` at the strict benchmark and switch the objective metric:

```bash
uv run python tools/inference/overnight_grammar.py \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl \
  --sweep-profile rlm_quality \
  --score-key avg_strict_quality_score
```

For the current Apple Silicon local-quality winner, run the focused `qwen3.5:122b` sweep:

```bash
uv run python tools/inference/overnight_grammar.py \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl \
  --sweep-profile rlm_qwen35_focus \
  --model qwen3.5:122b \
  --score-key avg_strict_quality_score \
  --objective quality_max \
  --max-p95-latency-ms 20000
```

Start a launchd canary:

```bash
zsh tools/inference/launch_grammar_sweep.sh canary --model mistral:latest
```

Start a longer unattended sweep:

```bash
zsh tools/inference/launch_grammar_sweep.sh full --model mistral:latest
```

Start an unattended RLM-focused sweep:

```bash
zsh tools/inference/launch_rlm_sweep.sh full --model mistral:latest
```

Start the pure quality-max overnight sweep:

```bash
zsh tools/inference/launch_rlm_sweep.sh full \
  --run-root tools/inference/runs_rlm_quality \
  --max-runs 24 \
  --sweep-profile rlm_quality \
  --objective quality_max \
  --max-p95-latency-ms 30000 \
  --model mistral:latest
```

Run a compact multi-model comparison batch:

```bash
uv run python tools/inference/research_claw_lite.py \
  --profile rlm \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl \
  --model-profile m5max_local \
  --candidate-set quick \
  --examples-limit 3
```

The built-in `m5max_local` profile is:

- `qwen3:30b`
- `gpt-oss:120b`
- `qwen3.5:122b`

The built-in `m5max_local_extended` profile adds the bigger local challengers:

- `qwen3.5:122b`
- `gpt-oss:120b`
- `llama4:latest`
- `mixtral:8x22b`
- `llama3.1:70b`
- `qwen3:30b`

The built-in `m5max_local_focus` profile keeps only the stronger local RLM-quality shortlist:

- `qwen3.5:122b`
- `mixtral:8x22b`
- `llama3.1:70b`
- `qwen3:30b`

`research_claw_lite.py` now writes a session `results.tsv` plus `progress.png` as it runs. `progress.png` is the objective-score frontier, not raw strict quality, so ties at `1.0` still separate by useful latency/score. A session-local `quality_progress.png` is also written for the raw strict-quality view.

`m5max_local_risky` contains only `deepseek-v2:236b`. Keep that separate unless you explicitly want to probe the memory ceiling.

## Recommended local model matrix (current benchmark takeaways)

- `Quality-first MLX`: `nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx`
- `Speed-first MLX`: `mlx-community/gpt-oss-120b-MXFP4-Q8`
- `Reliable balance MLX`: `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit`

The built-in MLX model profiles are:

- `m5max_mlx_practical`
  - `nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx`
  - `mlx-community/gpt-oss-120b-MXFP4-Q8`
  - `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit`
- `m5max_mlx_quality`
  - Qwen mxfp4, then Llama 4, then gpt-oss Q8
- `m5max_mlx_speed`
  - gpt-oss Q8, then Qwen mxfp4, then Llama 4

Run the current practical MLX shortlist:

```bash
uv run python tools/inference/research_claw_lite.py \
  --backend mlx \
  --model-profile m5max_mlx_practical \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl \
  --profile rlm \
  --candidate-set quick \
  --examples-limit 3
```

For `mlx` runs on this machine, prefer the dedicated uv MLX environment:

```bash
uv run python tools/inference/run_grammar_bench.py \
  --backend mlx \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --mlx-python-bin /Users/thrashr888/.venvs/mlx-bench-qwen/bin/python \
  --controller rlm_recursive \
  --benchmark tools/inference/benchmarks/rlm_editor_strict_working.jsonl
```

Run the product wrapper:

```bash
uv run tools/inference/studio_server.py
```

Then open `http://127.0.0.1:8765`.
