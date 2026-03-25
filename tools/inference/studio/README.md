# RLM Studio

`RLM Studio` is a small local workbench around the current inference harness.

It gives you:

- a dark-mode browser UI for trying real memory-aware editing examples
- a fast `Quick` mode backed by the promoted fast route
- a slower `Deep` mode backed by the promoted highest-quality route
- memory snippet editing and preserve-term controls
- a simple edited-text-first workflow with advanced details tucked away
- a lightweight request log under `tools/inference/studio/logs/`

## Run

```bash
uv run tools/inference/studio_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

## Notes

- The server reuses `run_grammar_bench.py` directly, so it stays aligned with the research harness.
- `Quick` uses the promoted fast config from `tools/inference/runs_research_claw/latest/promotions.json` when available.
- `Deep` uses the promoted highest-quality config, falling back to the nightly recursive route when needed.
- If you provide a reference string, RLM Studio will also return benchmark-style quality scores for that example.
