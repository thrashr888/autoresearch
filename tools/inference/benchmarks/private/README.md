# Private Benchmarks

Keep real Ethertext examples here.

The intended workflow is:

1. Start from one of the committed `*.template.jsonl` files in this directory.
2. Create your own private JSONL next to it, for example:
   - `ethertext_grammar_private.jsonl`
   - `ethertext_rlm_private.jsonl`
3. Merge it into a working benchmark with `prepare_grammar_bench.py`.

Examples:

```bash
uv run tools/inference/prepare_grammar_bench.py \
  --profile grammar \
  --custom-bench tools/inference/benchmarks/private/ethertext_grammar_private.jsonl
```

```bash
uv run tools/inference/prepare_grammar_bench.py \
  --profile rlm \
  --custom-bench tools/inference/benchmarks/private/ethertext_rlm_private.jsonl
```

Schema per row:

- `id` - unique stable identifier
- `instruction` - the exact editing instruction
- `input` - raw text the model sees
- `reference` - preferred corrected output
- `memory` - optional list of stored context snippets
- `preserve_terms` - optional exact strings that must survive
- `required_terms` - optional exact strings or phrases that must still appear in the final output
- `forbidden_terms` - optional distractor strings or phrases that must not appear in the final output
- `checks` - optional structural constraints such as `preserve_inline_code`, `preserve_bullets`, `preserve_line_breaks`
- `metadata` - optional tags like `source`, `difficulty`, `category`

`prepare_grammar_bench.py` validates the merged rows and fails on malformed or duplicate IDs.
