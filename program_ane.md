# autoresearch (ANE fork)

This is the Apple Neural Engine version of the autoresearch loop.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read the in-scope files**:
   - `README.md`
   - `prepare.py`
   - `prepare_ane.py`
   - `train_ane.py`
4. **Verify the base cache exists**: make sure `~/.cache/autoresearch/` contains parquet shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Verify the ANE token cache exists**: make sure `~/.cache/autoresearch/ane/manifest.json` exists. If not, run `uv run prepare_ane.py`.
6. **Verify the ANE repo exists**: `ANE_HOME` should point at a checkout of `https://github.com/maderix/ANE`, or the repo should exist at `../ANE` or `/tmp/ANE`.
7. **Initialize results.tsv**: create `results.tsv` with the usual header if it doesn't exist yet.

## Experimentation

Each experiment runs by launching:

```bash
uv run train_ane.py
```

By default this calibrates the current model for a fixed 300-second training budget and then launches the real run. Use `--steps` only for quick debugging.

The ANE fork intentionally differs from upstream autoresearch:

- Training is done by the external ANE Objective-C trainer.
- Validation is done in Python from the saved checkpoint.
- `val_bpb` is an **ANE-local metric** using a shorter sequence length and a much smaller validation token budget so runs stay practical on a Mac. Compare ANE results only against other ANE results from this fork.

**What you CAN do:**

- Modify `train_ane.py`.
- Change architecture, optimizer knobs, step budget, evaluation budget, or ANE compilation parameters.

**What you CANNOT do:**

- Modify `prepare.py` unless the human explicitly asks for it.
- Change the ANE repo in place unless the human explicitly asks for that repo to become part of the search space.

## Output format

`train_ane.py` prints the same summary block shape as upstream:

```text
---
val_bpb:          1.234567
training_seconds: 300.0
total_seconds:    345.0
peak_vram_mb:     512.0
mfu_percent:      0.00
total_tokens_M:   0.6
num_steps:        2500
num_params_M:     22.3
depth:            8
```

`peak_vram_mb` in the ANE fork is process peak resident memory, not CUDA VRAM.

## Logging results

Use the same `results.tsv` schema as upstream:

```text
commit	val_bpb	memory_gb	status	description
```

## The loop

1. Inspect git state.
2. Hack `train_ane.py`.
3. Commit.
4. Run `uv run train_ane.py > run.log 2>&1`.
5. Extract `val_bpb` and `peak_vram_mb` from `run.log`.
6. If the run crashes, inspect the last 50 lines of `run.log`, fix obvious issues, and retry.
7. Record the result in `results.tsv`.
8. Keep the commit only if `val_bpb` improved within this ANE fork.
