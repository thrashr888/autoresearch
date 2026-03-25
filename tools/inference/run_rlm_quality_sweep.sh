#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)

export PYTHONUNBUFFERED=1

exec "$repo_dir/.venv/bin/python" -u "$tools_dir/overnight_grammar.py" \
  --benchmark "$tools_dir/benchmarks/rlm_editor_working.jsonl" \
  --run-root "$tools_dir/runs_rlm_quality" \
  --max-runs 24 \
  --sweep-profile rlm_quality \
  --objective quality_max \
  --max-p95-latency-ms 30000 \
  "$@"
