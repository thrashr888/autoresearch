#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)

export PYTHONUNBUFFERED=1

exec "$repo_dir/.venv/bin/python" -u "$tools_dir/overnight_grammar.py" \
  --benchmark "$tools_dir/benchmarks/rlm_editor_working.jsonl" \
  --sweep-profile rlm_focus \
  --objective quality_first \
  --max-p95-latency-ms 15000 \
  "$@"
