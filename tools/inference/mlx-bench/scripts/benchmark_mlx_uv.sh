#!/bin/bash
set -euo pipefail
exec /Users/thrashr888/.venvs/mlx-bench-qwen/bin/python /Users/thrashr888/Workspace/autoresearch/tools/inference/mlx-bench/scripts/benchmark_mlx.py "$@"
