#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)

export PYTHONUNBUFFERED=1

exec "$repo_dir/.venv/bin/python" -u "$tools_dir/overnight_mlx.py" "$@"
