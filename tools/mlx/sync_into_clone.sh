#!/bin/zsh
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: tools/mlx/sync_into_clone.sh /path/to/autoresearch-mlx" >&2
  exit 1
fi

target_repo="$1"
tools_dir=$(cd -- "$(dirname "$0")" && pwd)

if [ ! -d "$target_repo/.git" ]; then
  echo "Expected a git clone at $target_repo" >&2
  exit 1
fi

mkdir -p "$target_repo/tools/mlx"

cp "$tools_dir/prepare.py" "$target_repo/prepare.py"
cp "$tools_dir/train.py" "$target_repo/train.py"
cp "$tools_dir/overnight_mlx.py" "$target_repo/tools/mlx/overnight_mlx.py"
cp "$tools_dir/run_mlx_sweep.sh" "$target_repo/tools/mlx/run_mlx_sweep.sh"
cp "$tools_dir/launch_mlx_sweep.sh" "$target_repo/tools/mlx/launch_mlx_sweep.sh"

chmod +x "$target_repo/tools/mlx/run_mlx_sweep.sh" "$target_repo/tools/mlx/launch_mlx_sweep.sh"

gitignore="$target_repo/.gitignore"
for pattern in "runs/" "runs_canary/" "launchd_logs/" "*.out" "*.err"; do
  if ! grep -qxF "$pattern" "$gitignore"; then
    printf '%s\n' "$pattern" >> "$gitignore"
  fi
done

echo "Synced MLX tooling into $target_repo"
