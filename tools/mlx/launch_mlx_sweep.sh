#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)
logs_dir="$repo_dir/launchd_logs"

mkdir -p "$logs_dir"

usage() {
  cat <<'EOF'
Usage:
  tools/mlx/launch_mlx_sweep.sh canary
  tools/mlx/launch_mlx_sweep.sh full
  tools/mlx/launch_mlx_sweep.sh stop canary|full
  tools/mlx/launch_mlx_sweep.sh status canary|full
  tools/mlx/launch_mlx_sweep.sh tail canary|full
EOF
}

job_kind="${1:-}"
case "$job_kind" in
  canary)
    label="com.codex.autoresearch.mlx.canary"
    stdout_path="$logs_dir/canary.out"
    stderr_path="$logs_dir/canary.err"
    shift
    cmd=(
      "/usr/bin/caffeinate" "-ims"
      "$tools_dir/run_mlx_sweep.sh"
      "--profile" "safe"
      "--run-root" "runs_canary"
      "--max-runs" "1"
      "--time-budget-seconds" "60"
      "--heartbeat-secs" "10"
      "--quick-eval-interval-secs" "0"
      "$@"
    )
    ;;
  full)
    label="com.codex.autoresearch.mlx.full"
    stdout_path="$logs_dir/full.out"
    stderr_path="$logs_dir/full.err"
    shift
    cmd=(
      "/usr/bin/caffeinate" "-ims"
      "$tools_dir/run_mlx_sweep.sh"
      "--profile" "safe"
      "--run-root" "runs"
      "--max-runs" "36"
      "--time-budget-seconds" "180"
      "--heartbeat-secs" "20"
      "--quick-eval-interval-secs" "0"
      "$@"
    )
    ;;
  stop)
    target="${2:-}"
    case "$target" in
      canary) label="com.codex.autoresearch.mlx.canary" ;;
      full) label="com.codex.autoresearch.mlx.full" ;;
      *) usage; exit 1 ;;
    esac
    launchctl remove "$label" >/dev/null 2>&1 || true
    echo "Stopped $label"
    exit 0
    ;;
  status)
    target="${2:-}"
    case "$target" in
      canary)
        label="com.codex.autoresearch.mlx.canary"
        run_root="runs_canary"
        ;;
      full)
        label="com.codex.autoresearch.mlx.full"
        run_root="runs"
        ;;
      *)
        usage
        exit 1
        ;;
    esac
    launchctl list "$label" || true
    if [ -L "$repo_dir/$run_root/latest.log" ]; then
      echo "Latest log: $repo_dir/$run_root/latest.log"
    fi
    if [ -f "$repo_dir/$run_root/status.json" ]; then
      echo "Status file: $repo_dir/$run_root/status.json"
    fi
    exit 0
    ;;
  tail)
    target="${2:-}"
    case "$target" in
      canary)
        log_path="$logs_dir/canary.out"
        run_root="runs_canary"
        ;;
      full)
        log_path="$logs_dir/full.out"
        run_root="runs"
        ;;
      *)
        usage
        exit 1
        ;;
    esac
    if [ -f "$log_path" ]; then
      echo "launchd stdout: $log_path"
      tail -n 40 "$log_path"
    fi
    if [ -L "$repo_dir/$run_root/latest.log" ]; then
      echo
      echo "latest run log: $repo_dir/$run_root/latest.log"
      tail -n 40 "$repo_dir/$run_root/latest.log"
    fi
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

launchctl remove "$label" >/dev/null 2>&1 || true
launchctl submit -l "$label" -o "$stdout_path" -e "$stderr_path" -- "${cmd[@]}"
echo "Submitted $label"
echo "stdout: $stdout_path"
echo "stderr: $stderr_path"
