#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
logs_dir="$tools_dir/launchd_logs"

mkdir -p "$logs_dir"

usage() {
  cat <<'EOF'
Usage:
  tools/inference/launch_rlm_quality_sweep.sh full
  tools/inference/launch_rlm_quality_sweep.sh stop
  tools/inference/launch_rlm_quality_sweep.sh status
  tools/inference/launch_rlm_quality_sweep.sh tail
EOF
}

job_kind="${1:-}"
label="com.codex.autoresearch.rlm.quality"
stdout_path="$logs_dir/rlm-quality.out"
stderr_path="$logs_dir/rlm-quality.err"
run_root="$tools_dir/runs_rlm_quality"

case "$job_kind" in
  full)
    shift
    cmd=(
      "/usr/bin/caffeinate" "-ims"
      "/bin/zsh" "$tools_dir/run_rlm_quality_sweep.sh"
      "$@"
    )
    ;;
  stop)
    launchctl remove "$label" >/dev/null 2>&1 || true
    echo "Stopped $label"
    exit 0
    ;;
  status)
    launchctl list "$label" || true
    if [ -f "$run_root/latest/status.json" ]; then
      echo "Status file: $run_root/latest/status.json"
    fi
    if [ -L "$run_root/latest/latest.log" ]; then
      echo "Latest log: $run_root/latest/latest.log"
    fi
    exit 0
    ;;
  tail)
    if [ -f "$stdout_path" ]; then
      echo "launchd stdout: $stdout_path"
      tail -n 40 "$stdout_path"
    fi
    if [ -L "$run_root/latest/latest.log" ]; then
      echo
      echo "latest run log: $run_root/latest/latest.log"
      tail -n 40 "$run_root/latest/latest.log"
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
