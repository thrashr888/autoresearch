#!/bin/zsh
set -euo pipefail

tools_dir=$(cd -- "$(dirname "$0")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)
logs_dir="$tools_dir/launchd_logs"

mkdir -p "$logs_dir"

usage() {
  cat <<'EOF'
Usage:
  tools/inference/launch_grammar_sweep.sh canary
  tools/inference/launch_grammar_sweep.sh full
  tools/inference/launch_grammar_sweep.sh stop canary|full
  tools/inference/launch_grammar_sweep.sh status canary|full
  tools/inference/launch_grammar_sweep.sh tail canary|full
EOF
}

job_kind="${1:-}"
case "$job_kind" in
  canary)
    label="com.codex.autoresearch.grammar.canary"
    stdout_path="$logs_dir/canary.out"
    stderr_path="$logs_dir/canary.err"
    shift
    cmd=(
      "/usr/bin/caffeinate" "-ims"
      "$tools_dir/run_grammar_sweep.sh"
      "--run-root" "$tools_dir/runs_canary"
      "--max-runs" "3"
      "$@"
    )
    ;;
  full)
    label="com.codex.autoresearch.grammar.full"
    stdout_path="$logs_dir/full.out"
    stderr_path="$logs_dir/full.err"
    shift
    cmd=(
      "/usr/bin/caffeinate" "-ims"
      "$tools_dir/run_grammar_sweep.sh"
      "--run-root" "$tools_dir/runs"
      "--max-runs" "18"
      "$@"
    )
    ;;
  stop)
    target="${2:-}"
    case "$target" in
      canary) label="com.codex.autoresearch.grammar.canary" ;;
      full) label="com.codex.autoresearch.grammar.full" ;;
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
        label="com.codex.autoresearch.grammar.canary"
        run_root="$tools_dir/runs_canary"
        ;;
      full)
        label="com.codex.autoresearch.grammar.full"
        run_root="$tools_dir/runs"
        ;;
      *)
        usage
        exit 1
        ;;
    esac
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
    target="${2:-}"
    case "$target" in
      canary)
        log_path="$logs_dir/canary.out"
        run_root="$tools_dir/runs_canary"
        ;;
      full)
        log_path="$logs_dir/full.out"
        run_root="$tools_dir/runs"
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
