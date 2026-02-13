#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
STATUS_LOG="$ROOT/logs/apex_watchdog.log"
START_SCRIPT="$ROOT/tools/start_apex_autonomous.sh"
GOLDEN_FILE="$ROOT/config/GOLDEN_GENOME.json"

INTERVAL_SEC="${APEX_WATCH_INTERVAL_SEC:-60}"
STALE_SEC="${APEX_WATCH_STALE_SEC:-1800}"          # 30 minutes
HARD_STALE_SEC="${APEX_WATCH_HARD_STALE_SEC:-3600}" # 60 minutes

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$STATUS_LOG"
}

cleanup_python() {
  pkill -f "optimize_superperformance.py|tools/autonomous_walkforward.py|tools/autonomous_tuner.py|run_phase3_validation.py|multiprocessing.spawn|multiprocessing.resource_tracker" || true
  sleep 2
  # Kill any remaining project-scoped python workers.
  pids="$(ps ax -o pid,command | awk '/Gemini Learning Stock Screener/ && /python/ {print $1}' | tr '\n' ' ')"
  if [[ -n "${pids// }" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids || true
  fi
}

start_run() {
  log "Starting autonomous tuner run"
  if "$START_SCRIPT" >>"$STATUS_LOG" 2>&1; then
    log "Start command completed"
  else
    log "Start command failed"
  fi
}

log "Watchdog started | interval=${INTERVAL_SEC}s stale=${STALE_SEC}s hard_stale=${HARD_STALE_SEC}s"

while true; do
  if [[ -s "$GOLDEN_FILE" ]]; then
    log "Golden genome detected at $GOLDEN_FILE. Holding state (no restart)."
    sleep "$INTERVAL_SEC"
    continue
  fi

  pid=""
  if [[ -f "$ROOT/logs/apex_super_autonomous.pid" ]]; then
    pid="$(tr -d '[:space:]' < "$ROOT/logs/apex_super_autonomous.pid")"
  fi

  latest_log="$(ls -t "$ROOT"/logs/apex_super_autonomous_live_*.log 2>/dev/null | head -n1 || true)"
  now_epoch="$(date +%s)"
  log_age=999999
  if [[ -n "$latest_log" ]]; then
    mtime="$(stat -f %m "$latest_log" 2>/dev/null || echo 0)"
    if [[ "$mtime" -gt 0 ]]; then
      log_age=$((now_epoch - mtime))
    fi
  fi

  alive=0
  if [[ -n "$pid" ]] && [[ "$pid" != "0" ]] && ps -p "$pid" >/dev/null 2>&1; then
    alive=1
  fi

  stat_field=""
  if [[ "$alive" -eq 1 ]]; then
    stat_field="$(ps -o stat= -p "$pid" | tr -d ' ' || true)"
  fi

  worker_count="$(ps -ax -o command | awk '/multiprocessing.spawn/ && /pipe_handle/ {c++} END{print c+0}')"
  worker_cpu_sum="$(ps -ax -o %cpu,command | awk '/multiprocessing.spawn/ && /pipe_handle/ {s+=$1} END{printf "%.1f", s+0}')"

  if [[ "$alive" -eq 0 ]]; then
    log "No active tuner pid found (pid='${pid:-none}'). Restarting."
    cleanup_python
    start_run
    sleep "$INTERVAL_SEC"
    continue
  fi

  if [[ "$stat_field" == *T* ]]; then
    log "Detected stopped process state (stat=${stat_field}). Restarting."
    cleanup_python
    start_run
    sleep "$INTERVAL_SEC"
    continue
  fi

  if [[ "$log_age" -gt "$HARD_STALE_SEC" ]]; then
    log "Hard stale log age ${log_age}s > ${HARD_STALE_SEC}s. Restarting."
    cleanup_python
    start_run
    sleep "$INTERVAL_SEC"
    continue
  fi

  if [[ "$log_age" -gt "$STALE_SEC" ]] && [[ "$worker_count" -eq 0 ]]; then
    log "Stale log age ${log_age}s and no workers. Restarting."
    cleanup_python
    start_run
    sleep "$INTERVAL_SEC"
    continue
  fi

  log "Healthy | pid=${pid} stat=${stat_field} workers=${worker_count} cpu_sum=${worker_cpu_sum}% log_age=${log_age}s"
  sleep "$INTERVAL_SEC"
done
