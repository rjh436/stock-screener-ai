#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

# Memory/process hygiene before each launch.
pkill -f "optimize_superperformance.py|tools/autonomous_tuner.py|run_phase3_validation.py|multiprocessing.spawn|multiprocessing.resource_tracker" || true
sleep 2

# Clear stale optimization state to avoid sticky local minima.
rm -f superperformance_results.csv optimizer_checkpoint_sp.pkl

ts="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/apex_super_autonomous_live_${ts}.log"

export APEX_OBJECTIVE_PROFILE="${APEX_OBJECTIVE_PROFILE:-superperformance}"
export APEX_UNIVERSE="${APEX_UNIVERSE:-RUSSELL3000}"
export APEX_GENERATIONS="${APEX_GENERATIONS:-6}"
export APEX_POPULATION_SIZE="${APEX_POPULATION_SIZE:-40}"
export APEX_MAX_LOOPS="${APEX_MAX_LOOPS:-0}"
export APEX_MAX_WORKERS="${APEX_MAX_WORKERS:-6}"
export APEX_WORKER_CHUNKSIZE="${APEX_WORKER_CHUNKSIZE:-4}"
export APEX_REQUIRE_KNOWN_WINNERS="${APEX_REQUIRE_KNOWN_WINNERS:-1}"
export APEX_KNOWN_WINNER_CACHE_ONLY="${APEX_KNOWN_WINNER_CACHE_ONLY:-1}"
export APEX_KNOWN_WINNER_TIMEOUT_SEC="${APEX_KNOWN_WINNER_TIMEOUT_SEC:-1800}"
export APEX_TARGET_CAGR="${APEX_TARGET_CAGR:-35}"
export APEX_MIN_CAGR_PROMOTE_SUPER="${APEX_MIN_CAGR_PROMOTE_SUPER:-35}"
export APEX_MIN_TRADES_FLOOR="${APEX_MIN_TRADES_FLOOR:-120}"

nohup ./.venv/bin/python -u tools/autonomous_tuner.py </dev/null >"$LOG" 2>&1 &
PID="$!"
sleep 2

if ps -p "$PID" >/dev/null 2>&1; then
  echo "$PID" > logs/apex_super_autonomous.pid
  echo "$LOG" > logs/apex_super_autonomous.logpath
  echo "PID: $PID"
  echo "LOG: $LOG"
  tail -n 20 "$LOG" || true
  exit 0
fi

echo "START_FAILED"
echo "LOG: $LOG"
tail -n 80 "$LOG" || true
exit 1
