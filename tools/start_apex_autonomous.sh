#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

# Memory/process hygiene before each launch.
pkill -f "optimize_superperformance.py|tools/autonomous_walkforward.py|tools/autonomous_tuner.py|run_phase3_validation.py|multiprocessing.spawn|multiprocessing.resource_tracker" || true
sleep 2

ts="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/apex_super_autonomous_live_${ts}.log"

# Rotate optimization state instead of deleting tracked artifacts.
if [[ -f superperformance_results.csv ]]; then
  mv superperformance_results.csv "superperformance_results.pre_autonomous_${ts}.csv" 2>/dev/null || true
fi
if [[ -f optimizer_checkpoint_sp.pkl ]]; then
  mv optimizer_checkpoint_sp.pkl "optimizer_checkpoint_sp.pre_autonomous_${ts}.pkl" 2>/dev/null || true
fi

export APEX_UNIVERSE="${APEX_UNIVERSE:-RUSSELL3000}"
export APEX_WF_TRAIN_START="${APEX_WF_TRAIN_START:-2006-02-16}"
export APEX_WF_TRAIN_END="${APEX_WF_TRAIN_END:-2018-12-31}"
export APEX_WF_OOS1_START="${APEX_WF_OOS1_START:-2019-01-01}"
export APEX_WF_OOS1_END="${APEX_WF_OOS1_END:-2021-12-31}"
export APEX_WF_OOS2_START="${APEX_WF_OOS2_START:-2022-01-01}"
export APEX_GENERATIONS="${APEX_GENERATIONS:-6}"
export APEX_POPULATION_SIZE="${APEX_POPULATION_SIZE:-20}"
export APEX_MAX_WORKERS="${APEX_MAX_WORKERS:-10}"
export APEX_MAX_WORKERS_HARD_CAP="${APEX_MAX_WORKERS_HARD_CAP:-6}"
export APEX_MP_START_METHOD="${APEX_MP_START_METHOD:-fork}"
export APEX_PRUNE_PREPARED_DF="${APEX_PRUNE_PREPARED_DF:-1}"
export APEX_MEMORY_UTILIZATION="${APEX_MEMORY_UTILIZATION:-0.72}"
export APEX_MEMORY_HEADROOM_GB="${APEX_MEMORY_HEADROOM_GB:-6}"
export APEX_POOL_MAX_TASKS_PER_CHILD="${APEX_POOL_MAX_TASKS_PER_CHILD:-4}"
export APEX_MARKET_EXPOSURE_MODE="${APEX_MARKET_EXPOSURE_MODE:-exposure}"
export APEX_WF_MAX_LOOPS="${APEX_WF_MAX_LOOPS:-4}"
export APEX_WF_NO_IMPROVE_STOP="${APEX_WF_NO_IMPROVE_STOP:-2}"
export APEX_WF_REQUIRE_KNOWN_WINNERS="${APEX_WF_REQUIRE_KNOWN_WINNERS:-1}"
export APEX_WF_KNOWN_MIN_PF="${APEX_WF_KNOWN_MIN_PF:-1.25}"
export APEX_WF_KNOWN_PF_REQUIRED="${APEX_WF_KNOWN_PF_REQUIRED:-2}"
export APEX_WF_OOS_MAX_DD="${APEX_WF_OOS_MAX_DD:-35}"
export APEX_WF_OOS_MIN_PF="${APEX_WF_OOS_MIN_PF:-1.20}"
export APEX_WF_OOS1_MIN_CAGR="${APEX_WF_OOS1_MIN_CAGR:-8}"
export APEX_WF_OOS2_MIN_CAGR="${APEX_WF_OOS2_MIN_CAGR:-15}"

nohup ./.venv/bin/python -u tools/autonomous_walkforward.py </dev/null >"$LOG" 2>&1 &
PID="$!"
disown "$PID" 2>/dev/null || true
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
