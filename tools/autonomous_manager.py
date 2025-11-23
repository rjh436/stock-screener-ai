import os
import json
import time
import re
import subprocess
import datetime

EVO_FILE = "optimization/evolution.py"
ENGINE_FILE = "execution/engine.py"
METRICS_FILE = "latest_metrics.json"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🤖 {msg}")
    with open("evolution_journal.txt", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

def modify_hold_time(increase=True):
    with open(EVO_FILE, 'r') as f: content = f.read()
    if increase:
        log("ACTION: Increasing Hold Times (Patience)")
        new_range = "[50, 70, 90, 120]"
    else:
        log("ACTION: Decreasing Hold Times (Responsiveness)")
        new_range = "[20, 30, 45, 60]"
    
    content = re.sub(r"random\.choice\(\[.*?\]\)", f"random.choice({new_range})", content)
    with open(EVO_FILE, 'w') as f: f.write(content)

def modify_position_size(increase=True):
    with open(ENGINE_FILE, 'r') as f: content = f.read()
    match = re.search(r"pos_fraction = (0\.\d+)", content)
    if match:
        current = float(match.group(1))
        new_val = current + 0.05 if increase else max(0.05, current - 0.05)
        log(f"ACTION: Adjusting Position Size: {current:.2f} -> {new_val:.2f}")
        content = content.replace(f"pos_fraction = {current}", f"pos_fraction = {new_val}")
        with open(ENGINE_FILE, 'w') as f: f.write(content)

def run_cycle(i):
    log(f"--- Starting Cycle {i+1}/20 ---")
    
    # Determine python executable
    python_exec = "python3"
    if os.path.exists(".venv/bin/python"):
        python_exec = ".venv/bin/python"
        
    subprocess.run([python_exec, "optimization/optimizer.py"])
    
    if not os.path.exists(METRICS_FILE):
        log("❌ Failure: No metrics found.")
        return

    with open(METRICS_FILE, 'r') as f: stats = json.load(f)
    
    profit = stats.get("avg_profit", 0)
    cagr = stats.get("cagr", 0) * 100
    win_rate = stats.get("win_rate", 0)
    hold = stats.get("hold_days", 0)
    
    log(f"RESULTS: CAGR {cagr:.1f}% | Profit {profit:.2f}% | WR {win_rate:.1f}% | Hold {hold:.1f}d")
    
    # --- AUTONOMOUS DECISION MATRIX ---
    if profit < 2.0:
        log("⚠️  Profit Target Missed (<2.0%). Forcing longer holds.")
        modify_hold_time(increase=True)
    elif cagr < 25.0 and win_rate > 65.0:
        log("⚠️  CAGR Target Missed but Safety High. Increasing Leverage.")
        modify_position_size(increase=True)
    elif win_rate < 55.0:
        log("⚠️  Win Rate Critical. Reducing Leverage for safety.")
        modify_position_size(increase=False)
    else:
        log("✅ Performance Stable. Continuing Evolution.")

if __name__ == "__main__":
    for i in range(20):
        run_cycle(i)
        time.sleep(10)
