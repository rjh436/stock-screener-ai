import os
import json
import time
import re
import subprocess

EVO_FILE = "optimization/evolution.py"
GEN_FILE = "config/generated_strategies.json"
METRICS_FILE = "latest_metrics.json"

def run_cycle(i):
    print(f"\n🤖 MANAGER: Starting Optimization Cycle {i+1}")
    
    # Determine python executable
    python_exec = "python3"
    if os.path.exists(".venv/bin/python"):
        python_exec = ".venv/bin/python"
        
    subprocess.run([python_exec, "optimization/optimizer.py"])
    
    if not os.path.exists(METRICS_FILE):
        print("❌ No metrics generated.")
        return

    with open(METRICS_FILE, 'r') as f:
        stats = json.load(f)
        
    profit = stats.get("avg_profit", 0)
    win_rate = stats.get("win_rate", 0)
    trades = stats.get("trades", 0)
    
    print(f"📊 REPORT: Profit {profit:.2f}% | WR {win_rate:.1f}% | Trades {trades}")
    
    # Autonomous Decision Logic
    with open(EVO_FILE, 'r') as f: evo_code = f.read()
    
    if profit < 2.0:
        print("⚠️  Profit Low. Action: Extending Hold Times.")
        evo_code = re.sub(r"time_stop\": random.choice\(\[.*?\]\)", 
                          "time_stop\": random.choice([50, 70, 90, 120])", evo_code)
    
    elif win_rate < 60.0:
        print("⚠️  Win Rate Low. Action: Tightening Stops.")
        evo_code = re.sub(r"stop_loss_atr\": round\(random.uniform\(.*?\)","stop_loss_atr\": round(random.uniform(4.0, 6.0)", evo_code)
        
    with open(EVO_FILE, 'w') as f: f.write(evo_code)

if __name__ == "__main__":
    # Reset Strategy Pool to clean out the "Glitch" strategies
    print("🧹 Cleaning Strategy Pool...")
    with open(GEN_FILE, 'w') as f: f.write("[]") # Empty list to force re-seed
    
    for i in range(10): # Run 10 full cycles
        run_cycle(i)
