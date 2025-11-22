import subprocess
import json
import os
import re
import time

def run_optimizer():
    print("Running Optimizer...")
    # Run the command and wait for it to finish
    subprocess.run(["./Run_Optimizer.command"], check=True)

def read_metrics():
    if not os.path.exists("latest_metrics.json"):
        return None
    with open("latest_metrics.json", "r") as f:
        return json.load(f)

def modify_strategies(increase_time_stop=False):
    config_path = "config/generated_strategies.json"
    if not os.path.exists(config_path):
        print("Config not found!")
        return

    with open(config_path, "r") as f:
        strategies = json.load(f)
    
    modified = False
    if increase_time_stop:
        print("Increasing time_stop by 10 days for all strategies...")
        for strat in strategies:
            current_stop = strat.get("time_stop", 40)
            strat["time_stop"] = current_stop + 10
            modified = True
    
    if modified:
        with open(config_path, "w") as f:
            json.dump(strategies, f, indent=4)
        print("Strategies updated.")

def modify_evolution(decrease_stop_loss=False):
    evo_path = "optimization/evolution.py"
    if not os.path.exists(evo_path):
        print("Evolution script not found!")
        return
    
    with open(evo_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    
    if decrease_stop_loss:
        print("Decreasing stop_loss_atr range in evolution.py...")
        for line in lines:
            if "stop_loss_atr" in line and "random.uniform" in line:
                # Extract current range
                match = re.search(r"random\.uniform\(([\d\.]+),\s*([\d\.]+)\)", line)
                if match:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    # Decrease by 0.5, keep min reasonable
                    new_start = max(2.0, start - 0.5)
                    new_end = max(3.0, end - 0.5)
                    
                    new_line = line.replace(f"random.uniform({start}, {end})", f"random.uniform({new_start}, {new_end})")
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
    
    if modified:
        with open(evo_path, "w") as f:
            f.writelines(new_lines)
        print("Evolution logic updated.")
    else:
        print("No stop_loss_atr range found to update.")

def main():
    for i in range(10):
        print(f"\n=== Auto-Evolver Loop {i+1}/10 ===")
        
        # Step A: Run Optimizer
        try:
            run_optimizer()
        except subprocess.CalledProcessError as e:
            print(f"Optimizer failed with error: {e}")
            continue
        except KeyboardInterrupt:
            print("Stopped by user.")
            break

        # Step B: Read Metrics
        metrics = read_metrics()
        if not metrics:
            print("No metrics found. Skipping analysis.")
            continue
        
        print(f"Metrics: {metrics}")
        
        # Step C: Analyze & Mutate
        avg_profit = metrics.get("avg_profit", 0)
        trades = metrics.get("trades", 0)
        
        if avg_profit < 1.5:
            print(f"Avg Profit {avg_profit:.2f}% < 1.5%. Increasing Time Stops.")
            modify_strategies(increase_time_stop=True)
        
        if trades < 50:
            print(f"Trades {trades} < 50. Decreasing Stop Loss ATR Range.")
            modify_evolution(decrease_stop_loss=True)
            
        # Step D: Status
        print("Status: Optimizing... Restarting Loop.")
        time.sleep(2) # Short pause

if __name__ == "__main__":
    main()
