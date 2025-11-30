import os
import re


def configure_optimization_goals():
    print("🎯 CONFIGURING OPTIMIZER GOALS...")

    # --- 1. UPGRADE THE JUDGE (Optimizer) ---
    opt_path = "optimization/optimizer.py"
    
    new_fitness_logic = """def calculate_fitness(result):
    # SMART FITNESS FUNCTION (Role-Based Objectives)
    name = result.get("strategy", "")
    cagr = result.get("cagr", 0)
    avg_profit = result.get("avg_profit_pct", 0)
    hit_rate = result.get("hit_rate", 0)
    trades = result.get("total_trades", 0)

    # 1. Global Sanity Check (Dead Strategies)
    if trades < 10: return -1e6

    # 2. MACHINE GUN GOAL: Avg Profit > 5%
    if "MachineGun" in name:
        # Hard Constraint: Must make real money per trade
        if avg_profit < 5.0:
            # Penalty: The further from 5%, the worse the score
            return -5000.0 + (avg_profit * 100) 
        
        # If Constraint Met: Optimize for Total Return (CAGR)
        # Bonus for maintaining high activity (>100 trades)
        activity_bonus = min(trades, 200) / 2.0
        return (cagr * 5000) + activity_bonus

    # 3. SNIPER GOAL: High Hit Rate (Without sacrificing Power)
    if "Sniper" in name or "Gen12" in name:
        # Safety Floor: Don't break the Wealth Building (40% CAGR / 15% Profit)
        if cagr < 0.40 or avg_profit < 15.0:
            return -5000.0
            
        # If Floor Met: Maximize Precision (Hit Rate)
        # We weight Hit Rate heavily to push it up from 57%
        return (hit_rate * 100) + (cagr * 1000)

    # Fallback for generic strategies
    return result.get("Score", 0)"""

    if os.path.exists(opt_path):
        with open(opt_path, "r") as f:
            content = f.read()
        
        start_idx = content.find("def calculate_fitness(result):")
        if start_idx != -1:
            next_def = content.find("def load_optimization_data", start_idx)
            pre = content[:start_idx]
            post = content[next_def:] if next_def != -1 else ""
            with open(opt_path, "w") as f:
                f.write(pre + new_fitness_logic + "\n\n" + post)
            print("   ✅ Optimizer Updated: Goals Injected (MG > 5% Profit, Sniper > Hit Rate).")
        else:
            print("   ⚠️ Could not find 'calculate_fitness' to replace.")
    else:
        print("   ❌ optimizer.py not found.")

    # --- 2. UPGRADE THE MUTATION (Evolution) ---
    evo_path = "optimization/evolution.py"
    if os.path.exists(evo_path):
        with open(evo_path, "r") as f:
            evo_content = f.read()
            
        if "1.15" in evo_content:
            new_evo_content = evo_content.replace("1.15", "1.25")
            with open(evo_path, "w") as f:
                f.write(new_evo_content)
            print("   ✅ Evolution Updated: Profit Targets allowed up to 25%.")
        else:
            print("   ℹ️ Evolution range already updated or not found.")
    else:
        print("   ❌ evolution.py not found.")

    print("\n🚀 READY. Run './Run_Optimizer.command' to evolve toward these goals.")


if __name__ == "__main__":
    configure_optimization_goals()
