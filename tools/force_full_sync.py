import json
import os
import re

def force_full_sync():
    print("🔄 STARTING MASTER SYNC: Enforcing 'Precision Build' State...")

    # --- 1. CONFIGURATION SYNC (Strategies) ---
    config_path = "config/generated_strategies.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            strategies = json.load(f)
        
        modified_config = False
        for strat in strategies:
            if strat.get("name") == "Strategy_Apex_Alpha_MachineGun":
                # A. Fix Time Stop
                if strat.get("time_stop") != 10:
                    strat["time_stop"] = 10
                    print("   ✅ Config: Time Stop forced to 10.")
                    modified_config = True
                
                # B. Remove SMA200 Filter
                orig_len = len(strat["entry_rules"])
                strat["entry_rules"] = [r for r in strat["entry_rules"] if not (r.get("col") == "close" and r.get("ref") == "sma200")]
                if len(strat["entry_rules"]) < orig_len:
                    print("   ✅ Config: SMA200 Filter removed.")
                    modified_config = True

                # C. Ensure Profit Target
                has_target = any(r.get("type") == "profit_target" for r in strat.get("exit_rules", []))
                if not has_target:
                    strat["exit_rules"].append({"type": "profit_target", "val": 1.06})
                    print("   ✅ Config: 6% Profit Target injected.")
                    modified_config = True
        
        if modified_config:
            with open(config_path, "w") as f:
                json.dump(strategies, f, indent=4)
            print("   💾 Strategies.json synced.")
        else:
            print("   INFO: Strategies.json already correct.")

    # --- 2. ENGINE SYNC (Ranking Logic) ---
    engine_path = "execution/engine.py"
    if os.path.exists(engine_path):
        with open(engine_path, "r") as f:
            content = f.read()
        
        # We need to verify if the 'Velocity Booster' logic exists
        if "Velocity Booster" not in content:
            # We must replace the old ranking logic with the new one
            # Targeted replacement for the 'Machine Gun' else block
            pattern = r"(else:\s+rsi2 = row\.get\(\"rsi2\", 50\).*?if vol_rel > 1\.5: score \+= 10)"
            
            new_logic = """else: 
        # MACHINE GUN MODE: Deep Dips + High Volatility (Velocity)
        rsi2 = row.get(\"rsi2\", 50)
        score += (100 - rsi2) * 2.0  # Base: How deep is the dip?
        
        # NEW: Velocity Booster (Prioritize High Volatility Stocks)
        close_px = row.get(\"close\", 1.0)
        if close_px > 0:
            atr_pct = (row.get(\"atr14\", 0) / close_px) * 100
            if atr_pct > 3.0: score += 15  # High Velocity
            elif atr_pct > 2.0: score += 5
            
        # NEW: Smart Trend Filter (Soft Filter)
        if row.get(\"close\", 0) > row.get(\"sma200\", 999999):
            score += 20
        
        vol_rel = row.get(\"volume\", 0) / (row.get(\"vol_ma20\", 1) + 1)
        if vol_rel > 1.5: score += 10"""
            
            # Use dotall to match across newlines
            new_content = re.sub(pattern, new_logic, content, flags=re.DOTALL)
            
            if new_content != content:
                with open(engine_path, "w") as f:
                    f.write(new_content)
                print("   ✅ Engine.py synced (Velocity + Soft Trend logic added).")
            else:
                print("   ⚠️ Engine.py regex match failed. Please check manually.")
        else:
            print("   INFO: Engine.py already correct.")

    # --- 3. EVOLUTION SYNC (Profit Optimization) ---
    evo_path = "optimization/evolution.py"
    if os.path.exists(evo_path):
        with open(evo_path, "r") as f:
            content = f.read()
            
        if "Mutate Profit Target" not in content:
            # Inject new mutate method
            mutation_logic = """    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant[\"name\"] = mutant[\"name\"] + \"_mut\"
        r = random.random()
        
        if r < 0.2:
            # Mutate Time Stop
            mutant[\"time_stop\"] = random.choice([5, 8, 10, 12, 15, 20])
        elif r < 0.4:
            # Mutate Profit Target (NEW)
            if \"exit_rules\" in mutant:
                found_target = False
                for rule in mutant[\"exit_rules\"]:
                    if rule.get(\"type\") == \"profit_target\":
                        found_target = True
                        current = float(rule.get(\"val\", 1.06))
                        if random.random() < 0.5:
                            new_val = current + random.choice([-0.01, 0.01])
                        else:
                            new_val = 1.0 + (random.randint(4, 15) / 100.0)
                        rule[\"val\"] = round(max(1.04, min(1.15, new_val)), 2)
                
                if not found_target and random.random() < 0.3:
                     mutant[\"exit_rules\"].append({"type": "profit_target", "val": round(1.0 + (random.randint(4, 12) / 100.0), 2)})

        elif r < 0.6: 
            if mutant[\"entry_rules\"]:
                idx = random.randint(0, len(mutant[\"entry_rules\"])-1)
                mutant[\"entry_rules\"][idx] = self._random_rule()
        elif r < 0.8:
            if len(mutant[\"entry_rules\"]) < 5:
                mutant[\"entry_rules\"].append(self._random_rule())
        else:
            mutant[\"stop_loss_atr\"] = round(random.uniform(2.0, 5.0), 1)
            
        return mutant"""

            pattern = r"def mutate\(self, genome: Dict\) -> Dict:[\s\S]*?return mutant"
            new_content = re.sub(pattern, mutation_logic, content)
            
            with open(evo_path, "w") as f:
                f.write(new_content)
            print("   ✅ Evolution.py synced (Profit Target Mutation enabled).")
        else:
            print("   INFO: Evolution.py already correct.")

    print("\n✅ MASTER SYNC COMPLETE. System is ready for Optimization.")

if __name__ == "__main__":
    force_full_sync()
