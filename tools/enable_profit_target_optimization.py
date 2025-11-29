import os
import re

def enable_profit_optimization():
    print("🧬 Upgrading Evolution Engine to tune Profit Targets...")
    
    evo_path = "optimization/evolution.py"
    if not os.path.exists(evo_path):
        print(f"❌ Error: {evo_path} not found.")
        return

    with open(evo_path, "r") as f:
        content = f.read()

    # Mutation logic that includes Profit Target tuning
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
                        # Mutate between 4% and 15%
                        current = float(rule.get(\"val\", 1.06))
                        # Small nudge or random reset
                        if random.random() < 0.5:
                            new_val = current + random.choice([-0.01, 0.01])
                        else:
                            new_val = 1.0 + (random.randint(4, 15) / 100.0)
                        rule[\"val\"] = round(max(1.04, min(1.15, new_val)), 2)
                
                # If no profit target exists, add one occasionally
                if not found_target and random.random() < 0.3:
                     mutant[\"exit_rules\"].append({
                        \"type\": \"profit_target\", 
                        \"val\": round(1.0 + (random.randint(4, 12) / 100.0), 2)
                     })

        elif r < 0.6: 
            # Mutate Entry Rules (Replace one)
            if mutant[\"entry_rules\"]:
                idx = random.randint(0, len(mutant[\"entry_rules\"])-1)
                mutant[\"entry_rules\"][idx] = self._random_rule()
        elif r < 0.8:
            # Add a new rule (Constraint tightening)
            if len(mutant[\"entry_rules\"]) < 5:
                mutant[\"entry_rules\"].append(self._random_rule())
        else:
            # Mutate Stop Loss
            mutant[\"stop_loss_atr\"] = round(random.uniform(2.0, 5.0), 1)
            
        return mutant"""

    # Replace the existing mutate method
    pattern = r"def mutate\(self, genome: Dict\) -> Dict:[\s\S]*?return mutant"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, mutation_logic, content)
        with open(evo_path, "w") as f:
            f.write(new_content)
        print("✅ Evolution Engine Upgraded: Can now optimize Profit Targets.")
    else:
        print("⚠️ Could not auto-patch evolution.py. Structure may differ.")

if __name__ == "__main__":
    enable_profit_optimization()
