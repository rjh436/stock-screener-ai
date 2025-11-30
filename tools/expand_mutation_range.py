import os
import re


def expand_mutation_range():
    print("⏳ EXPANDING MUTATION HORIZON (Giving strategies room to breathe)...")
    
    evo_path = "optimization/evolution.py"
    if not os.path.exists(evo_path):
        print(f"❌ Error: {evo_path} not found.")
        return

    with open(evo_path, "r") as f:
        content = f.read()

    # We want to replace the restrictive time_stop list with a wider one
    # Current: random.choice([5, 8, 10, 12, 15, 20])
    # New:     random.choice([10, 15, 20, 25, 30, 35, 40, 45])
    
    old_range_pattern = r"random\.choice\(\[5, 8, 10, 12, 15, 20\]\)"
    new_range = "random.choice([10, 15, 20, 25, 30, 35, 40, 45])"
    
    if re.search(old_range_pattern, content):
        new_content = re.sub(old_range_pattern, new_range, content)
        with open(evo_path, "w") as f:
            f.write(new_content)
        print("   ✅ Evolution Updated: Time Stops can now mutate up to 45 days.")
    else:
        # Fallback search if the list isn't exact
        print("   ⚠️ Exact list match failed. Attempting generic list replacement...")
        # Look for any random.choice assigning to time_stop
        generic_pattern = r"mutant\[\"time_stop\"\] = random\.choice\(\[.*?\]\)"
        replacement = f'mutant["time_stop"] = {new_range}'
        
        if re.search(generic_pattern, content):
            new_content = re.sub(generic_pattern, replacement, content)
            with open(evo_path, "w") as f:
                f.write(new_content)
            print("   ✅ Evolution Updated (Generic Match): Time Stops expanded.")
        else:
            print("   ❌ Could not locate time_stop mutation logic.")


if __name__ == "__main__":
    expand_mutation_range()
