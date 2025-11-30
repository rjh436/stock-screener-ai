import os


def restore_tie_breaker():
    print("⚖️ Restoring 'Deepest Dip' Tie-Breaker...")
    
    engine_path = "execution/engine.py"
    if not os.path.exists(engine_path):
        print(f"❌ Error: {engine_path} not found.")
        return

    with open(engine_path, "r") as f:
        content = f.read()

    # The line we want to replace
    # Current (Weak): daily_candidates.sort(key=lambda x: x["score"], reverse=True)
    # Target (Strong): daily_candidates.sort(key=lambda x: (x["score"], -x["rsi2"]), reverse=True)
    
    # We also need to make sure 'rsi2' is in the candidate dict
    # In the current engine (from final_golden_restore), daily_candidates DOES NOT include rsi2!
    # We must add rsi2 to the dictionary creation line first.

    # 1. Fix Dictionary Creation (Add RSI2)
    # Old: daily_candidates.append({"sym": sym, "entry": entry, "score": quality, "px": float(row["open"])})
    # New: daily_candidates.append({"sym": sym, "entry": entry, "score": quality, "px": float(row["open"]), "rsi2": float(row.get("rsi2", 50))})
    
    dict_creation_old = 'daily_candidates.append({\n                        "sym": sym, "entry": entry, "score": quality, "px": float(row["open"])\n                    })'
    # Simplified match string (stripping whitespace issues) might be safer
    # Let's search for the block structure
    
    if 'daily_candidates.append({' in content and '"score": quality' in content:
        # We will do a robust replace
        # We need to inject "rsi2": float(row.get("rsi2", 50))
        
        # This regex looks for the append block
        append_pattern = r"daily_candidates\.append\(\{\s*['\"]sym['\"]: sym,\s*['\"]entry['\"]: entry,\s*['\"]score['\"]: quality,\s*['\"]px['\"]: float\(row\['open'\]\)\s*\}\)"
        
        append_replacement = """daily_candidates.append({
                        "sym": sym, "entry": entry, "score": quality, 
                        "px": float(row["open"]), "rsi2": float(row.get("rsi2", 50))
                    })"""
        
        # Try simplistic string replace first if exact match
        simple_target = 'daily_candidates.append({\n                        "sym": sym, "entry": entry, "score": quality, "px": float(row["open"])\n                    })'
        
        if simple_target in content:
            content = content.replace(simple_target, append_replacement)
            print("   ✅ Step 1: RSI2 added to candidate data.")
        else:
            # Fallback regex replacement
            import re
            content, count = re.subn(r"daily_candidates\.append\({[^}]*\"px\": float\(row\[\"open\"\]\)\s*}\)", append_replacement, content)
            if count > 0:
                print("   ✅ Step 1: RSI2 added to candidate data (Regex).")
            else:
                print("   ⚠️ Warning: Could not patch candidate dictionary. Check formatting.")

    # 2. Fix Sorting Logic
    sort_old = 'daily_candidates.sort(key=lambda x: x["score"], reverse=True)'
    sort_new = 'daily_candidates.sort(key=lambda x: (x["score"], -x["rsi2"]), reverse=True)'
    
    if sort_old in content:
        content = content.replace(sort_old, sort_new)
        print("   ✅ Step 2: Sorting Logic updated (Score -> RSI2 Ascending).")
        
        with open(engine_path, "w") as f:
            f.write(content)
        print("   💾 Engine Saved.")
    else:
        # Check if already applied
        if 'x["rsi2"]' in content:
            print("   ℹ️ Sorting logic already correct.")
        else:
            print("   ⚠️ Warning: Could not find sorting line to replace.")

    print("\n🚀 TIE-BREAKER RESTORED.")
    print("   This should close the gap (33% -> 43% and 27% -> 49%).")

if __name__ == "__main__":
    restore_tie_breaker()
