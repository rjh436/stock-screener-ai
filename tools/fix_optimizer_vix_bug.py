import os


def fix_optimizer_vix_bug():
    print("🚑 Fixing VIX Ambiguity Bug in Optimizer...")
    
    file_path = "optimization/optimizer.py"
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    with open(file_path, "r") as f:
        content = f.read()
    
    # The problematic line causing the crash
    bad_line = '    vix = g_data.get("$VIX") or g_data.get("VIX")'
    
    # The fix: Check for None explicitly before using the dataframe
    good_block = """    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")"""
        
    if bad_line in content:
        new_content = content.replace(bad_line, good_block)
        with open(file_path, "w") as f:
            f.write(new_content)
        print("   ✅ Optimizer Patched: Fixed DataFrame truth value error.")
    else:
        print("   ⚠️ Could not match exact line. Please check optimization/optimizer.py line 52 manually.")


if __name__ == "__main__":
    fix_optimizer_vix_bug()
