import os


def fix_schwab_quote_call():
    print("🔧 FIXING SCHWAB CLIENT (Removing List Brackets from Quote Call)...")
    
    file_path = "data/schwab_client.py"
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    with open(file_path, "r") as f:
        content = f.read()
    
    # The bad line causing the ['NWL'] error
    bad_line = "r = self._cli.get_quote([symbol])"
    
    # The correct line (passing string directly)
    good_line = "r = self._cli.get_quote(symbol)"
    
    if bad_line in content:
        new_content = content.replace(bad_line, good_line)
        with open(file_path, "w") as f:
            f.write(new_content)
        print("   ✅ Schwab Client Patched: get_quote(symbol) now passes a string.")
    else:
        # Fallback check if indentation varies
        if "get_quote([symbol])" in content:
            new_content = content.replace("get_quote([symbol])", "get_quote(symbol)")
            with open(file_path, "w") as f:
                f.write(new_content)
            print("   ✅ Schwab Client Patched (Loose Match): get_quote(symbol) updated.")
        else:
            print("   ⚠️ Could not locate the specific line 'r = self._cli.get_quote([symbol])'. Please check manually.")


if __name__ == "__main__":
    fix_schwab_quote_call()
