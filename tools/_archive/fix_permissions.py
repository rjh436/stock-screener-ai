import os

def fix_file(filename):
    if not os.path.exists(filename):
        print(f"❌ {filename} not found.")
        return

    # Read content
    with open(filename, 'rb') as f:
        content = f.read()

    # Convert CRLF to LF
    content = content.replace(b'\r\n', b'\n')

    # Write back
    with open(filename, 'wb') as f:
        f.write(content)
    
    # Make executable
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | 0o111)
    
    print(f"✅ Fixed line endings and permissions for {filename}")

if __name__ == "__main__":
    fix_file("Run_Screener.command")
    fix_file("Run_Optimizer.command")
