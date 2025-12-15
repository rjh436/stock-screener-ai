import json
import os
import time
import datetime


def verify_state():
    config_path = "config/generated_strategies.json"
    metrics_path = "latest_metrics.json"
    
    print("🔍 SYSTEM STATE VERIFICATION")
    print("============================")

    # 1. Check Config File Status
    if os.path.exists(config_path):
        mtime = os.path.getmtime(config_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"📂 Config File: FOUND")
        print(f"🕒 Last Modified: {dt} ({time.time() - mtime:.1f}s ago)")
        
        try:
            with open(config_path, "r") as f:
                strategies = json.load(f)
            
            print(f"\n📋 Active Strategies ({len(strategies)}):")
            gen14_found = False
            for s in strategies:
                name = s.get("name", "Unknown")
                print(f"   - {name}")
                if "Wealth" in name and "Gen 14" in name:
                    gen14_found = True
            
            print("\n✅ STATUS CHECK:")
            if gen14_found:
                print("   PASS: Apex Wealth (Gen 14) is INSTALLED.")
            else:
                print("   FAIL: Apex Wealth (Gen 14) is MISSING.")
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print("❌ Config File: MISSING")

    # 2. Check Latest Metrics
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            print("\n📊 Latest Run Metrics:")
            print(f"   CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"   Trades: {metrics.get('trades', 0)}")
        except:
            print("\n⚠️ Could not read latest metrics.")
    else:
        print("\n⚠️ No recent metrics found.")


if __name__ == "__main__":
    verify_state()
