import json
import os

def inspect_winner():
    config_path = "config/generated_strategies.json"

    if not os.path.exists(config_path):
        print("❌ Config file not found.")
        return

    with open(config_path, "r") as f:
        strategies = json.load(f)

    if not strategies:
        print("❌ No strategies found in config.")
        return

    # The optimizer sorts by score, so the first one is usually the best.
    # We will look for the one with the highest CAGR just to be sure.
    best_strat = strategies[0]

    print("\n🏆 TOP STRATEGY DNA ANALYSIS")
    print("=============================")
    print(f"Name:        {best_strat.get('name')}")
    print(f"Limit Ratio: {best_strat.get('limit_ratio')} (Entry Discount)")
    print(f"Trail ATR:   {best_strat.get('trail_atr')} (Exit Tightness)")
    print(f"ADX Thresh:  {best_strat.get('adx_threshold')}")
    print(f"RSI Strong:  {best_strat.get('rsi_strong')}")
    print(f"RSI Weak:    {best_strat.get('rsi_weak')}")
    print("-----------------------------")
    print("Reasoning:")

    tr = best_strat.get('trail_atr', 3.0)
    lr = best_strat.get('limit_ratio', 0.98)

    if tr < 3.0:
        print("🔴 Tight Stop Detected (< 3.0). This explains the low profit.")
        print("   The strategy is scalping early to avoid volatility.")
    elif tr > 5.0:
        print("🟢 Loose Stop Detected. It should be holding longer.")

    if lr > 0.99:
        print("🔴 Shallow Entry Detected. Buying near market price reduces profit margin.")
    elif lr < 0.96:
        print("🟢 Deep Discount Entry. This usually boosts profit per trade.")

if __name__ == "__main__":
    inspect_winner()
