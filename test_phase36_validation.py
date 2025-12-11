"""
Phase 3.6 Validation Test - S&P 1500
Tests critical fixes on small dataset before full deployment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.engine import run_backtest
from strategies.strategy_loader import load_strategies
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
import json

print("\n" + "="*70)
print("🔍 PHASE 3.6 VALIDATION TEST (S&P 1500)")
print("="*70)

# Load strategies
config_path = "config/generated_strategies.json"
if not os.path.exists(config_path):
    print("❌ ERROR: config/generated_strategies.json not found!")
    sys.exit(1)

with open(config_path) as f:
    configs = json.load(f)

print(f"\n✅ Loaded {len(configs)} strategy configs")

# Test on 30 stocks from S&P 1500 (fast validation)
print("\n📥 Fetching data for 30 S&P 1500 stocks...")
symbols = get_index_symbols("S&P 1500")[:30]
data = fetch_data_pack(symbols, days=1260)

print(f"✅ Loaded {len([d for d in data.values() if d is not None])} stock datasets")

strategies = load_strategies(configs)
print(f"✅ Loaded {len(strategies)} strategy objects\n")

print("="*70)
print("RUNNING BACKTESTS...")
print("="*70)

results = []

for strat in strategies:
    try:
        print(f"\n🔄 Testing: {strat.name}...")
        result = run_backtest(strat, data)
        
        print(f"  CAGR: {result['cagr']:.1%}")
        print(f"  Win Rate: {result['hit_rate']:.1f}%")
        print(f"  Avg Profit: {result['avg_profit_pct']:.2f}%")
        print(f"  Total Trades: {result['total_trades']}")
        print(f"  Max Drawdown: {result['max_drawdown_pct']:.1f}%")
        
        # Critical check: worst single trade
        worst = 0
        if result['trades_list']:
            worst = min([t['Return%'] for t in result['trades_list']])
            print(f"  Worst Trade: {worst:.2f}%", end=" ")
            
            if worst < -20:
                print("❌ FAIL - Still seeing catastrophic losses!")
                status = "❌"
            elif worst < -15:
                print("⚠️  WARN - Better but needs improvement")
                status = "⚠️"
            else:
                print("✅ PASS - Risk control working!")
                status = "✅"
        else:
            status = "⚪"
            print("  No trades executed")
        
        results.append({
            'strategy': strat.name,
            'cagr': result['cagr'],
            'win_rate': result['hit_rate'],
            'worst_trade': worst,
            'trades': result['total_trades'],
            'status': status
        })
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

for r in results:
    print(f"{r['status']} {r['strategy'][:35]:35} | Worst: {r['worst_trade']:6.2f}% | WR: {r['win_rate']:5.1f}% | Trades: {r['trades']}")

print("\n" + "="*70)
if results:
    worst_overall = min([r['worst_trade'] for r in results if r['worst_trade'] != 0])
    avg_win_rate = sum([r['win_rate'] for r in results]) / len(results)
    
    print(f"Worst Loss Across All Strategies: {worst_overall:.2f}%")
    print(f"Average Win Rate: {avg_win_rate:.1f}%")
    print()
    
    if worst_overall > -15:
        print("✅ VALIDATION PASSED - All fixes working correctly!")
        print("   → Ready for full S&P 1500 backtest")
    else:
        print("⚠️  VALIDATION INCOMPLETE - Review worst trades above")
        print("   → May need additional tuning")
else:
    print("❌ No results - check for errors above")

print("="*70 + "\n")
