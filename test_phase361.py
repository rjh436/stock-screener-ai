"""Quick validation of Phase 3.6.1 fixes"""
import sys
import os
sys.path.insert(0, '.')

print("🧪 Testing Phase 3.6.1 Fixes...\n")

# Test 1: Import check
print("1️⃣ Checking imports...")
try:
    from strategies.generic import GenericStrategy
    from simulation.paper_trader import PaperTrader
    print("   ✅ All imports successful\n")
except Exception as e:
    print(f"   ❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Strategy creation
print("2️⃣ Creating test strategy...")
try:
    genome = {
        "name": "Test",
        "type": "test",
        "entry_rules": [],
        "exit_rules": [],
        "stop_loss_atr": 3.0,
        "time_stop": 45
    }
    strat = GenericStrategy(genome)
    print(f"   ✅ Strategy created: {strat.name}\n")
except Exception as e:
    print(f"   ❌ Strategy creation failed: {e}\n")
    sys.exit(1)

# Test 3: Paper trader creation
print("3️⃣ Initializing paper trader...")
try:
    pt = PaperTrader()
    print(f"   ✅ Paper trader initialized")
    print(f"   📊 Cash: ${pt.state['cash']:,.0f}")
    print(f"   📊 Strategies loaded: {len(pt.strategies)}\n")
except Exception as e:
    print(f"   ❌ Paper trader failed: {e}\n")
    sys.exit(1)

print("="*60)
print("✅ ALL TESTS PASSED - Ready for backtest")
print("="*60)
print("\nNext step: Run full backtest")
print("Command: python3 test_phase36_validation.py")
