import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from execution.engine import prepare_backtest_data, run_backtest
    from data.loader import fetch_data_pack
    from data.universe import get_universe_symbols
    from strategies.strategy_loader import load_strategies
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def verify_minervini():
    print("🚀 VERIFY_MINERVINI: Starting Zombie Runner Check...")
    
    # 1. Load Data
    print("...Loading Data (Russell 3000, 2020-2025)...")
    symbols = get_universe_symbols("RUSSELL3000")
    # Fetching enough days to cover 2020-2025 (approx 1500 trading days + buffer)
    # User asked for 2500 days.
    data = fetch_data_pack(symbols, days=2500, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=2500, backtest_mode=True)
    
    # 2. Prepare Data & Apply Critical Patch
    print("...Preparing Data & Patching...")
    prepared = prepare_backtest_data(data, symbols, start_date="2020-01-01", global_data=g_data)
    
    # --- CRITICAL PATCH (Copied from optimize_midnight.py) ---
    print("🔧 Applying 'high_20_prev' and 'atr' Patch...")
    for sym, s_data in prepared.enriched.items():
        df = s_data.df
        # Breakout Entry calc
        if 'high' in df.columns:
            df['high_20_prev'] = df['high'].rolling(window=20).max().shift(1)
            
        # ATR calc for Stop Loss
        if 'atr' not in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
             df['tr'] = np.maximum(
                df['high'] - df['low'], 
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))
                )
            )
             df['atr'] = df['tr'].rolling(window=14).mean()
    # ---------------------------------------------------------

    # 3. Define Golden Rules Strategy
    strategy_config = {
        "name": "Minervini_Verifier_V1",
        "entry_rules": [
             {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99},
             {"col": "close", "op": ">", "ref": "sma50"} 
        ],
        "exit_rules": [], # Logic handled by engine params
        "risk_parameters": {
            "stop_loss_atr": 2.0,
            "max_positions": 5,
            "risk_per_trade": 0.02, 
            "max_pos_size_pct": 0.20
        },
        "execution_parameters": {
            "exit_sma": "sma10",        # Initial Tight Leash
            "enable_partial_profit": True,
            "partial_profit_r": 2.0,
            "move_stop_to_be": True,
            "partial_profit_day": 0,    # Immediate profit taking allowed
            "regime_ma": "sma200"
        },
        # Legacy/Helper Params
        "min_rs": 85,
        "vol_ma_ratio": 1.0,
        "adx_threshold": 15
    }

    print(f"🛡️  Strategy Config: {strategy_config['execution_parameters']}")

    # 4. Run Backtest
    print("...Running Backtest...")
    strat = load_strategies([strategy_config])[0]
    # Force Attributes just in case
    strat.min_rs = 85
    strat.vol_ma_ratio = 1.0
    strat.adx_threshold = 15
    
    results = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date="2020-01-01",
        end_date="2025-12-31",
        global_data=g_data
    )
    
    if not results:
        print("❌ CRITICAL: No results returned.")
        return

    res = results[0] if isinstance(results, list) else results
    final_val = res.get("final_value", 100000)
    cagr = ((final_val / 100000.0) ** (1/6) - 1) * 100 # Approx 6 years
    dd = res.get("max_drawdown", 0) * 100
    trades = res.get("total_trades", 0)
    
    print("\n" + "="*40)
    print(f"🏁 RESULT: CAGR: {cagr:.2f}% | DD: {dd:.2f}% | Trades: {trades}")
    print("="*40)
    
    if cagr > 15.0:
        print("✅ SUCCESS: Strategy > 15% CAGR. Zombie Runner is ALIVE!")
    else:
        print("⚠️  WARNING: Strategy underperformed (< 15%). Logic check needed.")

if __name__ == "__main__":
    verify_minervini()
