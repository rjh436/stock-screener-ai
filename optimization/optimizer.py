import sys
import os
import json
import itertools
import concurrent.futures
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.schwab_client import sd
from execution.engine import run_backtest
from data.indices import get_index_symbols

# Import Strategies
from strategies.rhcts import StrategyRHCTS
from strategies.connors_rsi import StrategyConnorsRSI
from strategies.cbc import StrategyCBC
from strategies.raptor import StrategyRaptor
from strategies.hmp import StrategyHMP
from strategies.cgm import StrategyCGM10
from strategies.r2x import StrategyR2X
from strategies.mindful import StrategyMindful
from strategies.generic import GenericStrategy
from optimization.evolution import EvolutionEngine

# Config Path
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/strategies.json'))
GEN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json'))

def fetch_sp100_symbols():
    # Use the shared logic to fetch full S&P 100
    return get_index_symbols("S&P 100")

def fetch_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    print(f"Fetching data for {len(symbols)} symbols over last {days} days...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50) # +50 for warmup
    
    data = {}
    for sym in symbols:
        try:
            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                df = df.set_index("datetime").sort_index()
                data[sym] = df
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    return data

def optimize_strategy(name, strategy_class, grid, data_map):
    print(f"\n--- Optimizing {name} ---")
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Testing {len(combinations)} parameter combinations...")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for params in combinations:
            strat = strategy_class(params)
            futures.append(executor.submit(run_backtest, strat, data_map))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Backtest failed: {e}")

    # Filter: Win Rate > 50%, Profit > 0% (relaxed for broad search)
    valid_results = [r for r in results if r["hit_rate"] > 50 and r["pnl_pct"] > 0]
    
    if not valid_results:
        print(f"No profitable configs found for {name}. Picking best PnL regardless.")
        valid_results = results
        
    if not valid_results:
        print(f"No results at all for {name}.")
        return None

    # Sort by PnL
    best_result = sorted(valid_results, key=lambda x: x["pnl_pct"], reverse=True)[0]
    
    print(f"Best {name}: {best_result['pnl_pct']:.2f}% return, {best_result['hit_rate']:.2f}% win rate, {best_result['total_trades']} trades")
    print(f"Best Params: {best_result['params']}")
    
    if best_result['total_trades'] == 0:
        print(f"⚠️ WARNING: {name} had 0 trades. Skipping update.")
        return None
        
    return best_result

def run_optimization():
    print("Starting Apex-Schwab Multi-Strategy Optimizer...")
    
    # 1. Data Prep
    symbols = fetch_sp100_symbols()
    data_map = fetch_data(symbols, days=400)
    
    # Fetch Global Data
    print("Fetching Global Data (SPY, VIX)...")
    global_data = {}
    spy_map = fetch_data(["SPY"], days=400)
    if "SPY" in spy_map: global_data["SPY"] = spy_map["SPY"]
    
    vix_map = fetch_data(["$VIX"], days=400)
    if "$VIX" in vix_map: global_data["VIX"] = vix_map["$VIX"]
    else:
        vix_map = fetch_data(["VIX"], days=400)
        if "VIX" in vix_map: global_data["VIX"] = vix_map["VIX"]
    
    if not data_map:
        print("No data fetched. Aborting.")
        return

    # 2. Define Strategy Grids
    strategies_to_optimize = [
        {
            "name": "RHCTS",
            "class": StrategyRHCTS,
            "grid": {
                "vol_threshold": [500_000, 1_000_000],
                "rsi_limit": [60, 70, 75],
                "crsi_limit": [15, 20, 25],
                "time_stop": [10, 15]
            }
        },
        {
            "name": "ConnorsRSI",
            "class": StrategyConnorsRSI,
            "grid": {
                "crsi_limit": [10, 15, 20],
                "time_stop": [5, 10, 15]
            }
        },
        {
            "name": "CBC",
            "class": StrategyCBC,
            "grid": {
                "vol_mult": [1.3, 1.5, 2.0],
                "time_stop": [10, 15, 20]
            }
        },
        {
            "name": "Raptor",
            "class": StrategyRaptor,
            "grid": {
                "rsi_limit": [10, 15, 20],
                "time_stop": [15, 20, 25]
            }
        },
        {
            "name": "HMP",
            "class": StrategyHMP,
            "grid": {
                "rsi_limit": [30, 35, 40],
                "time_stop": [5, 10, 15]
            }
        },
        {
            "name": "CGM10",
            "class": StrategyCGM10,
            "grid": {
                "vol_mult_bx": [1.5, 2.0],
                "time_stop": [20, 25, 30]
            }
        },
        {
            "name": "R2X",
            "class": StrategyR2X,
            "grid": {
                "rsi_threshold_low": [2, 5, 8],
                "time_stop": [5, 10, 15]
            }
        },
        {
            "name": "Mindful",
            "class": StrategyMindful,
            "grid": {
                "time_stop": [7, 9, 12]
            }
        }
    ]

    # 3. Run Optimization Loop
    updates = {}
    
    # for strat_conf in strategies_to_optimize:
    #     res = optimize_strategy(strat_conf["name"], strat_conf["class"], strat_conf["grid"], data_map)
    #     if res:
    #         updates[strat_conf["name"]] = res["params"]
    #         updates[strat_conf["name"]]["avg_win_duration"] = res.get("avg_win_duration", 10)

    # 4. Update Config
    if not updates:
        print("No updates to save (Grid Search skipped).")
        # return  <-- REMOVED THIS LINE

    try:
        with open(CONFIG_PATH, "r") as f:
            current_config = json.load(f)
    except:
        current_config = {}
        
    for name, params in updates.items():
        current_config[name] = params
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(current_config, f, indent=4)
        
    print("\nGlobal Configuration updated successfully.")

    # --- 5. Run Evolution (Experimental) ---
    run_evolution(data_map, global_data)

def run_evolution(data_map, global_data):
    print("\n--- Starting Evolutionary Strategy Generation ---")
    engine = EvolutionEngine()
    
    # Load existing population if exists
    try:
        with open(GEN_CONFIG_PATH, "r") as f:
            saved_pop = json.load(f)
            if saved_pop:
                print(f"Loaded {len(saved_pop)} existing generated strategies.")
                engine.population = saved_pop
            else:
                print("Initializing new population...")
                engine.generate_initial_population()
    except:
        print("Initializing new population...")
        engine.generate_initial_population()

    # Run Evolution Loop
    generations = 10
    
    for gen in range(generations):
        # Run Backtests for Population
        print(f"\nEvaluating Generation {engine.generation_count} ({len(engine.population)} strategies)...")
        
        pop_results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(run_backtest, GenericStrategy(genome), data_map, None, 100000.0, None, global_data): genome for genome in engine.population}
            
            for future in concurrent.futures.as_completed(futures):
                genome = futures[future]
                try:
                    res = future.result()
                    pop_results.append({
                        "genome": genome,
                        "pnl_pct": res["pnl_pct"],
                        "cagr": res.get("cagr", 0.0),
                        "hit_rate": res["hit_rate"],
                        "avg_profit_pct": res.get("avg_profit_pct", 0.0),  # Added
                        "sharpe": res.get("sharpe", 0.0),  # Added
                        "trades": res["total_trades"]
                    })
                except Exception as e:
                    print(f"Evolution backtest failed for {genome['name']}: {e}")

        # REDESIGNED FITNESS FUNCTION: Exponentially Reward Swing Trading
        # Philosophy: Harshly punish scalping, massively reward 10%+ avg profit
        def calculate_fitness(result):
            win_rate = result.get("hit_rate", 0)  # 0-1 (e.g., 0.70 = 70%)
            avg_profit_pct = result.get("avg_profit_pct", 0)  # Percentage (e.g., 10.0 = 10%)
            trades = result.get("trades", 0)
            sharpe = result.get("sharpe", 0)
            cagr = result.get("cagr", 0)
            
            # Disqualify immediately if critical failures
            if trades == 0 or cagr < 0.05:  # No trades or <5% CAGR
                return -999.0
            
            # PRIMARY SCORE: Avg Profit % (EXPONENTIAL REWARDS)
            # This is the core metric - we want 10%+ avg profit
            if avg_profit_pct >= 15.0:
                # Jackpot: 15%+ avg profit = 1000 points (best strategies)
                profit_score = 1000.0
            elif avg_profit_pct >= 10.0:
                # Target hit: 10-15% = 600-900 points (exponential scale)
                profit_score = 600 + (avg_profit_pct - 10.0) * 80  # 10%=600, 15%=1000
            elif avg_profit_pct >= 5.0:
                # Acceptable: 5-10% = 200-600 points (linear scale)
                profit_score = 200 + (avg_profit_pct - 5.0) * 80  # 5%=200, 10%=600
            else:
                # Scalping penalty: <5% = 0-100 points (harsh penalty)
                profit_score = avg_profit_pct * 20  # 1%=20, 5%=100
            
            # SECONDARY SCORE: Win Rate (30% weight, max 30 points)
            # Still important, but subordinate to avg profit
            win_score = win_rate * 30.0  # 70% = 21 points
            
            # TERTIARY SCORE: CAGR (10% weight, max 20 points)
            # Cap at 20 points to prevent CAGR from dominating
            cagr_score = min(cagr * 10.0, 20.0)  # 20% CAGR = cap
            
            # BONUS: Sharpe Ratio (max 15 points)
            sharpe_score = min(sharpe * 7.5, 15.0)  # Sharpe 2.0 = cap
            
            # TRADE FREQUENCY CALIBRATION (Swing Trading)
            # Target: 200-800 trades/year (weekly frequency x 100 stocks)
            # Over ~1 year backtest = 200-800 trades total
            if trades < 200:
                # Too few setups - missing opportunities
                trade_penalty = (200 - trades) / 200 * 20.0  # Up to -20
            elif trades > 800:
                # Overtrading - likely scalping noise
                trade_penalty = (trades - 800) / 800 * 30.0  # Up to -30
            else:
                trade_penalty = 0  # Sweet spot (200-800)
            
            final_score = profit_score + win_score + cagr_score + sharpe_score - trade_penalty
            return final_score
                
        ranked = sorted(pop_results, key=calculate_fitness, reverse=True)
        
        print(f"Top Strategies (Gen {engine.generation_count}):")
        for i, r in enumerate(ranked[:3]):
            print(f"{i+1}. {r['genome']['name']}: {r['cagr']*100.0:.2f}% CAGR, {r['hit_rate']:.2f}% win rate")

        # Evolve (unless last gen)
        if gen < generations - 1:
            engine.evolve(ranked)
    
    # Save Final Population
    with open(GEN_CONFIG_PATH, "w") as f:
        json.dump(engine.population, f, indent=4)
    print("\nEvolution complete. Final population saved.")

if __name__ == "__main__":
    run_optimization()
