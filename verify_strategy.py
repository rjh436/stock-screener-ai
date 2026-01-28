import json
import sys
import os

# 1. Fix Path to allow imports from root
sys.path.append(os.getcwd())

from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from strategies.generic import GenericStrategy


def run_verification():
    print("Starting SEPA Verification Run...")

    # 2. Load the Optimized Genome
    genome_path = 'config/best_sepa_genome.json'
    if not os.path.exists(genome_path):
        print(f"Error: {genome_path} not found.")
        return

    with open(genome_path, 'r') as f:
        genome = json.load(f)

    # 3. Initialize Strategy
    strat = GenericStrategy(genome)
    strat._name = "SEPA_CHAMPION_VERIFY"

    print(
        "Loaded Strategy: "
        f"RS>{genome['rs_rating']} | "
        f"VCP<{genome['bb_width_max']} | "
        f"Stop:{genome.get('stop_loss_atr')}xATR"
    )

    # 4. Fetch Data (2020 Momentum Basket)
    symbols = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'ENPH', 'MRNA', 'ZM', 'PTON', 'SHOP', 'DOCU']
    print(f"Fetching data for {len(symbols)} tickers...")

    data = fetch_data_pack(symbols, days=3000, backtest_mode=True)
    g_data = fetch_data_pack(['SPY'], days=3000, backtest_mode=True)

    if not data:
        print("No data returned.")
        return

    # 5. Run Backtest (The Covid Bull Run)
    print("Running Backtest (March 2020 - March 2021)...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)

    results = run_backtest(
        strat,
        prepared,
        start_cash=100000.0,
        start_date='2020-03-01',
        end_date='2021-03-01',
        global_data=g_data
    )

    # Handle result format (list vs dict)
    res = results[0] if isinstance(results, list) else results

    # 6. Final Report
    profit = res['final_value'] - 100000
    ret_pct = (profit / 100000) * 100

    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS")
    print(f"Final Balance: ${res['final_value']:,.2f}")
    print(f"Total Return:  +{ret_pct:.2f}%")
    print(f"Total Trades:  {res['total_trades']}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_verification()
