#!/usr/bin/env python3
import copy
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy

WINDOWS = {
    "10Y": ("2016-02-23", "2026-02-27")
}

def perturb_genome(base_cfg: dict, scale: float) -> dict:
    mutant = copy.deepcopy(base_cfg)
    # Target numerical parameters typical for risk/reward and trailing stops
    keys_to_perturb = [
        "stop_loss_pct", "time_stop_profit_pct", "partial_profit_pct",
        "trailing_stop_atr_multiplier", "breakeven_at_pct"
    ]
    
    for k in keys_to_perturb:
        if k in mutant and isinstance(mutant[k], (int, float)):
            val = float(mutant[k])
            # Randomly shift by +/- scale%
            shift = val * scale * random.choice([1, -1])
            mutant[k] = round(val + shift, 4)
            
    # Also perturb scoring weights if present
    if "scoring_weights" in mutant:
        for w_k, w_v in mutant["scoring_weights"].items():
            if isinstance(w_v, (int, float)):
                shift = float(w_v) * scale * random.choice([1, -1])
                mutant["scoring_weights"][w_k] = round(float(w_v) + shift, 2)
                
    # Re-enforce constraints!
    mutant["allow_margin"] = False
    mutant["same_day_open_entries"] = 0
    return mutant

def evaluate_genome(cfg: dict, prepared: any, window: tuple) -> dict:
    strat = SuperperformanceStrategy(cfg)
    try:
        res = run_backtest([strat], prepared, start_cash=100000.0, start_date=window[0], end_date=window[1])
        summary = res[0] if isinstance(res, list) else res
        return {
            "cagr": summary.get("cagr", 0.0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "trades": summary.get("total_trades", 0),
            "summary": summary
        }
    except Exception as e:
        logging.error(f"Genomic evaluation failed: {e}")
        return {"cagr": 0.0, "max_drawdown": 1.0, "trades": 0}

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # We already know the top candidates based on the ranking logs (bypassing JSON key collision)
    top_candidates = ["Superperformance", "Superperformance Practical Selective V4 Candidate"]
    logging.info(f"Top candidates for optimization: {top_candidates}")
    
    # Load market data once
    logging.info("Loading SP1500 + SPY/VIX market data...")
    symbols = get_index_symbols("S&P 1500") or []
    data_dict = fetch_data_pack(symbols, days=3500)
    start_date, end_date = WINDOWS["10Y"]
    prepared = prepare_backtest_data(data_dict, start_date=start_date)
    
    # Let's map names back to source files (hardcoded for brevity based on UI_STRATEGIES)
    name_to_file = {
        "Superperformance": "config/superperformance_winner.json",
        "Superperformance (Practical EOD No Leverage)": "config/superperformance_practical_no_leverage_optimized.json",
        "Superperformance (Practical EOD Cash V2)": "config/superperformance_practical_eod_cash_v2.json",
        "Superperformance Practical Selective V3": "config/superperformance_practical_selective_v3.json",
        "Superperformance Practical Selective V4 Candidate": "config/superperformance_practical_selective_v4_candidate.json",
        "Superperformance Practical Risk-Off Only": "config/superperformance_practical_riskoff_only.json",
        "Superperformance Alpha B4": "config/superperformance_alpha_b4.json",
    }
    
    for candidate in top_candidates:
        file_path = name_to_file.get(candidate)
        if not file_path or not os.path.exists(file_path):
            continue
            
        with open(file_path, "r") as f:
            base_cfg = json.load(f)
            
        logging.info(f"--- Stress Testing {candidate} ---")
        base_eval = evaluate_genome(base_cfg, prepared, WINDOWS["10Y"])
        logging.info(f"Base 10Y CAGR: {base_eval['cagr']:.2%} | DD: {base_eval['max_drawdown']:.2%}")
        
        # Perturbation analysis: scale 10% and 20%
        scales = [0.10, 0.20]
        for scale in scales:
            logging.info(f"Running +/- {int(scale*100)}% perturbation tests...")
            results = []
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = []
                for _ in range(5): # 5 mutants per scale
                    mutant = perturb_genome(base_cfg, scale)
                    futures.append(ex.submit(evaluate_genome, mutant, prepared, WINDOWS["10Y"]))
                
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    
            avg_cagr = sum(r["cagr"] for r in results) / len(results)
            avg_dd = sum(r["max_drawdown"] for r in results) / len(results)
            logging.info(f"Avg Mutant 10Y CAGR: {avg_cagr:.2%} | Avg DD: {avg_dd:.2%}")
            
            # Reject brittle winners
            if avg_cagr < (base_eval["cagr"] * 0.70):
                logging.warning(f"Strategy {candidate} is BRITTLE! Drops >30% performance with {int(scale*100)}% perturbation.")
            else:
                logging.info(f"Strategy {candidate} is ROBUST against {int(scale*100)}% perturbation.")

if __name__ == "__main__":
    main()
