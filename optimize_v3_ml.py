import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from strategies.generic import GenericStrategy

# ============================================================================
# CONFIGURATION: QULLAMAGGIE V3 (BAYESIAN + REGIME SAMPLING)
# ============================================================================
STRATEGY_TEMPLATE = {
    "name": "Apex Kinetic VCP (Optimizer)",
    "type": "breakout",
    "entry_rules": [
        {"col": "rsi14", "op": ">", "val": 55},
        {"col": "rs_rating", "op": ">", "val": 80},
        {"col": "adr_pct", "op": ">", "val": 2.5},
        {"col": "bb_width", "op": "<", "val": 0.25},
        {"col": "close", "op": ">", "ref": "donchian_20"},
        {"col": "volume", "op": ">", "ref": "vol_ma20", "mult": 1.5},
    ],
    "market_filter_mode": "traffic_light",
    "yellow_rs_floor": 85,
    "red_bypass_rs": 95,
    "risk_per_trade": 0.015,
    "max_positions": 10,
    "scoring_type": "breakout",
    "min_entry_score": 0.0,
}

WFV_WINDOWS = [
    {
        "name": "Regime_Grind_2006",
        "test_start": "2006-01-01",
        "test_end": "2007-12-31",
        "min_trades": 10,
    },
    {
        "name": "Regime_Crisis_2008",
        "test_start": "2008-01-01",
        "test_end": "2008-12-31",
        "min_trades": 0,
        "max_dd_cap": 35.0,
    },
    {
        "name": "Regime_Chop_2014",
        "test_start": "2014-01-01",
        "test_end": "2015-12-31",
        "min_trades": 10,
    },
    {
        "name": "Regime_Boom_2020",
        "test_start": "2020-01-01",
        "test_end": "2021-12-31",
        "min_trades": 10,
    },
    {
        "name": "Regime_Bear_2022",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
        "min_trades": 0,
    },
]

WORKER_COUNT = 12
TOTAL_TRIALS = int(os.environ.get("APEX_V3_TRIALS", "200"))
STUDY_NAME = "apex_v3_hpo"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_v3.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
SEED = 42

_WORKER_CACHE = None


def worker_init():
    if _WORKER_CACHE is None:
        raise RuntimeError("Worker cache is not initialized.")


def _prepare_cache():
    universe = get_index_symbols("R3000")
    if not universe:
        universe = get_index_symbols("SP1500")
    if not universe:
        raise RuntimeError("No universe symbols loaded.")

    raw_data = fetch_data_pack(universe, backtest_mode=True)
    if not raw_data:
        raise RuntimeError("No data loaded for universe.")

    return prepare_backtest_data(raw_data, None, None, None)


def _extract_trades_df(trades_list):
    if not trades_list:
        return pd.DataFrame(columns=["entry_dt", "exit_dt", "return_pct"])

    df = pd.DataFrame(trades_list)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    if "Entry Date" in df.columns:
        df["entry_dt"] = pd.to_datetime(df["Entry Date"], errors="coerce")
    elif "entry date" in cols_lower:
        df["entry_dt"] = pd.to_datetime(df[cols_lower["entry date"]], errors="coerce")
    else:
        df["entry_dt"] = pd.NaT

    if "Exit Date" in df.columns:
        df["exit_dt"] = pd.to_datetime(df["Exit Date"], errors="coerce")
    elif "exit date" in cols_lower:
        df["exit_dt"] = pd.to_datetime(df[cols_lower["exit date"]], errors="coerce")
    else:
        df["exit_dt"] = pd.NaT

    if "Return %" in df.columns:
        df["return_pct"] = pd.to_numeric(df["Return %"], errors="coerce")
    elif "return %" in cols_lower:
        df["return_pct"] = pd.to_numeric(df[cols_lower["return %"]], errors="coerce")
    elif "return_pct" in cols_lower:
        df["return_pct"] = pd.to_numeric(df[cols_lower["return_pct"]], errors="coerce")
    elif "Entry" in df.columns and "Exit" in df.columns:
        entry = pd.to_numeric(df["Entry"], errors="coerce")
        exit_px = pd.to_numeric(df["Exit"], errors="coerce")
        df["return_pct"] = (exit_px - entry) / entry * 100.0
    elif "entry" in cols_lower and "exit" in cols_lower:
        entry = pd.to_numeric(df[cols_lower["entry"]], errors="coerce")
        exit_px = pd.to_numeric(df[cols_lower["exit"]], errors="coerce")
        df["return_pct"] = (exit_px - entry) / entry * 100.0
    else:
        df["return_pct"] = np.nan

    return df


def _extract_equity_df(equity_curve):
    if not equity_curve:
        return pd.DataFrame(columns=["Equity"])

    df = pd.DataFrame(equity_curve)
    if "Date" not in df.columns or "Equity" not in df.columns:
        return pd.DataFrame(columns=["Equity"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce")
    df = df.dropna(subset=["Date", "Equity"]).sort_values("Date")
    return df.set_index("Date")


def _window_metrics(trades_df, equity_df, start_date, end_date):
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    eq_slice = equity_df.loc[(equity_df.index >= start_dt) & (equity_df.index <= end_dt)]
    if eq_slice.empty:
        return {
            "cagr": 0.0,
            "max_dd": 0.0,
            "trades": 0,
        }

    start_equity = float(eq_slice["Equity"].iloc[0])
    end_equity = float(eq_slice["Equity"].iloc[-1])
    span_days = (eq_slice.index[-1] - eq_slice.index[0]).days

    if span_days <= 0 or start_equity <= 0:
        cagr = 0.0
    else:
        cagr = (end_equity / start_equity) ** (365.0 / span_days) - 1.0

    equity = eq_slice["Equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = abs(drawdown.min() * 100.0) if equity.size else 0.0

    trades_window = trades_df[(trades_df["exit_dt"] >= start_dt) & (trades_df["exit_dt"] <= end_dt)]
    returns = trades_window["return_pct"].dropna().to_numpy(dtype=float)
    trade_count = int(returns.size)

    return {
        "cagr": float(cagr),
        "max_dd": float(max_dd),
        "trades": trade_count,
    }


def _score_window(metrics, window):
    min_trades = int(window.get("min_trades", 0) or 0)
    if metrics["trades"] < min_trades:
        return 0.0

    max_dd_cap = float(window.get("max_dd_cap", 40.0) or 40.0)
    if metrics["max_dd"] > max_dd_cap:
        return 0.0

    max_dd = metrics["max_dd"]
    if not np.isfinite(max_dd) or max_dd <= 0:
        return 0.0

    cagr = metrics["cagr"]
    if not np.isfinite(cagr):
        return 0.0

    calmar = cagr / (max_dd / 100.0)
    if not np.isfinite(calmar):
        return 0.0

    return float(calmar)


def _build_strategy(params):
    strat = dict(STRATEGY_TEMPLATE)
    strat["name"] = (
        f"QM_{params['exit_mode']}_Stop{params['stop_loss_atr']:.2f}"
        f"_ADR{params['adr_pct']:.2f}"
    )
    strat["stop_loss_atr"] = params["stop_loss_atr"]
    strat["limit_ratio"] = params["limit_ratio"]
    strat["time_stop"] = params["time_stop"]
    strat["partial_profit_day"] = params["partial_profit_day"]

    entry_rules = [rule.copy() for rule in strat["entry_rules"]]
    for rule in entry_rules:
        if rule.get("col") == "rsi14":
            rule["val"] = params["rsi14"]
        elif rule.get("col") == "rs_rating":
            rule["val"] = params["rs_rating"]
        elif rule.get("col") == "adr_pct":
            rule["val"] = params["adr_pct"]
        elif rule.get("col") == "bb_width":
            rule["val"] = params["bb_width"]
    strat["entry_rules"] = entry_rules

    exit_mode = params["exit_mode"]
    if exit_mode.startswith("trail"):
        strat["trail_atr"] = float(exit_mode.replace("trail", ""))
        strat["exit_rules"] = []
    else:
        strat["trail_atr"] = 0.0
        strat["exit_rules"] = [{"col": "close", "op": "<", "ref": exit_mode}]

    return strat


def objective(trial):
    if _WORKER_CACHE is None:
        return 0.0

    params = {
        "stop_loss_atr": trial.suggest_float("stop_loss_atr", 1.2, 2.5),
        "adr_pct": trial.suggest_float("adr_pct", 2.0, 5.0),
        "bb_width": trial.suggest_float("bb_width", 0.08, 0.28),
        "rs_rating": trial.suggest_int("rs_rating", 75, 97),
        "rsi14": trial.suggest_int("rsi14", 45, 70),
        "exit_mode": trial.suggest_categorical(
            "exit_mode",
            ["ema10", "sma10", "trail2.5"],
        ),
        "limit_ratio": trial.suggest_float("limit_ratio", 1.0, 1.0015),
        "time_stop": trial.suggest_int("time_stop", 10, 40),
        "partial_profit_day": trial.suggest_int("partial_profit_day", 3, 5),
    }

    strat_conf = _build_strategy(params)
    result = run_backtest(
        GenericStrategy(strat_conf),
        None,
        None,
        start_cash=100000.0,
        start_date="2006-01-01",
        pre_calculated_data=_WORKER_CACHE,
    )

    if not isinstance(result, dict):
        return 0.0

    trades_df = _extract_trades_df(result.get("trades_list", []))
    equity_df = _extract_equity_df(result.get("equity_curve", []))
    if equity_df.empty:
        return 0.0

    window_scores = []
    window_attrs = {}

    for idx, window in enumerate(WFV_WINDOWS):
        metrics = _window_metrics(trades_df, equity_df, window["test_start"], window["test_end"])
        score = _score_window(metrics, window)
        window_scores.append(score)
        window_attrs[window["name"]] = {"score": score, **metrics}

        trial.report(score, step=idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not window_scores:
        return 0.0

    min_score = min(window_scores)
    avg_score = float(np.mean(window_scores)) if window_scores else 0.0
    final_score = (0.6 * min_score) + (0.4 * avg_score)

    trial.set_user_attr("window_metrics", window_attrs)
    return float(final_score)


def _optuna_worker(storage_url, study_name, n_trials, seed):
    sampler = TPESampler(seed=seed, n_startup_trials=50)
    pruner = MedianPruner(n_startup_trials=50, n_warmup_steps=0)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=False)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Starting Bayesian HPO with regime sampling.")
    print(f"Workers: {WORKER_COUNT}, Trials: {TOTAL_TRIALS}, Storage: {DB_PATH}")

    global _WORKER_CACHE
    _WORKER_CACHE = _prepare_cache()
    print(f"Prepared cache for {len(_WORKER_CACHE.enriched)} symbols.")

    sampler = TPESampler(seed=SEED, n_startup_trials=50)
    pruner = MedianPruner(n_startup_trials=50, n_warmup_steps=0)
    optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    ctx = mp.get_context("fork")
    trials_per_worker = TOTAL_TRIALS // WORKER_COUNT
    extras = TOTAL_TRIALS % WORKER_COUNT

    futures = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=ctx,
        initializer=worker_init,
    ) as executor:
        for idx in range(WORKER_COUNT):
            n_trials = trials_per_worker + (1 if idx < extras else 0)
            if n_trials <= 0:
                continue
            seed = SEED + idx
            futures.append(executor.submit(_optuna_worker, STORAGE_URL, STUDY_NAME, n_trials, seed))

        for future in as_completed(futures):
            future.result()

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    print("Optimization complete.")
    print(f"Best score: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
