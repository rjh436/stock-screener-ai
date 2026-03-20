from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import optimize_superperformance as sp_opt

MIN_TRADE_FLOOR_PER_YEAR = 6.0
MIN_TOTAL_TRADES_FLOOR = 24.0
MIN_STITCHED_OOS_FOLDS = 3
MIN_STITCHED_OOS_CAGR_PCT = 0.0
MIN_WORST_FOLD_RETURN_PCT = -35.0
MAX_GROSS_EXPOSURE_PCT = 1.0001
TOP2_YEAR_SHARE_SOFT_CAP = 0.75


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def dd_pct(value: Any) -> float:
    out = safe_float(value, 0.0)
    if out <= 1.0:
        out *= 100.0
    return abs(out)


def years_between(start_date: str, end_date: str) -> float:
    start = Path(start_date).name if "/" not in start_date else start_date
    end = Path(end_date).name if "/" not in end_date else end_date
    from pandas import Timestamp

    span_days = max(1, int((Timestamp(end) - Timestamp(start)).days))
    return span_days / 365.25


def has_metrics(metrics: Dict[str, Any]) -> bool:
    return bool(metrics) and any(k in metrics for k in ("cagr_pct", "total_trades", "trades_per_year"))


def metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float(metrics.get(key, default), default)


def rejection_score(code: int) -> float:
    return -1_000_000.0 - float(code)


def rejection_reason(code: int | None) -> str:
    if code is None:
        return ""
    if code < 100:
        return f"same-day open contamination ({code})"
    mapping = {
        100: "gross exposure above cash-only limit",
        150: "non-positive CAGR",
        151: "drawdown gate failed",
        200: "trade-count floor failed",
        300: "stitched OOS CAGR below gate",
        301: "stitched OOS worst fold below gate",
    }
    return mapping.get(int(code), f"rejection code {int(code)}")


def build_summary_metrics(
    *,
    cagr_pct: float,
    max_dd_pct: float,
    total_trades: int,
    sample_years: float,
    hit_rate_pct: float = 0.0,
    same_day_open_entries: int = 0,
    max_gross_exposure_pct: float = 1.0,
    final_value: float = 0.0,
    worst_12m_return_pct: float = float("nan"),
    stitched_oos_cagr_pct: float = float("nan"),
    stitched_oos_folds: int = 0,
    stitched_oos_worst_fold_return_pct: float = float("nan"),
    worst_24m_cagr_pct: float = float("nan"),
    median_24m_cagr_pct: float = float("nan"),
    rolling_24m_samples: int = 0,
    max_year_pnl_share: float = float("nan"),
    top2_year_pnl_share: float = float("nan"),
    negative_years: int = 0,
) -> Dict[str, Any]:
    years = max(0.01, safe_float(sample_years, 0.0))
    trades = int(max(0, int(total_trades)))
    return {
        "cagr_pct": safe_float(cagr_pct, 0.0),
        "hit_rate_pct": safe_float(hit_rate_pct, 0.0),
        "max_dd_pct": dd_pct(max_dd_pct),
        "sample_years": years,
        "total_trades": trades,
        "trades_per_year": trades / years,
        "same_day_open_entries": int(max(0, int(same_day_open_entries))),
        "max_gross_exposure_pct": safe_float(max_gross_exposure_pct, 0.0),
        "final_value": safe_float(final_value, 0.0),
        "worst_12m_return_pct": safe_float(worst_12m_return_pct, float("nan")),
        "stitched_oos_cagr_pct": safe_float(stitched_oos_cagr_pct, float("nan")),
        "stitched_oos_folds": int(max(0, int(stitched_oos_folds))),
        "stitched_oos_worst_fold_return_pct": safe_float(stitched_oos_worst_fold_return_pct, float("nan")),
        "worst_24m_cagr_pct": safe_float(worst_24m_cagr_pct, float("nan")),
        "median_24m_cagr_pct": safe_float(median_24m_cagr_pct, float("nan")),
        "rolling_24m_samples": int(max(0, int(rolling_24m_samples))),
        "max_year_pnl_share": safe_float(max_year_pnl_share, float("nan")),
        "top2_year_pnl_share": safe_float(top2_year_pnl_share, float("nan")),
        "negative_years": int(max(0, int(negative_years))),
    }


def pack_result_metrics(
    result: Dict[str, Any],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    sample_years: float | None = None,
) -> Dict[str, Any]:
    audit = result.get("audit_report") if isinstance(result.get("audit_report"), dict) else {}
    total_trades = int(result.get("total_trades", 0) or 0)
    years = sample_years
    if years is None:
        if start_date is not None and end_date is not None:
            years = max(0.01, years_between(start_date, end_date))
        else:
            years = max(0.0, safe_float(result.get("active_period_years", 0.0), 0.0))
    years = max(0.01, float(years or 0.0))
    equity_curve = result.get("equity_curve", []) or []
    stitched_oos_cagr_pct, stitched_oos_folds, stitched_oos_worst_fold = sp_opt._stitched_oos_cagr_pct(
        equity_curve,
        train_years=3,
        test_years=1,
    )
    worst_12m_return_pct = sp_opt._worst_rolling_12m_return_pct(equity_curve)
    worst_24m_cagr_pct, median_24m_cagr_pct, rolling_24m_samples = sp_opt._rolling_window_cagr_stats(
        equity_curve,
        window_years=2,
        step_days=21,
    )
    year_conc = sp_opt._year_concentration_stats(equity_curve)
    return {
        "cagr_pct": safe_float(result.get("cagr", 0.0), 0.0) * 100.0,
        "hit_rate_pct": safe_float(result.get("hit_rate", 0.0), 0.0),
        "max_dd_pct": dd_pct(result.get("max_drawdown_pct", 0.0)),
        "sample_years": years,
        "total_trades": total_trades,
        "trades_per_year": total_trades / years,
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": safe_float(audit.get("max_gross_exposure_pct", 0.0), 0.0),
        "final_value": safe_float(result.get("final_value", 0.0), 0.0),
        "worst_12m_return_pct": safe_float(worst_12m_return_pct, float("nan")),
        "stitched_oos_cagr_pct": safe_float(stitched_oos_cagr_pct, float("nan")),
        "stitched_oos_folds": int(stitched_oos_folds or 0),
        "stitched_oos_worst_fold_return_pct": safe_float(stitched_oos_worst_fold, float("nan")),
        "worst_24m_cagr_pct": safe_float(worst_24m_cagr_pct, float("nan")),
        "median_24m_cagr_pct": safe_float(median_24m_cagr_pct, float("nan")),
        "rolling_24m_samples": int(rolling_24m_samples or 0),
        "max_year_pnl_share": safe_float(year_conc.get("max_positive_share"), float("nan")),
        "top2_year_pnl_share": safe_float(year_conc.get("top2_positive_share"), float("nan")),
        "negative_years": int(year_conc.get("negative_years", 0) or 0),
    }


def select_primary_metrics(metrics_5y: Dict[str, Any], metrics_10y: Dict[str, Any]) -> Dict[str, Any]:
    return metrics_10y if has_metrics(metrics_10y) else metrics_5y


def core_rejection_code(
    primary: Dict[str, Any],
    *,
    metrics_5y: Dict[str, Any] | None = None,
    metrics_10y: Dict[str, Any] | None = None,
    max_drawdown_gate_pct: float | None = None,
    require_positive_cagr: bool = True,
) -> int | None:
    metrics_5y = metrics_5y or {}
    metrics_10y = metrics_10y or {}
    same_day = int(metrics_10y.get("same_day_open_entries", 0) or 0) + int(metrics_5y.get("same_day_open_entries", 0) or 0)
    if same_day > 0:
        return same_day

    max_gross = max(
        metric(metrics_5y, "max_gross_exposure_pct", 0.0),
        metric(metrics_10y, "max_gross_exposure_pct", 0.0),
        metric(primary, "max_gross_exposure_pct", 0.0),
    )
    if max_gross > MAX_GROSS_EXPOSURE_PCT:
        return 100

    cagr_pct = metric(primary, "cagr_pct", 0.0)
    if require_positive_cagr and cagr_pct <= 0.0:
        return 150

    max_dd_pct = abs(metric(primary, "max_dd_pct", 0.0))
    if max_drawdown_gate_pct is not None and max_dd_pct >= float(max_drawdown_gate_pct):
        return 151

    sample_years = max(metric(primary, "sample_years", 0.0), 0.0)
    total_trades = metric(primary, "total_trades", 0.0)
    min_required_trades = max(MIN_TOTAL_TRADES_FLOOR, sample_years * MIN_TRADE_FLOOR_PER_YEAR)
    if total_trades < min_required_trades:
        return 200

    stitched_folds = int(primary.get("stitched_oos_folds", 0) or 0)
    stitched_oos_cagr = metric(primary, "stitched_oos_cagr_pct", float("nan"))
    worst_fold = metric(primary, "stitched_oos_worst_fold_return_pct", float("nan"))
    if stitched_folds >= MIN_STITCHED_OOS_FOLDS:
        if math.isfinite(stitched_oos_cagr) and stitched_oos_cagr < MIN_STITCHED_OOS_CAGR_PCT:
            return 300
        if math.isfinite(worst_fold) and worst_fold < MIN_WORST_FOLD_RETURN_PCT:
            return 301

    return None


def promotion_gate_status(
    primary: Dict[str, Any],
    *,
    metrics_5y: Dict[str, Any] | None = None,
    metrics_10y: Dict[str, Any] | None = None,
    max_drawdown_gate_pct: float | None = None,
    require_positive_cagr: bool = True,
) -> tuple[bool, int | None, str]:
    code = core_rejection_code(
        primary,
        metrics_5y=metrics_5y,
        metrics_10y=metrics_10y,
        max_drawdown_gate_pct=max_drawdown_gate_pct,
        require_positive_cagr=require_positive_cagr,
    )
    return code is None, code, rejection_reason(code)


def baseline_promotion_status(
    champion: Dict[str, Any],
    baseline: Dict[str, Any] | None,
    *,
    score_key: str,
    max_drawdown_gate_pct: float | None = None,
) -> tuple[bool, str]:
    ok, _, reason = promotion_gate_status(
        champion,
        metrics_5y=champion,
        metrics_10y={},
        max_drawdown_gate_pct=max_drawdown_gate_pct,
    )
    if not ok:
        return False, reason
    if not baseline:
        return False, "baseline missing"
    label = str(champion.get("label", champion.get("name", "")) or "").strip().lower()
    if label == "baseline":
        return False, "champion is baseline"
    champion_score = safe_float(champion.get(score_key, float("-inf")), float("-inf"))
    baseline_score = safe_float(baseline.get(score_key, float("-inf")), float("-inf"))
    if champion_score <= baseline_score:
        return False, "champion did not beat baseline under shared robustness score"
    return True, ""


def score_candidate_row(metrics_5y: Dict[str, Any], metrics_10y: Dict[str, Any], max_trades_per_year: float) -> float:
    primary = select_primary_metrics(metrics_5y, metrics_10y)
    secondary = metrics_5y if primary is metrics_10y else {}
    rejection = core_rejection_code(primary, metrics_5y=metrics_5y, metrics_10y=metrics_10y)
    if rejection is not None:
        return rejection_score(rejection)

    cagr_primary = metric(primary, "cagr_pct", 0.0)
    cagr_secondary = metric(secondary, "cagr_pct", 0.0)
    dd_primary = abs(metric(primary, "max_dd_pct", 0.0))
    trades_primary = metric(primary, "trades_per_year", 0.0)
    stitched_oos_cagr = metric(primary, "stitched_oos_cagr_pct", float("nan"))
    worst_24m_cagr = metric(primary, "worst_24m_cagr_pct", float("nan"))
    median_24m_cagr = metric(primary, "median_24m_cagr_pct", float("nan"))
    top2_year_share = metric(primary, "top2_year_pnl_share", float("nan"))

    score = cagr_primary * 1.4 + cagr_secondary * 0.8 - dd_primary * 0.45
    if trades_primary > max_trades_per_year:
        score -= (trades_primary - max_trades_per_year) * 0.8
    if math.isfinite(stitched_oos_cagr):
        score += stitched_oos_cagr * 0.9
    if math.isfinite(worst_24m_cagr):
        score += max(worst_24m_cagr, -25.0) * 0.2
    if math.isfinite(median_24m_cagr):
        score += median_24m_cagr * 0.25
    if math.isfinite(top2_year_share) and top2_year_share > TOP2_YEAR_SHARE_SOFT_CAP:
        score -= (top2_year_share - TOP2_YEAR_SHARE_SOFT_CAP) * 120.0
    return float(score)


def single_window_fitness(result: Dict[str, Any], *, max_drawdown_gate_pct: float = 25.0) -> float:
    metrics = pack_result_metrics(result)
    rejection = core_rejection_code(
        metrics,
        metrics_5y=metrics,
        metrics_10y={},
        max_drawdown_gate_pct=max_drawdown_gate_pct,
    )
    if rejection is not None:
        return 0.0

    cagr_pct = metric(metrics, "cagr_pct", 0.0)
    max_dd_pct = abs(metric(metrics, "max_dd_pct", 0.0))
    hit_rate_pct = metric(metrics, "hit_rate_pct", 0.0)
    stitched_oos_cagr_pct = metric(metrics, "stitched_oos_cagr_pct", float("nan"))
    worst_24m_cagr_pct = metric(metrics, "worst_24m_cagr_pct", float("nan"))
    median_24m_cagr_pct = metric(metrics, "median_24m_cagr_pct", float("nan"))
    top2_year_pnl_share = metric(metrics, "top2_year_pnl_share", float("nan"))

    score = cagr_pct * 12.0
    score -= max_dd_pct * 0.45
    score += hit_rate_pct * 0.10
    if math.isfinite(stitched_oos_cagr_pct):
        score += stitched_oos_cagr_pct * 1.4
    if math.isfinite(worst_24m_cagr_pct):
        score += max(worst_24m_cagr_pct, -25.0) * 0.20
    if math.isfinite(median_24m_cagr_pct):
        score += median_24m_cagr_pct * 0.25
    if math.isfinite(top2_year_pnl_share) and top2_year_pnl_share > TOP2_YEAR_SHARE_SOFT_CAP:
        score -= (top2_year_pnl_share - TOP2_YEAR_SHARE_SOFT_CAP) * 120.0
    return float(max(0.0, score))
