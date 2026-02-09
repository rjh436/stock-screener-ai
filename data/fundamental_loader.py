from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import polars as pl
except Exception:  # pragma: no cover - dependency/runtime guard
    pl = None  # type: ignore[assignment]

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - dependency/runtime guard
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    import edgar
except Exception:  # pragma: no cover - dependency/runtime guard
    edgar = None  # type: ignore[assignment]


REVENUE_CONCEPTS: Tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)

NET_INCOME_CONCEPTS: Tuple[str, ...] = (
    "NetIncomeLoss",
    "ProfitLoss",
)

EPS_CONCEPTS: Tuple[str, ...] = (
    "EarningsPerShareDiluted",
    "DilutedEarningsPerShare",
    "EarningsPerShareBasicAndDiluted",
    "EarningsPerShareBasic",
)


@dataclass(frozen=True)
class LoaderConfig:
    output_dir: Path
    identity: str
    max_workers: int
    sec_rate_limit: int
    max_filings_per_ticker: int = 16
    max_tasks_per_child: int = 8
    request_pause_sec: float = 0.12
    overwrite: bool = False
    verbose: bool = True


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            continue
    try:
        dt2 = datetime.fromisoformat(text)
        if dt2.tzinfo is not None:
            dt2 = dt2.astimezone(timezone.utc).replace(tzinfo=None)
        return dt2
    except Exception:
        return None


def _first_present(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in candidates:
        if key in mapping:
            return mapping[key]
        lk = key.lower()
        if lk in lower:
            return lower[lk]
    return None


def _normalize_ticker(raw: Any) -> Optional[str]:
    sym = str(raw or "").strip().upper()
    if not sym:
        return None
    sym = sym.replace(".", "-").replace("/", "-")
    if sym in {"-", "NAN", "NONE", "NULL"}:
        return None
    if not re.fullmatch(r"[A-Z0-9\-]{1,10}", sym):
        return None
    return sym


def _read_tickers(path: Path) -> List[str]:
    if not path.exists():
        return []

    if path.suffix.lower() in {".json", ".txt", ".lst"}:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return []

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except Exception:
                return []
            if isinstance(payload, list):
                raw_symbols = payload
            elif isinstance(payload, dict):
                raw_symbols = payload.get("symbols") or payload.get("tickers") or []
            else:
                raw_symbols = []
        else:
            raw_symbols = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        raw_symbols = []

    deduped: List[str] = []
    seen = set()
    for raw in raw_symbols:
        sym = _normalize_ticker(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        deduped.append(sym)
    return deduped


def _existing_tickers(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    out: set[str] = set()
    try:
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("ticker="):
                continue
            ticker = _normalize_ticker(name.split("=", 1)[1])
            if ticker:
                out.add(ticker)
    except Exception:
        return set()
    return out


def _filings_to_list(filings_obj: Any, limit: int) -> List[Any]:
    if filings_obj is None:
        return []

    out: List[Any] = []

    # edgartools `Filings` objects are iterable and often expose `head()`.
    if hasattr(filings_obj, "head"):
        try:
            head_obj = filings_obj.head(limit)
            for filing in head_obj:
                out.append(filing)
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception:
            pass

    try:
        for filing in filings_obj:
            out.append(filing)
            if len(out) >= limit:
                break
    except Exception:
        pass

    if out:
        return out

    # Last resort: list-like indexing.
    try:
        n = len(filings_obj)
    except Exception:
        n = 0
    for idx in range(min(limit, n)):
        try:
            out.append(filings_obj[idx])
        except Exception:
            continue

    return out


def _to_records(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if pl is not None and isinstance(obj, pl.DataFrame):
        return obj.to_dicts()

    if hasattr(obj, "to_dataframe"):
        try:
            obj = obj.to_dataframe()
        except Exception:
            return []

    if hasattr(obj, "to_dict"):
        try:
            records = obj.to_dict("records")
            if isinstance(records, list):
                return [x for x in records if isinstance(x, dict)]
        except Exception:
            pass

    return []


def _query_concept_records(xbrl_obj: Any, concepts: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for concept in concepts:
        # Query API (preferred: lower memory than pulling full fact table).
        if hasattr(xbrl_obj, "query"):
            try:
                q = xbrl_obj.query()
                if hasattr(q, "by_statement_type"):
                    q = q.by_statement_type("IncomeStatement")
                if hasattr(q, "by_period_type"):
                    q = q.by_period_type("duration")
                if hasattr(q, "by_concept"):
                    q = q.by_concept(concept)
                part = _to_records(q)
                if part:
                    rows.extend(part)
                    continue
            except Exception:
                pass

        # Fallback API.
        if hasattr(xbrl_obj, "query_facts"):
            try:
                part = _to_records(xbrl_obj.query_facts(concept=concept))
                if part:
                    rows.extend(part)
                    continue
            except Exception:
                pass

    if rows:
        return rows

    # Last-resort full fact table scan.
    try:
        facts = getattr(xbrl_obj, "facts", None)
        full_rows = _to_records(facts)
    except Exception:
        full_rows = []

    concept_set = {c.lower() for c in concepts}
    filtered: List[Dict[str, Any]] = []
    for row in full_rows:
        row_concept = str(_first_present(row, ("concept", "name", "element")) or "").lower()
        if not row_concept:
            continue
        if any(c in row_concept for c in concept_set):
            filtered.append(row)
    return filtered


def _pick_fact_value(
    rows: Sequence[Mapping[str, Any]],
    filing_report_date: Optional[datetime],
) -> Tuple[Optional[float], Optional[datetime]]:
    if not rows:
        return None, None

    processed: List[Tuple[float, Optional[datetime], int, int]] = []
    # tuple = (value, period_end, non_consolidated_penalty, distance_days)

    for row in rows:
        val = _safe_float(_first_present(row, ("value", "amount", "fact_value", "numeric_value", "val")))
        if val is None:
            continue

        period_end = _safe_ts(
            _first_present(
                row,
                (
                    "period_end",
                    "end",
                    "end_date",
                    "period",
                    "date",
                    "fiscal_period_end",
                ),
            )
        )
        period_start = _safe_ts(
            _first_present(
                row,
                (
                    "period_start",
                    "start",
                    "start_date",
                    "fiscal_period_start",
                ),
            )
        )

        duration_penalty = 0
        if period_start is not None and period_end is not None:
            duration_days = (period_end - period_start).days
            # Quarterly duration range heuristic.
            if duration_days < 60 or duration_days > 130:
                duration_penalty = 1

        dim_val = _first_present(row, ("dimensions", "segment", "dimensional_values", "axis"))
        dim_text = str(dim_val or "").strip().lower()
        non_consolidated_penalty = 0
        if dim_text and dim_text not in {"{}", "none", "nan", "null"}:
            non_consolidated_penalty = 1

        if filing_report_date is not None and period_end is not None:
            distance = abs((period_end.date() - filing_report_date.date()).days)
        else:
            distance = 999_999

        processed.append((val, period_end, duration_penalty + non_consolidated_penalty, distance))

    if not processed:
        return None, None

    # Prefer consolidated quarterly values closest to report date.
    processed.sort(key=lambda x: (x[2], x[3]))
    value, period_end, _, _ = processed[0]
    return value, period_end


def _extract_quarter_record(filing: Any, request_pause_sec: float) -> Optional[Dict[str, Any]]:
    try:
        form = str(getattr(filing, "form", "") or "")
        filing_date = _safe_ts(getattr(filing, "filing_date", None))
        period_of_report = _safe_ts(getattr(filing, "period_of_report", None))
        report_date = period_of_report or filing_date

        xbrl_obj = filing.xbrl()
        if xbrl_obj is None:
            return None

        rev_rows = _query_concept_records(xbrl_obj, REVENUE_CONCEPTS)
        ni_rows = _query_concept_records(xbrl_obj, NET_INCOME_CONCEPTS)
        eps_rows = _query_concept_records(xbrl_obj, EPS_CONCEPTS)

        revenue, rev_end = _pick_fact_value(rev_rows, report_date)
        net_income, ni_end = _pick_fact_value(ni_rows, report_date)
        eps, eps_end = _pick_fact_value(eps_rows, report_date)

        quarter_end = rev_end or ni_end or eps_end or report_date
        if quarter_end is None:
            return None

        out = {
            "quarter_end": quarter_end.date().isoformat(),
            "filing_date": filing_date.date().isoformat() if filing_date else None,
            "form": form,
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
        }

        # Release XBRL objects aggressively to control process memory growth.
        del xbrl_obj
        gc.collect()
        if request_pause_sec > 0:
            time.sleep(request_pause_sec)
        return out
    except Exception:
        return None


def _build_company(symbol: str) -> Any:
    if edgar is None:
        raise RuntimeError("edgartools is not installed. Install with: pip install edgartools")

    company_cls = getattr(edgar, "Company", None)
    if company_cls is not None:
        return company_cls(symbol)

    entity_cls = getattr(edgar, "Entity", None)
    if entity_cls is not None:
        return entity_cls(symbol)

    raise RuntimeError("Unable to find edgar Company/Entity class in installed edgartools version")


def _worker_initializer(identity: str, per_worker_rate_limit: int) -> None:
    if edgar is None:
        return

    set_identity = getattr(edgar, "set_identity", None)
    if callable(set_identity):
        try:
            set_identity(identity)
        except Exception:
            pass

    set_rate_limit = getattr(edgar, "set_rate_limit", None)
    if callable(set_rate_limit):
        try:
            set_rate_limit(max(1, int(per_worker_rate_limit)))
        except Exception:
            pass


def _process_ticker(
    symbol: str,
    max_filings_per_ticker: int,
    request_pause_sec: float,
) -> Dict[str, Any]:
    try:
        company = _build_company(symbol)

        try:
            filings_obj = company.get_filings(
                form=["10-Q", "10-K"],
                is_xbrl=True,
                amendments=False,
            )
        except TypeError:
            try:
                filings_obj = company.get_filings(
                    form=["10-Q", "10-K"],
                    is_xbrl=True,
                )
            except TypeError:
                filings_obj = company.get_filings(form=["10-Q", "10-K"])
        filings = _filings_to_list(filings_obj, max_filings_per_ticker)
        if not filings:
            # Fallback path for older APIs that do not accept `is_xbrl`.
            try:
                filings_obj = company.get_filings(form=["10-Q", "10-K"], amendments=False)
            except TypeError:
                filings_obj = company.get_filings(form=["10-Q", "10-K"])
            filings = _filings_to_list(filings_obj, max_filings_per_ticker)

        rows: List[Dict[str, Any]] = []
        for filing in filings:
            record = _extract_quarter_record(filing, request_pause_sec=request_pause_sec)
            if record is not None:
                rows.append(record)

        if not rows:
            return {"ticker": symbol, "rows": [], "error": "no_quarterly_records"}

        df = pl.DataFrame(rows)
        if df.is_empty():
            return {"ticker": symbol, "rows": [], "error": "empty_frame"}

        # Normalize schema to avoid mixed-type errors from sparse XBRL payloads.
        if "quarter_end" in df.columns:
            df = df.with_columns(pl.col("quarter_end").cast(pl.Utf8, strict=False))
        if "filing_date" in df.columns:
            df = df.with_columns(pl.col("filing_date").cast(pl.Utf8, strict=False))
        if "form" in df.columns:
            df = df.with_columns(pl.col("form").cast(pl.Utf8, strict=False))
        for metric in ("revenue", "net_income", "eps"):
            if metric not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))
            else:
                df = df.with_columns(pl.col(metric).cast(pl.Float64, strict=False))

        # De-duplicate by quarter end, keeping the newest filing row.
        if "filing_date" in df.columns:
            df = df.sort(["quarter_end", "filing_date"], descending=[False, False])
        else:
            df = df.sort("quarter_end")
        df = df.unique(subset=["quarter_end"], keep="last", maintain_order=True)
        df = df.sort("quarter_end")

        for metric in ("revenue", "net_income", "eps"):
            metric_col = pl.col(metric).cast(pl.Float64, strict=False)
            prior_q = metric_col.shift(1)
            prior_y = metric_col.shift(4)

            qoq_expr = pl.when(
                prior_q.is_not_null() & (prior_q.abs() > 1e-12)
            ).then(((metric_col / prior_q) - 1.0) * 100.0).otherwise(None)

            yoy_expr = pl.when(
                prior_y.is_not_null() & (prior_y.abs() > 1e-12)
            ).then(((metric_col / prior_y) - 1.0) * 100.0).otherwise(None)

            df = df.with_columns(
                qoq_expr.alias(f"{metric}_qoq_growth_pct"),
                yoy_expr.alias(f"{metric}_yoy_growth_pct"),
            )

            df = df.with_columns(
                (
                    pl.col(f"{metric}_qoq_growth_pct")
                    - pl.col(f"{metric}_qoq_growth_pct").shift(1)
                ).alias(f"{metric}_qoq_accel_pct")
            )

        df = df.with_columns(
            pl.lit(symbol).alias("ticker"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("updated_at_utc"),
            pl.lit("sec_edgar_xbrl").alias("data_source"),
        )

        return {"ticker": symbol, "rows": df.to_dicts(), "error": None}
    except Exception as exc:
        return {"ticker": symbol, "rows": [], "error": str(exc)}


def _partition_path(root: Path, ticker: str) -> Path:
    return root / f"ticker={ticker}" / "fundamentals.parquet"


def _write_ticker_partition(root: Path, ticker: str, rows: Sequence[Mapping[str, Any]], overwrite: bool) -> int:
    if not rows:
        return 0
    if pl is None or pa is None or pq is None:
        raise RuntimeError("polars and pyarrow are required for parquet persistence")

    root.mkdir(parents=True, exist_ok=True)
    path = _partition_path(root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pl.DataFrame(rows)
    if not overwrite and path.exists():
        try:
            old_df = pl.read_parquet(path)
            combined = pl.concat([old_df, new_df], how="diagonal_relaxed")
        except Exception:
            combined = new_df
    else:
        combined = new_df

    if "quarter_end" in combined.columns:
        combined = combined.sort("quarter_end")
        combined = combined.unique(subset=["quarter_end"], keep="last", maintain_order=True)

    tmp_path = path.with_suffix(".tmp.parquet")
    table: pa.Table = combined.to_arrow()
    pq.write_table(
        table,
        tmp_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    tmp_path.replace(path)
    return combined.height


def refresh_fundamentals(
    tickers: Sequence[str],
    config: LoaderConfig,
) -> Dict[str, Any]:
    if edgar is None or pl is None or pa is None or pq is None:
        raise RuntimeError(
            "Required packages are missing (edgartools, polars, pyarrow). "
            "Install dependencies: pip install edgartools polars pyarrow"
        )

    if not config.identity.strip():
        raise ValueError(
            "SEC identity is required. Pass --identity 'Name email@example.com' "
            "or set SEC_EDGAR_IDENTITY env var."
        )

    normalized: List[str] = []
    seen = set()
    for raw in tickers:
        sym = _normalize_ticker(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)

    if not normalized:
        return {"requested": 0, "processed": 0, "written": 0, "errors": {}}

    cpu_count = os.cpu_count() or 4
    workers = max(1, min(config.max_workers, cpu_count))

    # Respect SEC guidance while keeping throughput high.
    if config.sec_rate_limit > 0:
        workers = min(workers, max(1, config.sec_rate_limit))
        per_worker_limit = max(1, config.sec_rate_limit // workers)
    else:
        per_worker_limit = 1

    if config.verbose:
        print(
            "[fundamental_loader] "
            f"tickers={len(normalized)} workers={workers} "
            f"sec_rate_limit={config.sec_rate_limit}/s per_worker={per_worker_limit}/s"
        )

    errors: Dict[str, str] = {}
    written = 0
    processed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initializer,
        initargs=(config.identity, per_worker_limit),
        max_tasks_per_child=max(1, int(config.max_tasks_per_child)),
    ) as pool:
        futures = {
            pool.submit(
                _process_ticker,
                ticker,
                int(config.max_filings_per_ticker),
                float(config.request_pause_sec),
            ): ticker
            for ticker in normalized
        }

        for future in as_completed(futures):
            ticker = futures[future]
            processed += 1
            try:
                payload = future.result()
            except Exception as exc:
                errors[ticker] = str(exc)
                continue

            err = str(payload.get("error") or "")
            rows = payload.get("rows") or []
            if err and not rows:
                errors[ticker] = err
                continue

            count = _write_ticker_partition(
                root=config.output_dir,
                ticker=ticker,
                rows=rows,
                overwrite=config.overwrite,
            )
            if count > 0:
                written += 1

            if config.verbose and processed % 100 == 0:
                print(
                    "[fundamental_loader] "
                    f"processed={processed}/{len(normalized)} written={written} errors={len(errors)}"
                )

    summary = {
        "requested": len(normalized),
        "processed": processed,
        "written": written,
        "errors": errors,
        "output_dir": str(config.output_dir),
    }

    if config.verbose:
        print(f"[fundamental_loader] done: {summary}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load SEC EDGAR quarterly income statement fundamentals into partitioned parquet."
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        default="data/cache_indices/russell3000_iwv.json",
        help="JSON/TXT file containing ticker symbols (default: Russell 3000 cache)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker override (e.g. NVDA,SMCI,AAPL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of tickers for partial refresh/debugging.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fundamentals/edgar_income",
        help="Root parquet partition directory.",
    )
    parser.add_argument(
        "--identity",
        type=str,
        default=os.getenv("SEC_EDGAR_IDENTITY", ""),
        help="SEC identity string: 'Name email@example.com'",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(os.cpu_count() or 4, 12)),
        help="Process count for XBRL parsing.",
    )
    parser.add_argument(
        "--sec-rate-limit",
        type=int,
        default=_env_int("SEC_EDGAR_RATE_LIMIT", 8),
        help="Aggregate SEC request budget per second.",
    )
    parser.add_argument(
        "--max-filings-per-ticker",
        type=int,
        default=16,
        help="How many 10-Q/10-K filings to inspect per ticker.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=8,
        help="Recycle worker processes to prevent long-run memory leakage.",
    )
    parser.add_argument(
        "--request-pause-ms",
        type=int,
        default=120,
        help="Small per-filing delay (ms) to smooth request bursts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite each ticker partition instead of merge+dedupe.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip symbols that already have parquet partitions (default: true).",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split work into N shards for resumable/distributed runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to process when --shard-count > 1.",
    )
    return parser.parse_args(argv)


def _resolve_cli_tickers(args: argparse.Namespace) -> List[str]:
    if args.tickers:
        raw = [x.strip() for x in str(args.tickers).split(",") if x.strip()]
        tickers: List[str] = []
        seen = set()
        for t in raw:
            sym = _normalize_ticker(t)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            tickers.append(sym)
        return tickers

    source = Path(str(args.universe_file))
    return _read_tickers(source)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tickers = _resolve_cli_tickers(args)
    if args.limit and args.limit > 0:
        tickers = tickers[: int(args.limit)]

    output_dir = Path(str(args.output_dir))
    if bool(args.skip_existing):
        already = _existing_tickers(output_dir)
        if already:
            tickers = [t for t in tickers if t not in already]
            if not bool(args.quiet):
                print(f"[fundamental_loader] skip-existing filtered {len(already)} already-loaded tickers")

    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index: {shard_index} (shard_count={shard_count})")
    if shard_count > 1:
        tickers = [t for idx, t in enumerate(tickers) if (idx % shard_count) == shard_index]
        if not bool(args.quiet):
            print(f"[fundamental_loader] shard {shard_index+1}/{shard_count} has {len(tickers)} tickers")

    cfg = LoaderConfig(
        output_dir=output_dir,
        identity=str(args.identity or ""),
        max_workers=int(args.max_workers),
        sec_rate_limit=int(args.sec_rate_limit),
        max_filings_per_ticker=int(args.max_filings_per_ticker),
        max_tasks_per_child=int(args.max_tasks_per_child),
        request_pause_sec=max(0.0, float(args.request_pause_ms) / 1000.0),
        overwrite=bool(args.overwrite),
        verbose=not bool(args.quiet),
    )

    summary = refresh_fundamentals(tickers=tickers, config=cfg)
    if summary["errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
