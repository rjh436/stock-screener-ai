#!/usr/bin/env python3
"""
Forensic UI + functional smoke test for Apex Sniper Streamlit app.

Covers:
- Mode switching (Live Screener, Backtest, Simulator)
- Benchmark workspace readiness (ETF / Hybrid / Stock)
- Stock Research workspace universe defaults and core actions
- Generic action triggers in research mode
- Screenshot artifacts for visual review
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

from playwright.sync_api import Page, sync_playwright


def _safe_text(page: Page, selector: str) -> str:
    try:
        return (page.locator(selector).first.inner_text() or "").strip()
    except Exception:
        return ""


def _click_option(page: Page, label: str) -> bool:
    started = time.time()
    while time.time() - started < 20:
        candidates = [
            page.locator('[data-testid="stSidebar"] label').filter(has_text=label).first,
            page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).first,
            page.locator("label").filter(has_text=label).first,
            page.get_by_text(label, exact=True).first,
        ]
        for loc in candidates:
            try:
                if loc.count() > 0:
                    loc.click(timeout=6000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(500)
    return False


def _click_mode(page: Page, label: str) -> bool:
    return _click_option(page, label)


def _click_workspace(page: Page, label: str) -> bool:
    return _click_option(page, label)


def _select_root_text(page: Page) -> str:
    try:
        all_selects = page.locator('[data-baseweb="select"]')
        sidebar_selects = page.locator('[data-testid="stSidebar"] [data-baseweb="select"]')
        total = all_selects.count()
        sidebar_total = sidebar_selects.count()
        if total == 0:
            return ""
        root = all_selects.nth(sidebar_total) if total > sidebar_total else all_selects.first
        if root.count() == 0:
            return ""
        return (root.inner_text() or "").strip()
    except Exception:
        return ""


def _wait_for_any(page: Page, patterns: list[str], timeout_sec: int) -> tuple[bool, str]:
    started = time.time()
    while time.time() - started < timeout_sec:
        page.wait_for_timeout(1800)
        body = page.text_content("body") or ""
        for pat in patterns:
            if re.search(pat, body, flags=re.I):
                return True, pat
    return False, ""


def _wait_for_button(page: Page, pattern: str, timeout_sec: int) -> bool:
    started = time.time()
    while time.time() - started < timeout_sec:
        try:
            btn = page.get_by_role("button", name=re.compile(pattern, re.I)).first
            if btn.count() > 0:
                return True
        except Exception:
            pass
        page.wait_for_timeout(1000)
    return False


def _wait_for_workspace_ready(page: Page, patterns: list[str], timeout_sec: int) -> bool:
    ok, _ = _wait_for_any(page, patterns, timeout_sec=timeout_sec)
    return ok


def run_audit(base_url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "checks": [],
        "console_errors": [],
        "page_errors": [],
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1720, "height": 1060})
        page = ctx.new_page()

        page.on("console", lambda msg: report["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: report["page_errors"].append(str(exc)))

        def snap(name: str) -> None:
            page.screenshot(path=str(out_dir / name), full_page=True)

        def check(name: str, fn: Callable[[], str]) -> bool:
            started = time.time()
            row = {"name": name, "status": "pass", "details": "", "elapsed_sec": 0.0}
            try:
                detail = fn()
                if detail:
                    row["details"] = str(detail)
            except Exception as exc:
                row["status"] = "fail"
                row["details"] = str(exc)
            row["elapsed_sec"] = round(time.time() - started, 2)
            report["checks"].append(row)
            return row["status"] == "pass"

        check(
            "open_app",
            lambda: (
                page.goto(base_url, wait_until="domcontentloaded", timeout=45000),
                (_ for _ in ()).throw(Exception("Streamlit shell loaded but benchmark workspace did not render"))
                if not _wait_for_workspace_ready(page, [r"Apex Sniper", r"Select Mode", r"Strategy Workspace"], timeout_sec=30)
                else None,
                page.wait_for_timeout(1000),
                snap("audit_01_home.png"),
                f"title={page.title()}",
            )[-1],
        )

        check(
            "default_workspace_is_hybrid",
            lambda: (
                (_ for _ in ()).throw(Exception("Hybrid benchmark is not the default landing workspace"))
                if not _wait_for_any(
                    page,
                    [r"Hybrid Benchmark", r"Refresh Hybrid Snapshot", r"Current Target Allocation"],
                    timeout_sec=20,
                )[0]
                else "default_workspace=hybrid",
            )[-1],
        )

        check(
            "live_etf_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Live Screener"))
                if not _click_mode(page, "Live Screener")
                else None,
                (_ for _ in ()).throw(Exception("Unable to select ETF Benchmark workspace"))
                if not _click_workspace(page, "ETF Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("ETF live workspace did not finish loading"))
                if not _wait_for_button(page, r"Refresh ETF Snapshot", timeout_sec=20)
                else None,
                snap("audit_02_live_mode.png"),
                (_ for _ in ()).throw(Exception("ETF live workspace missing refresh action"))
                if page.get_by_role("button", name=re.compile("Refresh ETF Snapshot", re.I)).count() == 0
                else "etf_live_workspace_ready",
            )[-1],
        )

        check(
            "live_hybrid_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Hybrid Benchmark workspace"))
                if not _click_workspace(page, "Hybrid Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Hybrid live workspace did not finish loading"))
                if not _wait_for_button(page, r"Refresh Hybrid Snapshot", timeout_sec=20)
                else None,
                (_ for _ in ()).throw(Exception("Hybrid live workspace missing refresh action"))
                if page.get_by_role("button", name=re.compile("Refresh Hybrid Snapshot", re.I)).count() == 0
                else "hybrid_live_workspace_ready",
            )[-1],
        )

        check(
            "live_stock_benchmark_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Benchmark workspace"))
                if not _click_workspace(page, "Stock Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Stock benchmark live workspace did not finish loading"))
                if not _wait_for_button(page, r"Refresh Stock Snapshot", timeout_sec=20)
                else None,
                (_ for _ in ()).throw(Exception("Stock benchmark live workspace missing refresh action"))
                if page.get_by_role("button", name=re.compile("Refresh Stock Snapshot", re.I)).count() == 0
                else "stock_benchmark_live_workspace_ready",
            )[-1],
        )

        check(
            "live_stock_research_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Research workspace"))
                if not _click_workspace(page, "Stock Research")
                else None,
                (_ for _ in ()).throw(Exception("Live stock research workspace did not finish loading after retry"))
                if not (
                    _wait_for_workspace_ready(page, [r"RUN SCAN", r"Daily Opportunity Scanner"], timeout_sec=20)
                    or (_click_workspace(page, "Stock Research") and _wait_for_workspace_ready(page, [r"RUN SCAN", r"Daily Opportunity Scanner"], timeout_sec=20))
                )
                else None,
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Live research: '{_select_root_text(page)}'"))
                if "RUSSELL3000" not in _select_root_text(page).upper()
                else f"default_universe={_select_root_text(page)}",
            )[-1],
        )

        def run_live_scan() -> str:
            run_btn = page.get_by_role("button", name=re.compile("RUN SCAN", re.I)).first
            if run_btn.count() == 0:
                _click_workspace(page, "Stock Research")
                _wait_for_workspace_ready(page, [r"RUN SCAN", r"Daily Opportunity Scanner"], timeout_sec=20)
                run_btn = page.get_by_role("button", name=re.compile("RUN SCAN", re.I)).first
            if run_btn.count() == 0:
                raise Exception("RUN SCAN button not found")
            run_btn.click(timeout=8000)
            ok, seen = _wait_for_any(
                page,
                [
                    r"Resolving RUSSELL3000 constituents",
                    r"Loaded .* symbols for RUSSELL3000",
                    r"20%\s*Complete",
                    r"Simulating|Analyzing symbols",
                ],
                timeout_sec=45,
            )
            snap("audit_03_live_after_run.png")
            if not ok:
                raise Exception("Live scan did not show expected progress text")
            return f"progress_signal={seen}"

        check("live_scan_progress", run_live_scan)

        check(
            "backtest_etf_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Backtest"))
                if not _click_mode(page, "Backtest")
                else None,
                (_ for _ in ()).throw(Exception("Unable to select ETF Benchmark workspace in Backtest"))
                if not _click_workspace(page, "ETF Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("ETF benchmark backtest view did not load"))
                if not _wait_for_any(page, [r"ETF Benchmark Lab", r"Run ETF Benchmark"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1400),
                snap("audit_04_backtest_mode.png"),
                (_ for _ in ()).throw(Exception("ETF benchmark backtest button not found"))
                if page.get_by_role("button", name=re.compile("Run ETF Benchmark", re.I)).count() == 0
                else "etf_backtest_ready",
            )[-1],
        )

        check(
            "backtest_hybrid_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Hybrid Benchmark workspace in Backtest"))
                if not _click_workspace(page, "Hybrid Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Hybrid benchmark backtest view did not load"))
                if not _wait_for_any(page, [r"Hybrid Benchmark Lab", r"Run Hybrid Benchmark"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1200),
                (_ for _ in ()).throw(Exception("Hybrid benchmark backtest button not found"))
                if page.get_by_role("button", name=re.compile("Run Hybrid Benchmark", re.I)).count() == 0
                else "hybrid_backtest_ready",
            )[-1],
        )

        check(
            "backtest_stock_benchmark_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Benchmark workspace in Backtest"))
                if not _click_workspace(page, "Stock Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Stock benchmark backtest view did not load"))
                if not _wait_for_any(page, [r"Stock Benchmark Lab", r"Run Stock Benchmark"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1200),
                (_ for _ in ()).throw(Exception("Stock benchmark backtest button not found"))
                if page.get_by_role("button", name=re.compile("Run Stock Benchmark", re.I)).count() == 0
                else "stock_benchmark_backtest_ready",
            )[-1],
        )

        check(
            "backtest_stock_research_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Research workspace in Backtest"))
                if not _click_workspace(page, "Stock Research")
                else None,
                (_ for _ in ()).throw(Exception("Backtest stock research workspace did not finish loading"))
                if not _wait_for_workspace_ready(page, [r"RUN BACKTEST", r"Stock Strategy Research Lab"], timeout_sec=20)
                else None,
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Backtest research: '{_select_root_text(page)}'"))
                if "RUSSELL3000" not in _select_root_text(page).upper()
                else f"default_universe={_select_root_text(page)}",
            )[-1],
        )

        def run_backtest() -> str:
            btn_1y = page.get_by_role("button", name=re.compile(r"^1 Year$", re.I)).first
            if btn_1y.count() > 0:
                btn_1y.click(timeout=7000)
                page.wait_for_timeout(800)

            run_btn = page.get_by_role("button", name=re.compile("RUN BACKTEST", re.I)).first
            if run_btn.count() == 0:
                raise Exception("RUN BACKTEST button not found")
            run_btn.click(timeout=10000)

            ok, seen = _wait_for_any(
                page,
                [
                    r"Simulating",
                    r"Stage:",
                    r"CAGR",
                    r"Accuracy gate blocked",
                    r"Backtest dataset is unavailable or empty",
                    r"Total Trades",
                ],
                timeout_sec=210,
            )
            snap("audit_05_backtest_after_run.png")
            if not ok:
                raise Exception("Backtest did not reach expected progress or result state")
            return f"state_signal={seen}"

        check("backtest_run_state", run_backtest)

        check(
            "simulator_etf_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Simulator"))
                if not _click_mode(page, "Simulator")
                else None,
                (_ for _ in ()).throw(Exception("Unable to select ETF Benchmark workspace in Simulator"))
                if not _click_workspace(page, "ETF Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("ETF simulator view did not load"))
                if not _wait_for_any(page, [r"ETF Paper Allocator", r"Refresh ETF Recommendation"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1400),
                snap("audit_06_simulator_mode.png"),
                (_ for _ in ()).throw(Exception("ETF simulator adopt action not found after snapshot load"))
                if not _wait_for_button(page, r"Adopt Latest ETF Allocation", timeout_sec=30)
                else "etf_simulator_ready",
            )[-1],
        )

        check(
            "simulator_hybrid_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Hybrid Benchmark workspace in Simulator"))
                if not _click_workspace(page, "Hybrid Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Hybrid simulator view did not load"))
                if not _wait_for_any(page, [r"Hybrid Benchmark Paper Allocator", r"Refresh Hybrid Recommendation"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1200),
                (_ for _ in ()).throw(Exception("Hybrid simulator controls did not stabilize"))
                if not _wait_for_workspace_ready(
                    page,
                    [r"Adopt Latest Hybrid Allocation", r"No hybrid recommendation is loaded yet"],
                    timeout_sec=30,
                )
                else "hybrid_simulator_ready",
            )[-1],
        )

        check(
            "simulator_stock_benchmark_workspace_ready",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Benchmark workspace in Simulator"))
                if not _click_workspace(page, "Stock Benchmark")
                else None,
                (_ for _ in ()).throw(Exception("Stock benchmark simulator view did not load"))
                if not _wait_for_any(page, [r"Stock Benchmark Paper Allocator", r"Refresh Stock Recommendation"], timeout_sec=15)[0]
                else None,
                page.wait_for_timeout(1200),
                (_ for _ in ()).throw(Exception("Stock benchmark simulator controls did not stabilize"))
                if not _wait_for_workspace_ready(
                    page,
                    [r"Adopt Latest Stock Allocation", r"No stock recommendation is loaded yet"],
                    timeout_sec=30,
                )
                else "stock_benchmark_simulator_ready",
            )[-1],
        )

        check(
            "simulator_stock_research_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to select Stock Research workspace in Simulator"))
                if not _click_workspace(page, "Stock Research")
                else None,
                (_ for _ in ()).throw(Exception("Simulator stock research workspace did not finish loading"))
                if not _wait_for_workspace_ready(page, [r"PHASE 1: Scan for New Entries", r"Paper Trader"], timeout_sec=20)
                else None,
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Simulator research: '{_select_root_text(page)}'"))
                if "RUSSELL3000" not in _select_root_text(page).upper()
                else f"default_universe={_select_root_text(page)}",
            )[-1],
        )

        def simulator_reset_guardrail() -> str:
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(1000)
            if not _wait_for_workspace_ready(page, [r"Emergency System Reset", r"Paper Trader"], timeout_sec=20):
                raise Exception("Simulator reset controls did not finish loading")
            reset_btn = page.locator("button").filter(has_text=re.compile(r"Emergency System Reset", re.I)).first
            if reset_btn.count() == 0:
                reset_btn = page.get_by_text(re.compile(r"Emergency System Reset", re.I)).first
            if reset_btn.count() > 0:
                try:
                    reset_btn.scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    pass
                reset_btn.click(timeout=10000)
            ok, _seen = _wait_for_any(page, [r"Confirm full simulator reset"], timeout_sec=20)
            if not ok:
                if reset_btn.count() == 0:
                    raise Exception("Emergency reset button not found")
                raise Exception("Reset confirmation guardrail did not appear")
            cancel_btn = page.get_by_role("button", name=re.compile(r"^Cancel$", re.I)).first
            if cancel_btn.count() == 0:
                raise Exception("Reset cancel button not found")
            cancel_btn.click(timeout=8000)
            page.wait_for_timeout(1000)
            snap("audit_08_simulator_reset_guardrail.png")
            return "two_step_confirmation_present"

        check("simulator_reset_guardrail", simulator_reset_guardrail)

        def run_sim_phase1() -> str:
            btn = page.get_by_role(
                "button",
                name=re.compile(r"PHASE 1: Scan for New Entries", re.I),
            ).first
            if btn.count() == 0:
                raise Exception("Simulator PHASE 1 button not found")
            btn.click(timeout=10000)
            ok, seen = _wait_for_any(
                page,
                [
                    r"Resolving RUSSELL3000 constituents",
                    r"Executing Scan & Governor",
                    r"Simulation Complete",
                    r"Scan Complete\. Orders Queued",
                ],
                timeout_sec=70,
            )
            snap("audit_07_simulator_after_phase1.png")
            if not ok:
                raise Exception("Simulator PHASE 1 did not show expected status")
            return f"status_signal={seen}"

        check("simulator_phase1_status", run_sim_phase1)

        browser.close()

    report["summary"] = {
        "passed": sum(1 for c in report["checks"] if c["status"] == "pass"),
        "failed": sum(1 for c in report["checks"] if c["status"] == "fail"),
        "console_error_count": len(report["console_errors"]),
        "page_error_count": len(report["page_errors"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic app UX + functional browser audit")
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--out-dir", default="output/playwright")
    args = parser.parse_args()

    report = run_audit(args.base_url, Path(args.out_dir))
    out_path = Path(args.out_dir) / "app_ux_functional_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))
    print(json.dumps(report["summary"]))
    for row in report["checks"]:
        print(f"{row['status'].upper()} {row['name']}: {row['details']}")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
