#!/usr/bin/env python3
"""
Forensic UI + functional smoke test for Apex Sniper Streamlit app.

Covers:
- Mode switching (Live Screener, Backtest, Simulator)
- Default universe assertions (expects RUSSELL3000 by default)
- Core action triggers in each mode
- Completion/error-state detection for Backtest
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


def _click_mode(page: Page, label: str) -> bool:
    candidates = [
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
    return False


def _select_root_text(page: Page) -> str:
    try:
        root = page.locator('[data-baseweb="select"]').first
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
                page.wait_for_timeout(1500),
                snap("audit_01_home.png"),
                f"title={page.title()}",
            )[-1],
        )

        check(
            "live_mode_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Live Screener"))
                if not _click_mode(page, "Live Screener")
                else None,
                page.wait_for_timeout(1200),
                snap("audit_02_live_mode.png"),
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Live: '{_select_root_text(page)}'"))
                if "RUSSELL3000" not in _select_root_text(page).upper()
                else f"default_universe={_select_root_text(page)}",
            )[-1],
        )

        def run_live_scan() -> str:
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
            "backtest_mode_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Backtest"))
                if not _click_mode(page, "Backtest")
                else None,
                page.wait_for_timeout(1400),
                snap("audit_04_backtest_mode.png"),
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Backtest: '{_select_root_text(page)}'"))
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
            "simulator_mode_default_universe",
            lambda: (
                (_ for _ in ()).throw(Exception("Unable to switch to Simulator"))
                if not _click_mode(page, "Simulator")
                else None,
                page.wait_for_timeout(1400),
                snap("audit_06_simulator_mode.png"),
                (_ for _ in ()).throw(Exception(f"Unexpected default universe in Simulator: '{_select_root_text(page)}'"))
                if "RUSSELL3000" not in _select_root_text(page).upper()
                else f"default_universe={_select_root_text(page)}",
            )[-1],
        )

        def simulator_reset_guardrail() -> str:
            reset_btn = page.get_by_role(
                "button",
                name=re.compile(r"Emergency System Reset", re.I),
            ).first
            if reset_btn.count() > 0:
                reset_btn.click(timeout=10000)
            ok, _seen = _wait_for_any(page, [r"Confirm full simulator reset"], timeout_sec=8)
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
