#!/usr/bin/env python3
"""
Playwright smoke test for Streamlit Live Screener and Simulator flows.

Usage:
  ./.venv/bin/python tools/test_live_simulator_e2e.py \
      --base-url http://localhost:8501 \
      --universe RUSSELL3000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def _click_mode(page, label: str) -> bool:
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
                    loc.click(timeout=5000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(500)
    return False


def _click_workspace(page, label: str) -> bool:
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
                    loc.click(timeout=5000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(500)
    return False


def _select_universe(page, universe: str) -> bool:
    try:
        all_selects = page.locator('[data-baseweb="select"]')
        sidebar_selects = page.locator('[data-testid="stSidebar"] [data-baseweb="select"]')
        total = all_selects.count()
        sidebar_total = sidebar_selects.count()
        if total == 0:
            return False
        select_root = all_selects.nth(sidebar_total) if total > sidebar_total else all_selects.first
        if select_root.count() == 0:
            return False
        select_root.click(timeout=5000)
        option = page.get_by_text(universe, exact=True).first
        option.wait_for(timeout=8000)
        option.click(timeout=5000)
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)
        return True
    except Exception:
        return False


def _wait_for_button(page, pattern: str, timeout_ms: int = 20000) -> bool:
    started = time.time()
    while (time.time() - started) * 1000 < timeout_ms:
        try:
            if page.get_by_role("button", name=re.compile(pattern, re.I)).count() > 0:
                return True
        except Exception:
            pass
        page.wait_for_timeout(500)
    return False


def _wait_for_text(page, pattern: str, timeout_ms: int = 20000) -> bool:
    started = time.time()
    while (time.time() - started) * 1000 < timeout_ms:
        try:
            if page.get_by_text(re.compile(pattern, re.I)).count() > 0:
                return True
        except Exception:
            pass
        page.wait_for_timeout(500)
    return False


def run_smoke(base_url: str, universe: str, live_timeout_sec: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "universe": universe,
        "steps": [],
        "console_errors": [],
        "page_errors": [],
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1720, "height": 1060})
        page = context.new_page()

        page.on("console", lambda msg: report["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: report["page_errors"].append(str(exc)))

        def save(name: str) -> None:
            page.screenshot(path=str(out_dir / name), full_page=True)

        def step(name: str, fn):
            start = time.time()
            item = {"name": name, "status": "pass", "details": "", "elapsed_sec": 0.0}
            try:
                details = fn()
                if details:
                    item["details"] = str(details)
            except Exception as exc:
                item["status"] = "fail"
                item["details"] = str(exc)
            item["elapsed_sec"] = round(time.time() - start, 2)
            report["steps"].append(item)
            return item["status"] == "pass"

        step(
            "open_app",
            lambda: (
                page.goto(base_url, wait_until="domcontentloaded", timeout=45000),
                (_ for _ in ()).throw(Exception("Streamlit shell loaded but the benchmark workspace did not render"))
                if not _wait_for_text(page, r"Live Screener|Backtest|Simulator", timeout_ms=30000)
                else None,
                page.wait_for_timeout(1000),
                save("live_sim_01_home.png"),
                f"title={page.title()}",
            )[-1],
        )

        step(
            "switch_live_screener",
            lambda: (
                (_ for _ in ()).throw(Exception("Live Screener mode click failed"))
                if not _click_mode(page, "Live Screener")
                else None,
                (_ for _ in ()).throw(Exception("Stock Research workspace click failed in Live Screener"))
                if not _click_workspace(page, "Stock Research")
                else None,
                (_ for _ in ()).throw(Exception("RUN SCAN did not appear after switching to Stock Research"))
                if not _wait_for_button(page, r"RUN SCAN", timeout_ms=20000)
                else None,
                save("live_sim_02_live.png"),
                "live_mode_ready",
            )[-1],
        )

        def live_scan_step():
            if not _select_universe(page, universe):
                raise Exception(f"Could not select {universe} in Live Screener")
            run_btn = page.get_by_role("button", name=re.compile("RUN SCAN", re.I)).first
            if run_btn.count() == 0:
                raise Exception("RUN SCAN button not found")
            run_btn.click(timeout=10000)
            state = "unknown"
            progress = ""
            start = time.time()
            while time.time() - start < live_timeout_sec:
                page.wait_for_timeout(2500)
                body = page.text_content("body") or ""
                m = re.search(r"\b\d{1,3}%\s*Complete\b", body, flags=re.I)
                if m:
                    progress = m.group(0)
                if re.search(r"Scan Complete|No setups|Top opportunities|results", body, flags=re.I):
                    state = "finished_or_results_visible"
                    break
                if re.search(rf"Resolving {re.escape(universe)}|Loaded .* symbols for {re.escape(universe)}", body, flags=re.I):
                    state = "in_progress"
            save("live_sim_03_live_after_scan.png")
            return f"state={state}, progress={progress or 'n/a'}"

        step("live_scan_smoke", live_scan_step)

        step(
            "switch_simulator",
            lambda: (
                (_ for _ in ()).throw(Exception("Simulator mode click failed"))
                if not _click_mode(page, "Simulator")
                else None,
                (_ for _ in ()).throw(Exception("Stock Research workspace click failed in Simulator"))
                if not _click_workspace(page, "Stock Research")
                else None,
                (_ for _ in ()).throw(Exception("Phase 1 scan button did not appear after switching to Stock Research"))
                if not _wait_for_button(page, r"PHASE 1: Scan for New Entries", timeout_ms=20000)
                else None,
                save("live_sim_04_simulator.png"),
                "simulator_mode_ready",
            )[-1],
        )

        def simulator_step():
            if not _select_universe(page, universe):
                raise Exception(f"Could not select {universe} in Simulator")
            phase1_btn = page.get_by_role(
                "button",
                name=re.compile(r"PHASE 1: Scan for New Entries", re.I),
            ).first
            if phase1_btn.count() == 0:
                raise Exception("Simulator Phase 1 button not found")
            phase1_btn.click(timeout=10000)
            page.wait_for_timeout(3000)
            body = page.text_content("body") or ""
            if not re.search(r"Resolving|Executing Scan|Simulation", body, flags=re.I):
                raise Exception("Simulator did not show expected phase status text")
            save("live_sim_05_simulator_after_phase1.png")
            return "phase1_started"

        step("simulator_phase1_smoke", simulator_step)

        browser.close()

    report["summary"] = {
        "passed": sum(1 for s in report["steps"] if s["status"] == "pass"),
        "failed": sum(1 for s in report["steps"] if s["status"] == "fail"),
        "console_error_count": len(report["console_errors"]),
        "page_error_count": len(report["page_errors"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Screener + Simulator smoke test")
    parser.add_argument("--base-url", default="http://localhost:8501", help="Streamlit URL")
    parser.add_argument("--universe", default="RUSSELL3000", help="Universe to select")
    parser.add_argument("--live-timeout-sec", type=int, default=150, help="Live scan wait timeout")
    parser.add_argument(
        "--out-dir",
        default="output/playwright",
        help="Directory for screenshots/report",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report = run_smoke(
        base_url=args.base_url,
        universe=args.universe,
        live_timeout_sec=max(30, int(args.live_timeout_sec)),
        out_dir=out_dir,
    )

    report_path = out_dir / "live_simulator_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(str(report_path))
    print(json.dumps(report["summary"]))
    for step in report["steps"]:
        print(f"{step['status'].upper()} {step['name']}: {step['details']}")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
