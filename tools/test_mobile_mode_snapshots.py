#!/usr/bin/env python3
"""Capture mobile viewport snapshots for the three primary modes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


MODE_RADIO_INDEX = {
    "Live Screener": 0,
    "Backtest": 1,
    "Simulator": 2,
}

WORKSPACE_RADIO_INDEX = {
    "ETF Benchmark": 3,
    "Hybrid Benchmark": 4,
    "Stock Benchmark": 5,
    "Stock Research": 6,
}


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
                    try:
                        loc.scroll_into_view_if_needed(timeout=3000)
                    except Exception:
                        pass
                    loc.click(timeout=5000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(500)
    idx = MODE_RADIO_INDEX.get(label)
    if idx is not None:
        try:
            page.get_by_role("radio").nth(idx).check(timeout=5000)
            return True
        except Exception:
            pass
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
                    try:
                        loc.scroll_into_view_if_needed(timeout=3000)
                    except Exception:
                        pass
                    loc.click(timeout=5000)
                    return True
            except Exception:
                pass
        page.wait_for_timeout(500)
    idx = WORKSPACE_RADIO_INDEX.get(label)
    if idx is not None:
        try:
            page.get_by_role("radio").nth(idx).check(timeout=5000)
            return True
        except Exception:
            pass
    return False


def _wait_for_text(page, pattern: str, timeout_ms: int = 12000) -> None:
    locator = page.get_by_text(pattern, exact=False)
    locator.first.wait_for(timeout=timeout_ms)


def _ensure_workspace(page, label: str) -> None:
    _open_sidebar_if_collapsed(page)
    _click_workspace(page, label)
    page.wait_for_timeout(1000)
    _close_sidebar_if_open(page)


def _open_sidebar_if_collapsed(page) -> None:
    try:
        expand_btn = page.locator('button[data-testid="stExpandSidebarButton"]').first
        if expand_btn.count() > 0:
            expand_btn.click(timeout=3000)
            page.wait_for_timeout(700)
    except Exception:
        pass


def _close_sidebar_if_open(page) -> None:
    try:
        close_btn = page.locator('button[data-testid="stBaseButton-headerNoPadding"]').first
        if close_btn.count() > 0:
            close_btn.click(timeout=3000)
            page.wait_for_timeout(700)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Mobile mode snapshots")
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--out-dir", default="output/playwright")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 430, "height": 932},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
        )
        page = context.new_page()
        page.goto(args.base_url, wait_until="domcontentloaded", timeout=45000)
        try:
            _wait_for_text(page, "Select Mode", timeout_ms=30000)
        except Exception:
            page.wait_for_timeout(3000)
        _ensure_workspace(page, "ETF Benchmark")
        try:
            _wait_for_text(page, "Current Target Allocation", timeout_ms=12000)
        except Exception:
            page.wait_for_timeout(2000)
        page.screenshot(path=str(out_dir / "mobile_01_live.png"), full_page=True)

        _open_sidebar_if_collapsed(page)
        if _click_mode(page, "Backtest"):
            page.wait_for_timeout(1000)
            _click_workspace(page, "ETF Benchmark")
            _close_sidebar_if_open(page)
            try:
                _wait_for_text(page, "Run ETF Benchmark", timeout_ms=12000)
            except Exception:
                _ensure_workspace(page, "ETF Benchmark")
                try:
                    _wait_for_text(page, "Run ETF Benchmark", timeout_ms=12000)
                except Exception:
                    page.wait_for_timeout(3000)
            page.screenshot(path=str(out_dir / "mobile_02_backtest.png"), full_page=True)

        _open_sidebar_if_collapsed(page)
        if _click_mode(page, "Simulator"):
            page.wait_for_timeout(1000)
            _click_workspace(page, "ETF Benchmark")
            _close_sidebar_if_open(page)
            try:
                _wait_for_text(page, "Adopt Latest ETF Allocation", timeout_ms=15000)
            except Exception:
                _ensure_workspace(page, "ETF Benchmark")
                try:
                    _wait_for_text(page, "Adopt Latest ETF Allocation", timeout_ms=15000)
                except Exception:
                    page.wait_for_timeout(4000)
            page.screenshot(path=str(out_dir / "mobile_03_simulator.png"), full_page=True)

        browser.close()

    print(str(out_dir / "mobile_01_live.png"))
    print(str(out_dir / "mobile_02_backtest.png"))
    print(str(out_dir / "mobile_03_simulator.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
