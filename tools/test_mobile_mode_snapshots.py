#!/usr/bin/env python3
"""Capture mobile viewport snapshots for the three primary modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def _click_mode(page, label: str) -> bool:
    candidates = [
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
    return False


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
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out_dir / "mobile_01_live.png"), full_page=True)

        _open_sidebar_if_collapsed(page)
        if _click_mode(page, "Backtest"):
            _close_sidebar_if_open(page)
            page.wait_for_timeout(1200)
            page.screenshot(path=str(out_dir / "mobile_02_backtest.png"), full_page=True)

        _open_sidebar_if_collapsed(page)
        if _click_mode(page, "Simulator"):
            _close_sidebar_if_open(page)
            page.wait_for_timeout(1200)
            page.screenshot(path=str(out_dir / "mobile_03_simulator.png"), full_page=True)

        browser.close()

    print(str(out_dir / "mobile_01_live.png"))
    print(str(out_dir / "mobile_02_backtest.png"))
    print(str(out_dir / "mobile_03_simulator.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
