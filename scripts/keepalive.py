"""Keep the Argo Float Monitor Streamlit app awake AND pre-loaded.

A plain HTTP GET to the Streamlit Cloud URL returns 200 with only the
static HTML shell — the Python process never starts. Playwright launches a
real headless Chromium that executes JavaScript and opens the WebSocket
connection that actually starts the app.

In addition to waking the app, this script enters WMO 7902198 in the
sidebar and clicks the GDAC download button. That way the float's NetCDF
files end up in the Streamlit Cloud filesystem cache, and a human visitor
arriving shortly afterward sees a fully-rendered dashboard immediately
rather than waiting on the cold-cache download.

Triggered every 6 hours by .github/workflows/keepalive.yml.

Author: Youran Li
"""

import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


APP_URL = "https://argo-float-monitor.streamlit.app/"
WMO_TO_PRELOAD = "7902198"

# Streamlit Cloud's sleep page button text and the in-app widgets we drive.
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
WMO_INPUT_LABEL = "WMO number"
DOWNLOAD_BUTTON_TEXT = "Download / refresh from GDAC"  # substring match; emoji-prefixed


def log(msg: str) -> None:
    """Time-stamped logging so the GitHub Actions log is easy to read."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # ── 1. Load the app ────────────────────────────────────────────────
        log(f"Navigating to {APP_URL}")
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        except PWTimeout:
            log("FATAL: timeout loading the app URL")
            browser.close()
            return 1

        # Give the SPA a moment to open the WebSocket and render.
        page.wait_for_timeout(6_000)

        # ── 2. If sleeping, click the wake-up button ───────────────────────
        wake_btn = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
        if wake_btn.count() > 0:
            log("App was sleeping — clicking wake button")
            wake_btn.first.click()
            log("Waiting 75s for the app to start…")
            page.wait_for_timeout(75_000)
        else:
            log("App is already awake")

        # ── 3. Find the WMO input and enter the target float ───────────────
        log(f"Entering WMO {WMO_TO_PRELOAD} in the sidebar")
        wmo_input = page.get_by_label(WMO_INPUT_LABEL)
        try:
            wmo_input.wait_for(state="visible", timeout=45_000)
        except PWTimeout:
            log("WARN: WMO input not visible — app is awake but UI did not finish loading")
            log("Keepalive succeeded regardless")
            browser.close()
            return 0

        wmo_input.fill(WMO_TO_PRELOAD)
        # Streamlit applies the value on blur or Enter — press Tab to commit.
        wmo_input.press("Tab")
        page.wait_for_timeout(2_000)

        # ── 4. Click the GDAC download button ──────────────────────────────
        log("Clicking the GDAC download button")
        # Use a substring match (has-text) because the actual button label is
        # "⬇ Download / refresh from GDAC" with a leading emoji.
        download_btn = page.locator(
            f"button:has-text('{DOWNLOAD_BUTTON_TEXT}')"
        ).first
        try:
            download_btn.wait_for(state="visible", timeout=30_000)
        except PWTimeout:
            log("WARN: download button not found — keepalive succeeded but pre-load skipped")
            browser.close()
            return 0

        download_btn.click()
        log("Waiting up to 90s for the GDAC fetch + parse to complete")
        # The dashboard shows a progress bar then a re-rendered page once done.
        # We do not need to verify completion; even a partial fetch warms the
        # filesystem cache. Give it a generous wait window.
        page.wait_for_timeout(90_000)

        log("Done — app awake and float pre-loaded")
        browser.close()
        return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        log(f"FATAL: {exc!r}")
        sys.exit(1)
