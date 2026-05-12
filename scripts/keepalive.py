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

v2 changes (vs v1):
  - Use clear() + type() instead of fill() so React's onChange fires
  - Press Enter to commit the input (Streamlit submits on Enter)
  - Verify the input's value after typing, before clicking Download
  - Retry once if verification fails

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
DOWNLOAD_BUTTON_TEXT = "Download / refresh from GDAC"  # substring; emoji-prefixed in app


def log(msg: str) -> None:
    """Time-stamped logging so the GitHub Actions log is easy to read."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def enter_wmo_and_verify(page, wmo: str, attempt: int = 1) -> bool:
    """Type WMO into the sidebar input and verify it landed in Streamlit state.

    Returns True on success, False if the value never stuck.
    """
    log(f"Entering WMO {wmo} (attempt {attempt})")
    wmo_input = page.get_by_label(WMO_INPUT_LABEL)
    try:
        wmo_input.wait_for(state="visible", timeout=45_000)
    except PWTimeout:
        log("WARN: WMO input never became visible")
        return False

    # Focus + select-all + delete avoids leaving stale default text behind.
    wmo_input.click()
    page.wait_for_timeout(300)
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    page.wait_for_timeout(200)

    # Real keystrokes — type() fires onChange / onInput so React state updates.
    wmo_input.type(wmo, delay=50)
    page.wait_for_timeout(500)

    # Streamlit text_input commits on Enter (or on blur).
    page.keyboard.press("Enter")

    # Let Streamlit rerun the script with the new WMO value.
    page.wait_for_timeout(3_000)

    # Verify: read the input's value back from the DOM.
    actual = wmo_input.input_value()
    log(f"Input value after typing: {actual!r}")
    if actual.strip() == wmo:
        log("WMO entered successfully")
        return True

    log(f"WARN: expected {wmo!r}, got {actual!r}")
    return False


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

        # ── 3. Enter WMO with verification + one retry ─────────────────────
        ok = enter_wmo_and_verify(page, WMO_TO_PRELOAD, attempt=1)
        if not ok:
            log("Retrying WMO entry after 5s pause…")
            page.wait_for_timeout(5_000)
            ok = enter_wmo_and_verify(page, WMO_TO_PRELOAD, attempt=2)
        if not ok:
            log("WARN: could not get WMO into the input after 2 tries")
            log("App is awake but pre-load skipped — exiting cleanly")
            browser.close()
            return 0

        # ── 4. Click the GDAC download button ──────────────────────────────
        log("Clicking the GDAC download button")
        download_btn = page.locator(
            f"button:has-text('{DOWNLOAD_BUTTON_TEXT}')"
        ).first
        try:
            download_btn.wait_for(state="visible", timeout=30_000)
        except PWTimeout:
            log("WARN: download button not visible — skipping")
            browser.close()
            return 0

        download_btn.click()
        log("Waiting up to 120s for the GDAC fetch + parse to complete")
        page.wait_for_timeout(120_000)

        # ── 5. Final verification: look for the success banner ─────────────
        # The app shows "Downloaded N files" in green when download succeeds.
        success_banner = page.locator("text=/Downloaded \\d+ files/")
        if success_banner.count() > 0:
            banner_text = success_banner.first.inner_text()
            log(f"SUCCESS banner: {banner_text}")
        else:
            log("No success banner found — download may still be in progress")

        log("Done — app awake and pre-load attempted")
        browser.close()
        return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        log(f"FATAL: {exc!r}")
        sys.exit(1)
