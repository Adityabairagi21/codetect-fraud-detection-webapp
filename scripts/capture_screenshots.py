"""
Drive the running app (http://127.0.0.1:5000) through the demo workflow with the
installed Chrome, and save screenshots + sample reports to screenshots/.

    venv\\Scripts\\python.exe scripts\\capture_screenshots.py
Needs: the app running, the demo accounts in .demo_credentials.txt (dev-only, git-ignored),
and `pip install playwright` (uses the local Chrome; no browser download).
"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:5000"
OUT = APP / "screenshots"
SAMPLE = APP / "data" / "sample_uploads" / "ibm_hi_small_2022-09-09_1000-1030.csv"
creds = dict(line.split()[:2] for line in (APP / ".demo_credentials.txt").read_text().splitlines()[1:] if line.strip())
ADMIN, ANALYST = "admin@demo.example.com", "analyst@demo.example.com"


def shot(page, name, full=True):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(900)                       # let Plotly finish drawing
    page.screenshot(path=OUT / f"{name}.png", full_page=full)
    print("saved", name)


def login(page, email):
    page.goto(f"{BASE}/login")
    page.fill("#email", email)
    page.fill("#password", creds[email])
    page.click("button[type=submit], input[type=submit]")
    page.wait_for_load_state("networkidle")


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(240_000)

        page.goto(f"{BASE}/login"); shot(page, "01_login", full=False)
        page.goto(f"{BASE}/register"); shot(page, "02_register", full=False)

        # ---------------- admin workflow
        login(page, ADMIN)
        page.goto(f"{BASE}/admin/datasets/upload")
        page.fill("#dataset_name", "IBM test window - 9 Sep 2022 10:00-10:30")
        page.set_input_files("#file", str(SAMPLE))
        shot(page, "03_upload_form", full=False)
        page.click("input[type=submit]")
        page.wait_for_url(re.compile(r"/datasets/\d+$"))
        ds_id = int(re.search(r"/datasets/(\d+)$", page.url).group(1))
        shot(page, "04_dataset_uploaded")

        page.click("text=Run fraud detection")
        page.wait_for_url(re.compile(r"/results"))
        shot(page, "05_detection_results")

        page.goto(f"{BASE}/admin/dashboard"); shot(page, "06_admin_dashboard")
        page.goto(f"{BASE}/graph?dataset={ds_id}")
        page.wait_for_selector("#network .main-svg"); shot(page, "07_network_suspicious")
        sender = page.locator("table tbody tr td a").first.inner_text().strip()
        page.goto(f"{BASE}/graph?dataset={ds_id}&account={sender}")
        page.wait_for_selector("#network .main-svg"); shot(page, "08_network_account")

        page.goto(f"{BASE}/alerts?dataset={ds_id}&risk=Critical"); shot(page, "09_alerts")
        page.locator("table a.btn", has_text="Review").first.click()
        page.wait_for_url(re.compile(r"/alerts/\d+"))
        page.select_option("#status", "Reviewed")
        page.fill("#review_remarks", "Checked the sender's counterparties and recent activity; escalated for KYC review.")
        page.click("input[type=submit]")
        page.wait_for_load_state("networkidle"); shot(page, "10_alert_reviewed")

        page.locator("dd.mono a").first.click()          # transaction reference on the alert page
        page.wait_for_url(re.compile(r"/transactions/\d+")); shot(page, "11_transaction_detail")

        page.goto(f"{BASE}/reports?dataset={ds_id}")
        for fmt in ("PDF", "CSV"):
            page.select_option("#report_format", fmt)
            with page.expect_download() as d:
                page.click("input[type=submit]")
            d.value.save_as(OUT / f"sample_report.{fmt.lower()}")
            print("saved report", fmt)
            page.goto(f"{BASE}/reports?dataset={ds_id}")
        shot(page, "12_reports")
        page.goto(f"{BASE}/model?dataset={ds_id}"); shot(page, "13_model_comparison")
        page.goto(f"{BASE}/admin/users"); shot(page, "14_admin_users")
        page.click("text=Sign out", force=True) if page.locator("text=Sign out").is_visible() else None
        ctx.clear_cookies()

        # ---------------- analyst workflow
        login(page, ANALYST)
        page.goto(f"{BASE}/user/dashboard"); shot(page, "15_user_dashboard")
        page.goto(f"{BASE}/datasets/{ds_id}/transactions?type=ACH&from=2022-09-09&to=2022-09-09")
        shot(page, "16_transaction_search")
        page.locator("text=Details").nth(3).click()
        page.wait_for_url(re.compile(r"/transactions/\d+"))
        if page.locator("text=Analyse transaction").count():
            page.click("text=Analyse transaction")
            page.wait_for_load_state("networkidle")
        shot(page, "17_analyst_single_analysis")
        page.goto(f"{BASE}/user/history"); shot(page, "18_analysis_history")
        page.goto(f"{BASE}/admin/datasets/upload"); shot(page, "19_access_denied_for_analyst", full=False)

        # ---------------- mobile
        m = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
        mp = m.new_page()
        mp.set_default_timeout(120_000)
        login(mp, ANALYST)
        mp.goto(f"{BASE}/user/dashboard"); shot(mp, "20_mobile_dashboard", full=False)
        browser.close()
    print("done:", len(list(OUT.glob("*.png"))), "screenshots")


if __name__ == "__main__":
    sys.exit(main())
