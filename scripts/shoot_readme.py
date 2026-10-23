"""Capture README screenshots with Playwright. usage: shoot_readme.py [URL] (default: live Pages site)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://parbproject.github.io/vol-forecast-ml/"
IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp, mob in (("overview", {"width": 1440, "height": 900}, False), ("mobile", {"width": 390, "height": 844}, True)):
        pg = b.new_page(viewport=vp, is_mobile=mob, device_scale_factor=2 if mob else 1)
        errs = []
        pg.on("console", lambda m, errs=errs: m.type == "error" and errs.append(m.text))
        pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(IMG / f"{name}.png"), full_page=not mob)
        print(name, "errors:", errs)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name="XLK", exact=True).click()
    pg.get_by_role("button", name="Gradient boosting").click()
    pg.select_option("select", "all")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(IMG / "xlk-all.png"), full_page=True)
    b.close()
