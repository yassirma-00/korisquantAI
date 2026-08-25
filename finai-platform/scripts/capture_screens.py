import asyncio
import pathlib
from playwright.async_api import async_playwright
BASE="http://127.0.0.1:8000"
# Chemin relatif au dépôt : un chemin absolu codait en dur le nom du
# dossier de développement et cassait le script sur une autre machine.
OUT=str(pathlib.Path(__file__).resolve().parents[1] / "docs" / "screens")
# page, wait_ms, label
PAGES=[("index",8000),("analysis",7000),("forecast",7000),("rl",7000),
       ("signals",9000),("stress",5000),("xai",7000),("portfolio",8000),
       ("risk",9000),("hyperparams",7000),("training",7000)]
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=["--no-sandbox"])
        pg=await b.new_page(viewport={"width":1460,"height":1000}, device_scale_factor=2)
        errs=[]
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
        # public pages first (no login)
        await pg.goto(f"{BASE}/landing.html", wait_until="networkidle"); await pg.wait_for_timeout(3000)
        await pg.screenshot(path=f"{OUT}/00_landing.png")
        await pg.goto(f"{BASE}/auth.html", wait_until="networkidle"); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{OUT}/01_auth.png")
        # login
        await pg.fill("#loginId","pwrisk"); await pg.fill("#loginPw","PwRiskPass123!")
        await pg.click("#loginSubmit"); await pg.wait_for_timeout(4000)
        await pg.evaluate("localStorage.setItem('korisquant:theme','light')")
        for i,(name,wait) in enumerate(PAGES, start=2):
            await pg.goto(f"{BASE}/{name}.html", wait_until="networkidle")
            await pg.wait_for_timeout(wait)
            await pg.screenshot(path=f"{OUT}/{i:02d}_{name}.png")
            print(f"  captured {name}")
        print("JS_ERRORS:", errs[:4] if errs else "none")
        await b.close()
asyncio.run(main())
