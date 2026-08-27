"""UI end-to-end test: drive every control a user can actually touch.

e2e_test.py proves the API is correct. This proves the page wired to it is,
which is a different failure mode: a working endpoint behind a button that
does not call it looks identical to a broken endpoint from the user's side.

Needs a running instance with at least one imported photo, and playwright:

    pip install playwright && playwright install chromium
    python ui_test.py --base http://127.0.0.1:5055

Reports console errors and failed requests too — a control can appear to work
while quietly 400ing underneath, which is exactly how the "Add photos" sheet
shipped broken for anyone without Immich configured.
"""
import argparse
import sys
import time

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  — ' + detail if detail and not cond else ''}",
          flush=True)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5055")
    ap.add_argument("--user", default="ui_test")
    ap.add_argument("--password", default="ui-test-password")
    ap.add_argument("--photo", default=None, help="image to upload; generated if omitted")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    import io, json, urllib.request

    photo = a.photo
    if not photo:
        import numpy as np
        from PIL import Image
        h, w = 900, 1350
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), np.float32)
        img[..., 2] = 0.55 - 0.25 * (yy / h); img[..., 1] = 0.35 + 0.25 * (yy / h); img[..., 0] = 0.30
        img[yy > h * 0.62] = (0.22, 0.34, 0.15)
        blob = ((yy - h*0.62)**2/(h*0.26)**2 + (xx - w*0.5)**2/(w*0.13)**2) < 1
        img[blob] = (0.80, 0.72, 0.60)
        img = np.clip(img + np.random.default_rng(11).normal(0, 0.02, img.shape), 0, 1)
        photo = "/tmp/ui_test_photo.jpg"
        Image.fromarray((img*255).astype("uint8"), "RGB").save(photo, quality=94)

    console, failed = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        pg.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        print("\n-- sign in --")
        pg.goto(a.base, wait_until="networkidle")
        check("/ redirects to the login screen", pg.url.rstrip("/").endswith("/login"), pg.url)
        pg.fill("#u", a.user); pg.fill("#p", a.password); pg.click("#go")
        try:
            pg.wait_for_url(lambda u: not u.endswith("/login"), timeout=20000)
            pg.wait_for_timeout(800)
            check("signing in reaches the library",
                  pg.query_selector("#lib-grid") is not None, pg.url)
        except Exception:
            return check("signing in reaches the library", False, pg.text_content("#msg") or "")

        print("\n-- add photos sheet --")
        pg.wait_for_timeout(1200)
        before = len(pg.query_selector_all(".tile:not(.pending)"))
        pg.click("#open-picker"); pg.wait_for_timeout(2500)
        sheet = pg.text_content(".sheet") or ""
        check("the sheet opens", bool(sheet.strip()))
        check("it does not open on a dead Immich pane",
              "Drop CR3" in sheet or "No Immich library" in sheet, sheet[:90])
        fi = pg.query_selector("#file-input")
        if check("the upload control is present", fi is not None):
            fi.set_input_files(photo)
            pg.wait_for_timeout(2500)
            pg.keyboard.press("Escape"); pg.wait_for_timeout(600)
            # Assert the SERVER took it, not merely that the click happened:
            # /api/gallery must now report something pending. "check(..., True)"
            # here would pass even with the upload handler deleted.
            queued = pg.evaluate(
                "fetch('/api/gallery').then(r => r.json())"
                ".then(g => g.some(i => i.pending))")
            check("uploading actually queues it server-side", queued is True, str(queued))

        print("\n-- import shows progress then finishes --")
        seen_pending = False
        end = time.time() + 420
        while time.time() < end:
            pg.reload(wait_until="networkidle"); pg.wait_for_timeout(1200)
            if pg.query_selector_all(".tile.pending"): seen_pending = True
            if len(pg.query_selector_all(".tile:not(.pending)")) > before: break
            time.sleep(3)
        n = len(pg.query_selector_all(".tile:not(.pending)"))
        check("the import finishes and appears in the library", n > before, f"{before} -> {n}")

        print("\n-- editor --")
        pg.query_selector_all(".tile:not(.pending)")[0].click(); pg.wait_for_timeout(4000)
        check("the editor opens", pg.query_selector("#hero") is not None)
        groups = pg.query_selector_all("#look-groups .chip, #look-groups button")
        # One group is correct for a photo with no usable masks — the generated
        # test image is a gradient and a blob, and segmentation rightly finds
        # nothing in it, so only the global looks are offered. Pass --photo with
        # a real photograph to exercise the region recipes as well.
        check("at least one look group is offered", len(groups) >= 1, f"{len(groups)} groups")
        if a.photo:
            check("a real photo yields per-photo recipes too", len(groups) >= 2,
                  f"{len(groups)} groups — expected a 'For this photo' group")
        looks = pg.query_selector_all("button.look")
        check("looks are listed", len(looks) > 0, f"{len(looks)} looks")
        if looks:
            src_before = pg.get_attribute("#hero", "src")
            looks[min(1, len(looks)-1)].click(); pg.wait_for_timeout(3000)
            check("choosing a look changes the hero image",
                  pg.get_attribute("#hero", "src") != src_before)

        for ctl, label in (("#strength", "strength slider"), ("#denoise", "denoise slider")):
            el = pg.query_selector(ctl)
            if el is None:
                check(f"{label} is present", False); continue
            src_before = pg.get_attribute("#hero", "src")
            # Move to whichever end is NOT where the slider already sits.
            # Denoise defaults to 0 on a clean photo (the amount ramps with
            # measured noise), so driving it to el.min set 0 to 0 and nothing
            # re-rendered — the test failed while the control was fine.
            moved = pg.eval_on_selector(ctl, """el => {
                const lo = parseFloat(el.min), hi = parseFloat(el.max);
                const cur = parseFloat(el.value);
                const target = (cur - lo) <= (hi - cur) ? hi : lo;
                if (target === cur) return false;
                el.value = target;
                el.dispatchEvent(new Event('input', {bubbles:true}));
                el.dispatchEvent(new Event('change', {bubbles:true}));
                return true; }""")
            pg.wait_for_timeout(3500)
            if not moved:
                check(f"the {label} could be moved", False, "slider min == max")
            else:
                check(f"the {label} re-renders the photo",
                      pg.get_attribute("#hero", "src") != src_before)

        crops = pg.query_selector_all("#crops .chip, #crops button")
        check("crop ratios are offered (Instagram included)", len(crops) >= 7, f"{len(crops)} ratios")
        if crops:
            crops[min(2, len(crops)-1)].click(); pg.wait_for_timeout(1500)
            check("choosing a crop shows the overlay", pg.query_selector("#crop-box") is not None)
        sizes = pg.query_selector_all("#sizes .chip, #sizes button")
        check("all four output sizes are offered", len(sizes) == 4, f"{len(sizes)}")
        check("hold-to-compare is present", pg.query_selector("#compare") is not None)

        print("\n-- download --")
        dl = pg.query_selector("#download")
        if check("the download button is present", dl is not None):
            with pg.expect_download(timeout=900000) as d:
                dl.click()
            path = d.value.path()
            size = __import__("os").path.getsize(path) if path else 0
            check("it downloads a real file", size > 20000, f"{size} bytes")

        print("\n-- settings --")
        pg.goto(f"{a.base}/settings", wait_until="networkidle"); pg.wait_for_timeout(1200)
        for sel, label in (("#immich-form", "Immich connection form"),
                           ("#pw-form", "change-password form"),
                           ("#mode", "processing mode selector")):
            check(f"{label} is present", pg.query_selector(sel) is not None)
        toggles = pg.query_selector_all("input[type=checkbox]")
        check("masking toggles are present", len(toggles) >= 3, f"{len(toggles)}")

        print("\n-- phone layout --")
        pg.set_viewport_size({"width": 390, "height": 844})
        pg.goto(a.base, wait_until="networkidle"); pg.wait_for_timeout(1800)
        check("library does not scroll sideways at 390px",
              pg.evaluate("document.documentElement.scrollWidth") <= 392)
        ts = pg.query_selector_all(".tile:not(.pending)")
        if ts:
            ts[0].click(); pg.wait_for_timeout(4000)
            check("editor does not scroll sideways at 390px",
                  pg.evaluate("document.documentElement.scrollWidth") <= 392)
        b.close()

    print("\n-- browser health --")
    check("no console errors", not console, "; ".join(dict.fromkeys(console))[:200])
    real = [f for f in dict.fromkeys(failed) if "/favicon" not in f]
    check("no failed requests", not real, "; ".join(real)[:200])

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
