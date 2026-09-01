"""
Generate reproducible test labels (no external image-generation tools needed).

Produces samples/*.png plus samples/manifest.csv, covering the cases the
stakeholders described:
  01 clean bourbon label that should PASS
  02 same label, application says 40% ABV        -> ABV FAIL
  03 title-case "Government Warning"             -> warning FAIL (Jenny's case)
  04 label reads STONE'S THROW, app says Stone's Throw -> brand PASS w/ case note (Dave's case)
  05 altered warning wording                     -> warning FAIL with diff
  06 rotated 90°                                 -> orientation correction
  07 photographed: perspective skew + glare + noise -> preprocessing stress test
  08 wine label (12.5% ABV, no proof), missing warning -> FAIL
  09 label missing net contents                  -> net contents FAIL
"""
from __future__ import annotations

import csv
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
SERIF_B = FONT_DIR + "DejaVuSerif-Bold.ttf"
SERIF = FONT_DIR + "DejaVuSerif.ttf"
SANS = FONT_DIR + "DejaVuSans.ttf"
SANS_B = FONT_DIR + "DejaVuSans-Bold.ttf"

WARNING_HEADER = "GOVERNMENT WARNING:"
WARNING_BODY = (
    "(1) According to the Surgeon General, women should not drink alcoholic beverages "
    "during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic "
    "beverages impairs your ability to drive a car or operate machinery, and may cause health problems."
)


def render_label(brand: str, class_type: str, abv_line: str, net: str, producer: str,
                 origin: str = "", warning_header: str = WARNING_HEADER, warning_body: str = WARNING_BODY,
                 bg=(246, 240, 226), fg=(30, 25, 20), width=1000, height=1400) -> Image.Image:
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, width - 30, height - 30], outline=fg, width=6)
    d.rectangle([44, 44, width - 44, height - 44], outline=fg, width=2)

    y = 110

    def center(text, font, gap):
        nonlocal y
        w = d.textlength(text, font=font)
        d.text(((width - w) / 2, y), text, font=font, fill=fg)
        y += gap

    brand_font = ImageFont.truetype(SERIF_B, 78 if len(brand) < 16 else 60)
    center(brand, brand_font, 120)
    center(class_type, ImageFont.truetype(SERIF, 40), 90)
    d.line([(160, y), (width - 160, y)], fill=fg, width=3)
    y += 50
    center(abv_line, ImageFont.truetype(SANS_B, 44), 80)
    center(net, ImageFont.truetype(SANS, 44), 110)
    if producer:
        center(producer, ImageFont.truetype(SANS, 30), 48)
    if origin:
        center(origin, ImageFont.truetype(SANS, 30), 48)

    if warning_header or warning_body:
        y = height - 380
        d.line([(160, y), (width - 160, y)], fill=fg, width=2)
        y += 30
        wf_b = ImageFont.truetype(SANS_B, 26)
        wf = ImageFont.truetype(SANS, 26)
        lines = textwrap.wrap(warning_header + " " + warning_body, width=58)
        # bold header on first line, regular body after
        x0 = 90
        first = lines[0]
        if first.startswith(warning_header):
            d.text((x0, y), warning_header, font=wf_b, fill=fg)
            hx = d.textlength(warning_header, font=wf_b)
            d.text((x0 + hx, y), first[len(warning_header):], font=wf, fill=fg)
        else:
            d.text((x0, y), first, font=wf, fill=fg)
        y += 36
        for ln in lines[1:]:
            d.text((x0, y), ln, font=wf, fill=fg)
            y += 36
    return img


def photograph(img: Image.Image, seed: int = 7) -> Image.Image:
    """Simulate a phone photo: perspective skew, uneven light, glare spot, blur, sensor noise."""
    rng = random.Random(seed)
    arr = np.array(img)
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: rng.uniform(-0.045, 0.045)  # noqa: E731
    dst = np.float32([[w * abs(j()), h * abs(j())], [w * (1 - abs(j())), h * abs(j())],
                      [w * (1 - abs(j())), h * (1 - abs(j()))], [w * abs(j()), h * (1 - abs(j()))]])
    M = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, M, (w, h), borderValue=(90, 80, 70))

    # lighting gradient + glare
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (0.75 + 0.35 * xx / w).astype(np.float32)
    glare = np.exp(-(((xx - w * 0.72) ** 2) / (2 * (w * 0.12) ** 2) + ((yy - h * 0.35) ** 2) / (2 * (h * 0.10) ** 2)))
    light = np.clip(grad + 0.9 * glare, 0, 1.6)[..., None]
    arr = np.clip(arr.astype(np.float32) * light, 0, 255).astype(np.uint8)

    out = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.8))
    noise = np.random.default_rng(seed).normal(0, 6, (h, w, 3))
    out = Image.fromarray(np.clip(np.array(out).astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out.resize((int(w * 0.75), int(h * 0.75)), Image.LANCZOS)


def main() -> None:
    rows = []

    def save(name, img, **app):
        img.save(OUT / name)
        rows.append({"filename": name, **app})

    bourbon = dict(brand="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                   abv_line="45% Alc./Vol. (90 Proof)", net="750 mL",
                   producer="Distilled and Bottled by Old Tom Distillery, Bardstown, KY")
    bourbon_app = dict(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey",
                       abv="45% Alc./Vol. (90 Proof)", net_contents="750 mL",
                       producer="Old Tom Distillery, Bardstown, KY", country_of_origin="")

    save("01_bourbon_clean.png", render_label(**bourbon), **bourbon_app)
    save("02_bourbon_abv_mismatch.png", render_label(**bourbon), **{**bourbon_app, "abv": "40% Alc./Vol. (80 Proof)"})
    save("03_bourbon_titlecase_warning.png", render_label(**bourbon, warning_header="Government Warning:"), **bourbon_app)

    stones = dict(brand="STONE'S THROW", class_type="American Single Malt Whiskey",
                  abv_line="43% Alc./Vol. (86 Proof)", net="750 mL", producer="Stone's Throw Distilling Co., Bend, OR")
    save("04_stones_throw_case.png", render_label(**stones),
         brand_name="Stone's Throw", class_type="American Single Malt Whiskey", abv="43%", net_contents="750 mL",
         producer="Stone's Throw Distilling Co.", country_of_origin="")

    altered = WARNING_BODY.replace("should not drink", "should avoid drinking").replace("may cause health problems", "can cause health issues")
    save("05_bourbon_altered_warning.png", render_label(**bourbon, warning_body=altered), **bourbon_app)

    rotated = render_label(**bourbon).rotate(90, expand=True)
    save("06_bourbon_rotated.png", rotated, **bourbon_app)

    save("07_bourbon_photo.jpg", photograph(render_label(**bourbon)), **bourbon_app)
    (OUT / "07_bourbon_photo.jpg").unlink()  # save as jpeg properly
    photograph(render_label(**bourbon)).save(OUT / "07_bourbon_photo.jpg", quality=82)

    wine = dict(brand="Château Belle Rivière", class_type="Red Wine", abv_line="12.5% Alc. by Vol.", net="750 mL",
                producer="Produced and Bottled by Belle Rivière, Bordeaux", origin="Product of France",
                warning_header="", warning_body="")
    save("08_wine_missing_warning.png", render_label(**wine, bg=(120, 30, 40), fg=(245, 235, 220)),
         brand_name="Chateau Belle Riviere", class_type="Red Wine", abv="12.5%", net_contents="750 mL",
         producer="Belle Rivière, Bordeaux", country_of_origin="Product of France")

    save("09_gin_missing_net_contents.png",
         render_label(brand="HARBOR LIGHT GIN", class_type="Distilled Gin", abv_line="47% Alc./Vol. (94 Proof)",
                      net="", producer="Harbor Light Spirits, Portland, ME"),
         brand_name="HARBOR LIGHT GIN", class_type="Distilled Gin", abv="47%", net_contents="750 mL",
         producer="Harbor Light Spirits", country_of_origin="")

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "brand_name", "class_type", "abv", "net_contents", "producer", "country_of_origin"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} labels to {OUT}")


if __name__ == "__main__":
    main()
