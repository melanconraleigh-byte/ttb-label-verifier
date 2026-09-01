"""
FastAPI application: serves the single-page UI and two JSON endpoints.

POST /api/verify        one image + application fields          -> one result
POST /api/verify-batch  many images + optional CSV manifest     -> list of results
GET  /api/health        liveness + tesseract version
"""
from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path

import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ocr import extract_text
from .verify import Application, GOVERNMENT_WARNING, verify

log = logging.getLogger("ttb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_BATCH_FILES = 300          # Sarah: importers dump 200-300 at once
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp"}

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="TTB Label Verifier", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if (ROOT / "samples").exists():
    app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "tesseract": str(pytesseract.get_tesseract_version())}


@app.get("/api/warning-text")
def warning_text() -> dict:
    return {"text": GOVERNMENT_WARNING}


# --------------------------------------------------------------------------- #
def _validate_upload(f: UploadFile, data: bytes) -> None:
    if f.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"{f.filename}: unsupported file type {f.content_type}. Use PNG, JPEG, WEBP, TIFF or BMP.")
    if not data:
        raise HTTPException(400, f"{f.filename}: file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{f.filename}: file exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB.")


def _process(filename: str, data: bytes, application: Application) -> dict:
    t0 = time.perf_counter()
    try:
        ocr = extract_text(data)
    except Exception as exc:  # corrupted image, unreadable format, etc.
        log.exception("OCR failed for %s", filename)
        return {
            "filename": filename,
            "error": f"Could not read image: {exc.__class__.__name__}",
            "overall": "error",
            "fields": [],
            "timing_ms": int((time.perf_counter() - t0) * 1000),
        }
    result = verify(application, ocr.text)
    result.update(
        {
            "filename": filename,
            "application": application.__dict__,
            "ocr": {
                "text": ocr.text,
                "confidence": ocr.confidence,
                "rotation_applied": ocr.rotation_applied,
                "attempts": ocr.attempts,
                "warnings": ocr.warnings,
            },
            "timing_ms": int((time.perf_counter() - t0) * 1000),
        }
    )
    log.info("%s -> %s in %dms (ocr conf %.0f)", filename, result["overall"], result["timing_ms"], ocr.confidence)
    return result


@app.post("/api/verify")
async def verify_single(
    image: UploadFile = File(...),
    brand_name: str = Form(""),
    class_type: str = Form(""),
    abv: str = Form(""),
    net_contents: str = Form(""),
    producer: str = Form(""),
    country_of_origin: str = Form(""),
) -> JSONResponse:
    data = await image.read()
    _validate_upload(image, data)
    application = Application(brand_name, class_type, abv, net_contents, producer, country_of_origin)
    return JSONResponse(_process(image.filename or "upload", data, application))


def _parse_manifest(raw: bytes) -> dict[str, Application]:
    """
    CSV with a `filename` column plus any of: brand_name, class_type, abv, net_contents,
    producer, country_of_origin. Header names are case/space-insensitive.
    """
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "Manifest CSV has no header row.")
    norm = {h: h.strip().lower().replace(" ", "_").replace("/", "_") for h in reader.fieldnames}
    aliases = {"brand": "brand_name", "type": "class_type", "class": "class_type", "class_type": "class_type",
               "alcohol_content": "abv", "volume": "net_contents", "bottler": "producer", "country": "country_of_origin",
               "origin": "country_of_origin", "file": "filename", "image": "filename"}
    out: dict[str, Application] = {}
    for row in reader:
        fields = {aliases.get(norm[h], norm[h]): (v or "").strip() for h, v in row.items() if h}
        fname = fields.get("filename")
        if not fname:
            continue
        out[fname.strip().lower()] = Application(
            brand_name=fields.get("brand_name", ""),
            class_type=fields.get("class_type", ""),
            abv=fields.get("abv", ""),
            net_contents=fields.get("net_contents", ""),
            producer=fields.get("producer", ""),
            country_of_origin=fields.get("country_of_origin", ""),
        )
    if not out:
        raise HTTPException(400, "Manifest CSV must include a 'filename' column with at least one row.")
    return out


@app.post("/api/verify-batch")
async def verify_batch(
    images: list[UploadFile] = File(...),
    manifest: UploadFile | None = File(None),
    brand_name: str = Form(""),
    class_type: str = Form(""),
    abv: str = Form(""),
    net_contents: str = Form(""),
    producer: str = Form(""),
    country_of_origin: str = Form(""),
) -> JSONResponse:
    """
    Batch mode. Application data comes from the manifest CSV when supplied (keyed by
    filename); otherwise the form fields are applied to every image (useful when one
    importer submits many photos of the same label).
    """
    if len(images) > MAX_BATCH_FILES:
        raise HTTPException(413, f"Batch limited to {MAX_BATCH_FILES} images.")
    per_file = _parse_manifest(await manifest.read()) if manifest and manifest.filename else {}
    default_app = Application(brand_name, class_type, abv, net_contents, producer, country_of_origin)

    t0 = time.perf_counter()
    results = []
    for img in images:
        data = await img.read()
        try:
            _validate_upload(img, data)
        except HTTPException as exc:
            results.append({"filename": img.filename, "overall": "error", "error": exc.detail, "fields": [], "timing_ms": 0})
            continue
        key = (img.filename or "").lower()
        application = per_file.get(key)
        if per_file and application is None:
            results.append({"filename": img.filename, "overall": "error", "fields": [], "timing_ms": 0,
                            "error": "No row in manifest for this filename."})
            continue
        results.append(_process(img.filename or "upload", data, application or default_app))

    summary = {s: sum(1 for r in results if r["overall"] == s) for s in ("pass", "warn", "fail", "error")}
    return JSONResponse({"results": results, "summary": summary, "total_ms": int((time.perf_counter() - t0) * 1000)})
