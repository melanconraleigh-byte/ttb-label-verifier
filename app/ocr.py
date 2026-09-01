"""
Local OCR pipeline built on Tesseract.

Design goals (from stakeholder interviews):
  * No outbound network calls  -> Tesseract runs in-process (Marcus: firewall blocks ML endpoints)
  * Results in ~5 seconds       -> single pass by default, extra passes only when the first looks bad (Sarah)
  * Tolerate imperfect photos   -> light preprocessing + orientation detection (Jenny)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Tesseract page segmentation mode 3 = fully automatic layout analysis. Labels mix a
# very large brand name with small body text; PSM 6 ("single uniform block") silently
# drops the oversized brand line, PSM 3 keeps it. Measured on the sample set.
_TESS_CONFIG = "--oem 3 --psm 3"
_MIN_ACCEPTABLE_CONFIDENCE = 55.0   # below this we try harder (rotations / alternate binarisation)
_TARGET_MIN_WIDTH = 1200            # Tesseract likes ~30px x-height; upscale small images
_MAX_WIDTH = 2400                   # phone photos are 3000-4000px; downscale so OCR stays fast


@dataclass
class OcrResult:
    text: str
    confidence: float                     # mean word confidence, 0-100
    rotation_applied: int                 # degrees the image was rotated before OCR
    elapsed_ms: int
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def load_image(data: bytes) -> np.ndarray:
    """Decode bytes (PNG/JPEG/WEBP/etc.) into a BGR numpy image, honouring EXIF orientation."""
    import io

    pil = Image.open(io.BytesIO(data))
    pil = ImageOps.exif_transpose(pil)     # phone photos often carry rotation in EXIF only
    pil = pil.convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def preprocess(img: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """
    Produce a clean, high-contrast grayscale image for Tesseract.

    Standard pass: grayscale -> upscale -> mild denoise -> Otsu threshold.
    Aggressive pass (used only when the first pass is low-confidence): CLAHE for
    uneven lighting / glare, then adaptive threshold which copes better with
    gradients across the label.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < _TARGET_MIN_WIDTH:
        scale = _TARGET_MIN_WIDTH / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif w > _MAX_WIDTH:
        scale = _MAX_WIDTH / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if aggressive:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract expects dark text on light background. If the label is inverted
    # (light text on dark bottle), flip it.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def _rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return img
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]
    return cv2.rotate(img, code)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def _run_tesseract(binary: np.ndarray) -> tuple[str, float]:
    """Return (text, mean_confidence). Confidence ignores -1 entries (non-word boxes)."""
    data = pytesseract.image_to_data(binary, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    lines: dict[tuple, list[str]] = {}
    for i, txt in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not txt.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(txt)
        words.append(txt)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for ws in lines.values())
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return text, mean_conf


def _detect_orientation(binary: np.ndarray) -> int | None:
    """Ask Tesseract's OSD engine which way the text is facing. Returns degrees or None.

    OSD is run on a half-size copy: orientation detection doesn't need full resolution
    and this roughly halves its cost (~0.35s vs ~0.7s on the samples)."""
    try:
        small = cv2.resize(binary, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        osd = pytesseract.image_to_osd(small, config="--psm 0")
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip()) % 360
    except pytesseract.TesseractError:
        return None
    return None


def extract_text(data: bytes) -> OcrResult:
    """
    OCR an image with an escalating strategy that stays fast in the common case:

      1. Standard preprocessing, no rotation.       (~0.5-1.5s)
      2. If confidence is poor, ask OSD for orientation and retry rotated.
      3. If still poor, retry with aggressive preprocessing.
      4. Keep whichever attempt scored best.
    """
    t0 = time.perf_counter()
    img = load_image(data)
    warnings: list[str] = []

    best_text, best_conf, best_rot, attempts = "", -1.0, 0, 0

    def attempt(rotation: int, aggressive: bool) -> None:
        nonlocal best_text, best_conf, best_rot, attempts
        attempts += 1
        binary = preprocess(_rotate(img, rotation), aggressive=aggressive)
        text, conf = _run_tesseract(binary)
        if conf > best_conf:
            best_text, best_conf, best_rot = text, conf, rotation

    attempt(0, aggressive=False)

    if best_conf < _MIN_ACCEPTABLE_CONFIDENCE:
        rot = _detect_orientation(preprocess(img))
        if rot:
            attempt(rot, aggressive=False)
            if best_rot == rot:
                warnings.append(f"Image appeared rotated; corrected by {rot}°.")

    if best_conf < _MIN_ACCEPTABLE_CONFIDENCE:
        attempt(best_rot, aggressive=True)

    if best_conf < _MIN_ACCEPTABLE_CONFIDENCE:
        warnings.append(
            "Low OCR confidence. The image may be blurry, low-resolution, or have glare. "
            "Treat results as a hint and review manually, or request a clearer image."
        )

    return OcrResult(
        text=best_text,
        confidence=round(best_conf, 1),
        rotation_applied=best_rot,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        attempts=attempts,
        warnings=warnings,
    )
