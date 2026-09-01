"""
Verification rules: compare what the applicant *said* (COLA application fields)
against what the label *shows* (OCR text).

Every check returns a FieldResult with a three-way status:
  PASS  - matches, agent can move on
  WARN  - almost certainly fine but needs a human glance (e.g. "STONE'S THROW" vs "Stone's Throw",
          or 1-2 characters that look like OCR noise)
  FAIL  - a real discrepancy, or the item couldn't be found on the label at all

The WARN tier exists because of Dave Morrison's point: label review needs judgment,
and a tool that turns every capitalisation difference into a hard rejection would be
ignored. The tool flags; the agent decides.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from enum import Enum

from rapidfuzz import fuzz

# 27 CFR 16.21 - the statement must appear exactly like this.
GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"   # field not supplied in the application, so nothing to compare


@dataclass
class FieldResult:
    field: str
    label: str            # human-readable name for the UI
    status: Status
    expected: str
    found: str
    message: str
    similarity: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"})


def normalize(s: str) -> str:
    """Case-fold, unify quotes/dashes, strip accents, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s).translate(_QUOTES)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().upper()


def alnum_only(s: str) -> str:
    """Even looser: letters and digits only. Used for brand names where OCR mangles punctuation."""
    return re.sub(r"[^A-Z0-9]", "", normalize(s))


def _best_window(needle: str, haystack: str) -> tuple[str, float]:
    """
    Find the substring of `haystack` that best matches `needle`, using the
    OCR line structure as candidate boundaries, then falling back to a sliding
    window of the same word count. Returns (best_text, score 0-100).
    """
    n_words = len(needle.split())
    candidates: list[str] = [ln for ln in haystack.splitlines() if ln.strip()]
    words = haystack.split()
    for width in (n_words, n_words + 1, max(1, n_words - 1)):
        candidates += [" ".join(words[i : i + width]) for i in range(0, max(1, len(words) - width + 1))]

    best_text, best_score = "", 0.0
    for cand in candidates:
        score = fuzz.ratio(normalize(needle), normalize(cand))
        if score > best_score:
            best_text, best_score = cand, score
    return best_text, best_score


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_text_field(field: str, label: str, expected: str, ocr_text: str,
                     pass_threshold: float = 92.0, warn_threshold: float = 80.0) -> FieldResult:
    """Generic fuzzy text match used for brand name and class/type."""
    if not expected or not expected.strip():
        return FieldResult(field, label, Status.SKIP, "", "", "Not provided in application.")

    # Exact, case-insensitive containment is the happy path.
    if normalize(expected) in normalize(ocr_text):
        if expected.strip() in ocr_text:
            return FieldResult(field, label, Status.PASS, expected, expected.strip(), "Exact match.", 100.0)
        found, _ = _best_window(expected, ocr_text)
        return FieldResult(
            field, label, Status.PASS, expected, found,
            "Matches (capitalisation differs, which TTB permits for this field).", 100.0,
        )

    found, score = _best_window(expected, ocr_text)
    # Punctuation-insensitive check catches OCR dropping apostrophes: STONES THROW vs STONE'S THROW
    if alnum_only(expected) and alnum_only(expected) in alnum_only(ocr_text):
        return FieldResult(
            field, label, Status.WARN, expected, found,
            "Matches ignoring punctuation/spacing - confirm by eye.", max(score, 95.0),
        )
    if score >= pass_threshold:
        return FieldResult(field, label, Status.WARN, expected, found,
                           f"Near match ({score:.0f}% similar) - likely OCR noise, confirm by eye.", score)
    if score >= warn_threshold:
        return FieldResult(field, label, Status.WARN, expected, found,
                           f"Partial match ({score:.0f}% similar) - review carefully.", score)
    return FieldResult(field, label, Status.FAIL, expected, found or "(not found)",
                       "Not found on label." if score < 50 else f"Does not match label ({score:.0f}% similar).", score)


_PCT_RE = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*%")
_PROOF_RE = re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:°\s*)?PROOF", re.I)


def parse_abv(s: str) -> float | None:
    """'45% Alc./Vol. (90 Proof)' -> 45.0 ; '45' -> 45.0 ; '12.5 %' -> 12.5"""
    if not s:
        return None
    m = _PCT_RE.search(s)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.fullmatch(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*", s)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def check_abv(expected: str, ocr_text: str) -> FieldResult:
    label = "Alcohol content"
    exp = parse_abv(expected)
    if exp is None:
        return FieldResult("abv", label, Status.SKIP, expected, "", "Not provided / unparseable in application.")

    # Prefer percentages that sit near the words ALC / VOL / ABV; fall back to any percentage.
    text_u = ocr_text.upper()
    found_pcts = [float(m.group(1).replace(",", ".")) for m in _PCT_RE.finditer(text_u)]
    near = [
        float(m.group(1).replace(",", "."))
        for m in _PCT_RE.finditer(text_u)
        if re.search(r"ALC|VOL|ABV", text_u[max(0, m.start() - 12): m.end() + 20])
    ]
    candidates = near or found_pcts
    proofs = [float(m.group(1)) for m in _PROOF_RE.finditer(text_u)]

    if not candidates and not proofs:
        return FieldResult("abv", label, Status.FAIL, f"{exp:g}%", "(not found)",
                           "No alcohol content statement found on label.")

    label_str = ", ".join(f"{c:g}%" for c in candidates) + (f" ({proofs[0]:g} proof)" if proofs else "")

    if any(abs(c - exp) < 0.05 for c in candidates):
        if proofs and abs(proofs[0] - 2 * exp) > 0.5:
            return FieldResult("abv", label, Status.WARN, f"{exp:g}%", label_str,
                               f"ABV matches, but the proof on the label ({proofs[0]:g}) is not 2x ABV ({2*exp:g}).")
        return FieldResult("abv", label, Status.PASS, f"{exp:g}%", label_str, "Alcohol content matches.")

    if not candidates and proofs and abs(proofs[0] - 2 * exp) < 0.5:
        return FieldResult("abv", label, Status.WARN, f"{exp:g}%", label_str,
                           "No % Alc./Vol. read, but proof is consistent with the application. Confirm by eye.")

    # OCR often drops a decimal point (12.5% -> 125%) or reads 5 as 6. Flag close misses as WARN.
    if any(abs(c - exp) <= 1.0 for c in candidates):
        return FieldResult("abv", label, Status.WARN, f"{exp:g}%", label_str,
                           "Alcohol content is close but not identical - confirm by eye (possible OCR misread).")
    return FieldResult("abv", label, Status.FAIL, f"{exp:g}%", label_str, "Alcohol content does not match.")


_VOL_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ML|MILLILITERS?|L|LITERS?|LITRES?|CL|FL\.?\s*OZ\.?|OZ)\b", re.I)
_TO_ML = {"ML": 1.0, "MILLILITER": 1.0, "L": 1000.0, "LITER": 1000.0, "LITRE": 1000.0,
          "CL": 10.0, "FLOZ": 29.5735, "OZ": 29.5735}


def parse_volume_ml(s: str) -> float | None:
    if not s:
        return None
    m = _VOL_RE.search(s)
    if not m:
        return None
    unit = re.sub(r"[^A-Z]", "", m.group(2).upper()).rstrip("S")
    unit = "FLOZ" if unit == "FLOZ" else unit
    return float(m.group(1).replace(",", ".")) * _TO_ML.get(unit, 1.0)


def check_net_contents(expected: str, ocr_text: str) -> FieldResult:
    label = "Net contents"
    exp_ml = parse_volume_ml(expected)
    if exp_ml is None:
        return FieldResult("net_contents", label, Status.SKIP, expected, "", "Not provided / unparseable in application.")
    found = [(m.group(0), parse_volume_ml(m.group(0))) for m in _VOL_RE.finditer(ocr_text)]
    if not found:
        return FieldResult("net_contents", label, Status.FAIL, expected, "(not found)", "No net contents statement found on label.")
    found_str = ", ".join(f for f, _ in found)
    if any(v is not None and abs(v - exp_ml) < 0.5 for _, v in found):
        return FieldResult("net_contents", label, Status.PASS, expected, found_str, "Net contents match.")
    return FieldResult("net_contents", label, Status.FAIL, expected, found_str, "Net contents do not match.")


def _diff_words(expected: str, found: str) -> list[dict]:
    """Word-level diff so the agent sees exactly which words differ."""
    a, b = expected.split(), found.split()
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        out.append({"op": tag, "expected": " ".join(a[i1:i2]), "found": " ".join(b[j1:j2])})
    return out


def _looks_like_ocr_noise(diffs: list[dict]) -> bool:
    """
    True when every difference is a same-length word swap where each swapped word is
    a near-miss of the original (machinery -> machinory). Any inserted, deleted, or
    genuinely different word ("not drink" -> "avoid drinking") is a wording change and
    must FAIL, however high the overall character similarity is.
    """
    for d in diffs:
        if d["op"] != "replace":
            return False
        a, b = d["expected"].split(), d["found"].split()
        if len(a) != len(b):
            return False
        for wa, wb in zip(a, b):
            if fuzz.ratio(alnum_only(wa), alnum_only(wb)) < 75:
                return False
    return True


def check_government_warning(ocr_text: str) -> tuple[FieldResult, list[dict]]:
    """
    The warning must be word-for-word (27 CFR 16.21) and 'GOVERNMENT WARNING:' must be
    capitalised and bold (27 CFR 16.22). OCR can verify wording and capitalisation; it
    cannot reliably verify bold, so that is left as an explicit manual check.
    """
    label = "Government warning"
    text = re.sub(r"\s+", " ", ocr_text.translate(_QUOTES))

    # Locate the statement start, tolerating case so we can report a capitalisation failure.
    m = re.search(r"GOVERNMENT\s+WARNING\s*:?", text, re.I)
    if not m:
        return FieldResult("government_warning", label, Status.FAIL, GOVERNMENT_WARNING, "(not found)",
                           "No government warning statement found on label."), []

    header_on_label = m.group(0)
    header_is_caps = header_on_label.upper() == header_on_label

    # Take from the header through the end of the statement (or ~same length if the end is garbled).
    tail = text[m.start():]
    end = re.search(r"health\s+problems\s*\.?", tail, re.I)
    found = tail[: end.end()] if end else tail[: len(GOVERNMENT_WARNING) + 20]
    found = found.strip()

    expected_body = normalize(GOVERNMENT_WARNING)
    found_body = normalize(found)
    score = fuzz.ratio(expected_body, found_body)
    diffs = _diff_words(GOVERNMENT_WARNING, found)

    if not header_is_caps:
        return FieldResult("government_warning", label, Status.FAIL, GOVERNMENT_WARNING, found,
                           f'"{header_on_label.strip()}" must be in capital letters: "GOVERNMENT WARNING:".', score), diffs

    if found_body == expected_body:
        return FieldResult("government_warning", label, Status.PASS, GOVERNMENT_WARNING, found,
                           "Wording matches exactly. Bold type on 'GOVERNMENT WARNING:' must still be confirmed by eye.", 100.0), []
    # OCR routinely drops a comma at a line break. If every word matches and only
    # punctuation differs, treat as a pass but say so - the agent still eyeballs it.
    if alnum_only(found) == alnum_only(GOVERNMENT_WARNING):
        return FieldResult("government_warning", label, Status.PASS, GOVERNMENT_WARNING, found,
                           "All words match. Punctuation and bold type on 'GOVERNMENT WARNING:' should be confirmed by eye.", score), []
    if _looks_like_ocr_noise(diffs):
        return FieldResult("government_warning", label, Status.WARN, GOVERNMENT_WARNING, found,
                           f"Wording differs by a few characters ({score:.1f}% similar). Likely OCR noise - confirm the highlighted words by eye.", score), diffs
    return FieldResult("government_warning", label, Status.FAIL, GOVERNMENT_WARNING, found,
                       f"Wording does not match the required statement ({score:.0f}% similar). See highlighted differences.", score), diffs


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
@dataclass
class Application:
    brand_name: str = ""
    class_type: str = ""
    abv: str = ""
    net_contents: str = ""
    producer: str = ""
    country_of_origin: str = ""


def verify(app: Application, ocr_text: str) -> dict:
    results: list[FieldResult] = [
        check_text_field("brand_name", "Brand name", app.brand_name, ocr_text),
        check_text_field("class_type", "Class / type", app.class_type, ocr_text, pass_threshold=90, warn_threshold=75),
        check_abv(app.abv, ocr_text),
        check_net_contents(app.net_contents, ocr_text),
        check_text_field("producer", "Producer / bottler", app.producer, ocr_text, pass_threshold=88, warn_threshold=70),
        check_text_field("country_of_origin", "Country of origin", app.country_of_origin, ocr_text),
    ]
    warning_result, warning_diffs = check_government_warning(ocr_text)
    results.append(warning_result)

    statuses = {r.status for r in results}
    if Status.FAIL in statuses:
        overall = Status.FAIL
    elif Status.WARN in statuses:
        overall = Status.WARN
    else:
        overall = Status.PASS

    return {
        "overall": overall.value,
        "fields": [r.to_dict() for r in results],
        "warning_diffs": warning_diffs,
        "counts": {s.value: sum(1 for r in results if r.status == s) for s in Status},
    }
