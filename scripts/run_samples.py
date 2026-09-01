"""Run every sample through OCR + verification and print a compact report with timings."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ocr import extract_text  # noqa: E402
from app.verify import Application, verify  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

with open(SAMPLES / "manifest.csv") as f:
    for row in csv.DictReader(f):
        data = (SAMPLES / row["filename"]).read_bytes()
        ocr = extract_text(data)
        app = Application(**{k: v for k, v in row.items() if k != "filename"})
        res = verify(app, ocr.text)
        print(f"\n=== {row['filename']}  overall={res['overall'].upper()}  "
              f"{ocr.elapsed_ms}ms  conf={ocr.confidence}  rot={ocr.rotation_applied}  attempts={ocr.attempts}")
        for fld in res["fields"]:
            if fld["status"] != "skip":
                print(f"  [{fld['status']:4}] {fld['label']:<20} {fld['message']}   found={fld['found'][:60]!r}")
        if "-v" in sys.argv:
            print("  --- OCR ---\n  " + ocr.text.replace("\n", "\n  "))
