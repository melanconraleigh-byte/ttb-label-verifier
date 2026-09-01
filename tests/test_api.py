"""End-to-end tests through the HTTP API against the generated sample labels (needs Tesseract)."""
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
client = TestClient(app)

EXPECTED = {
    "01_bourbon_clean.png": "pass",
    "02_bourbon_abv_mismatch.png": "fail",
    "03_bourbon_titlecase_warning.png": "fail",
    "04_stones_throw_case.png": "pass",
    "05_bourbon_altered_warning.png": "fail",
    "06_bourbon_rotated.png": "pass",
    "07_bourbon_photo.jpg": "pass",
    "08_wine_missing_warning.png": "fail",
    "09_gin_missing_net_contents.png": "fail",
}


def _manifest() -> dict[str, dict]:
    with open(SAMPLES / "manifest.csv") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


@pytest.mark.skipif(not SAMPLES.exists(), reason="run scripts/make_samples.py first")
@pytest.mark.parametrize("filename,expected", EXPECTED.items())
def test_sample_labels_classify_correctly(filename, expected):
    row = _manifest()[filename]
    with open(SAMPLES / filename, "rb") as f:
        res = client.post("/api/verify", files={"image": (filename, f, "image/png")},
                          data={k: v for k, v in row.items() if k != "filename"})
    assert res.status_code == 200
    body = res.json()
    assert body["overall"] == expected, body["fields"]
    assert body["timing_ms"] < 5000, "Sarah's rule: results in ~5 seconds"


def test_specific_failure_reasons():
    m = _manifest()
    def run(name):
        with open(SAMPLES / name, "rb") as f:
            return {x["field"]: x for x in client.post("/api/verify", files={"image": (name, f, "image/png")},
                    data={k: v for k, v in m[name].items() if k != "filename"}).json()["fields"]}
    assert run("02_bourbon_abv_mismatch.png")["abv"]["status"] == "fail"
    assert "capital letters" in run("03_bourbon_titlecase_warning.png")["government_warning"]["message"]
    assert run("09_gin_missing_net_contents.png")["net_contents"]["status"] == "fail"


def test_batch_with_manifest():
    files = [("images", (n, open(SAMPLES / n, "rb"), "image/png")) for n in ("01_bourbon_clean.png", "02_bourbon_abv_mismatch.png")]
    files.append(("manifest", ("manifest.csv", open(SAMPLES / "manifest.csv", "rb"), "text/csv")))
    res = client.post("/api/verify-batch", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["summary"] == {"pass": 1, "warn": 0, "fail": 1, "error": 0}


def test_batch_file_missing_from_manifest_is_reported_not_crashed():
    files = [("images", ("unknown.png", open(SAMPLES / "01_bourbon_clean.png", "rb"), "image/png")),
             ("manifest", ("manifest.csv", open(SAMPLES / "manifest.csv", "rb"), "text/csv"))]
    body = client.post("/api/verify-batch", files=files).json()
    assert body["results"][0]["overall"] == "error"
    assert "manifest" in body["results"][0]["error"]


def test_rejects_non_image():
    res = client.post("/api/verify", files={"image": ("x.txt", b"hello", "text/plain")}, data={})
    assert res.status_code == 415


def test_corrupt_image_returns_error_result_not_500():
    res = client.post("/api/verify", files={"image": ("bad.png", b"\x89PNG garbage", "image/png")}, data={})
    assert res.status_code == 200
    assert res.json()["overall"] == "error"
