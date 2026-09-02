# TTB Label Verifier

**Live prototype:** https://ttb-label-verifier-xw1m.onrender.com/ &nbsp;|&nbsp; **Source:** https://github.com/melanconraleigh-byte/ttb-label-verifier

A prototype that reads an alcohol-beverage label image, extracts the text with local OCR, and checks it against the fields from the COLA application: brand name, class/type, alcohol content, net contents, producer, country of origin, and the mandatory Government Warning statement. Single labels or batches of up to 300.

Runs entirely on the server. No image or application data leaves the machine, no API keys, no outbound network calls.

## Run it

**Docker (matches the deployed build):**

```bash
docker build -t ttb-label-verifier .
docker run -p 8000:8000 ttb-label-verifier
# open http://localhost:8000
```

**Locally without Docker** (needs Python 3.11+ and Tesseract 5 on your PATH: `apt install tesseract-ocr tesseract-ocr-osd` / `brew install tesseract`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_samples.py      # regenerates samples/*.png + manifest.csv (already committed)
uvicorn app.main:app --reload
```

**Tests:**

```bash
pytest            # 35 tests: rule unit tests + end-to-end through the API on the sample labels
python scripts/run_samples.py -v   # prints per-field results, timings and raw OCR for every sample
```

## Deploy

The repo carries a `Dockerfile`, a `render.yaml` and a `fly.toml`. Any of these works:

| Platform | Steps |
|---|---|
| Render | New → Blueprint → pick this repo. Pick a paid instance: a clean label measured 0.9 s there. The free tier is 0.1 CPU and too slow for the 5-second target. |
| Fly.io | `fly launch --copy-config --no-deploy && fly deploy` |
| Railway | New project → Deploy from GitHub → it detects the Dockerfile. |

Set `PORT` if the platform injects a different one; the container reads it.

## Using it

**Check one label.** Drop the image, type what the application says, press the button. Every field comes back as *Looks good*, *Needs a look*, or *Problem found* with a one-line reason, the value from the application, and the value read off the label. The Government Warning is always checked and shows a word-level diff when it is wrong. The sample buttons load bundled test labels with their application data pre-filled.

**Check a batch.** Drop all the images plus a CSV manifest (`filename, brand_name, class_type, abv, net_contents, producer, country_of_origin` — an example is linked in the UI). Results stream in ten at a time, sorted with problems first, and can be downloaded as CSV. Without a manifest, the fields typed on the single-label tab are applied to every image.

**API.** `POST /api/verify` (multipart: `image` + form fields) and `POST /api/verify-batch` (`images[]`, optional `manifest`). Interactive docs at `/docs`.

## How it works

```
image ──▶ preprocess (grayscale, resize, denoise, Otsu) ──▶ Tesseract ──▶ text + confidence
                          │ low confidence?                        │
                          ├─▶ OSD orientation → rotate → retry     ▼
                          └─▶ CLAHE + adaptive threshold → retry   rules engine ──▶ per-field PASS / WARN / FAIL
```

See [docs/APPROACH.md](docs/APPROACH.md) for the reasoning behind each decision, the matching rules, what was deliberately left out, and known limitations.

## Layout

```
app/ocr.py       preprocessing + Tesseract, escalation strategy, timing
app/verify.py    matching rules, one function per field, three-tier status
app/main.py      FastAPI: /, /api/verify, /api/verify-batch, /api/health
static/index.html  the whole UI (no build step, no framework)
scripts/make_samples.py  generates the nine test labels and manifest.csv
scripts/run_samples.py   CLI report over the samples
tests/           unit tests for rules, end-to-end tests through the API
```
