# Approach, decisions, and trade-offs

## What the interviews actually asked for

Reading the discovery notes as requirements rather than colour:

| Who | What they said | What it means for the build |
|---|---|---|
| Sarah | Scanning-vendor pilot took 30–40 s per label; agents abandoned it. "About 5 seconds." | Hard latency budget. Rules out per-label round-trips to hosted vision models under load. Measured here: 0.6–1.9 s for a clean label, ≤4.1 s when a rotation retry is needed. |
| Marcus | Firewall blocks outbound to ML endpoints; the vendor pilot half-broke because of it. | OCR must run in-process. Tesseract is bundled in the container; there is no network dependency at all. |
| Marcus | Standalone proof-of-concept, no COLA integration, nothing sensitive stored. | No database, no auth, no persistence. Images live in memory for the duration of one request. |
| Dave | "STONE'S THROW" vs "Stone's Throw" is obviously the same thing. Needs judgment. | Three-tier results (pass / needs a look / problem) instead of binary. Case, punctuation, and accent differences on names are passes with a note; OCR-noise-level differences are "needs a look"; only real discrepancies fail. |
| Jenny | Warning must be word-for-word; "GOVERNMENT WARNING:" must be all caps and bold; "Government Warning" in title case was rejected. | Warning check is strict on wording and on header capitalisation, with a word-level diff so the agent sees exactly what changed. Bold is flagged as a manual check because OCR cannot verify it reliably (see limitations). |
| Jenny | Angled, badly lit, glare-affected photos. | EXIF orientation, orientation detection (OSD), upscaling, Otsu → CLAHE + adaptive threshold escalation. Handles the synthetic phone-photo sample; real-world coverage is partial (see limitations). |
| Sarah | 200–300 label dumps from importers; Janet has asked for batch for years. | Batch tab: many images + CSV manifest keyed by filename, chunked uploads so results stream in, problems sorted to the top, CSV export. |
| Sarah | "Something my mother could figure out." Half the team over 50. | Two tabs, two steps each, one big button, 17 px base font, three colours that mean three things, plain-language status words, no icons without labels. |

## Why local OCR instead of a vision LLM

A multimodal model (Claude, GPT-4o) would read messy photos better and would understand "the same thing" the way Dave does. I still didn't use one, for three reasons that come straight from the notes: the firewall would block it (Marcus), 300-label batches at 3–8 s per call blow the latency budget (Sarah), and Tesseract's output is deterministic and explainable, which matters for a tool whose findings a federal agent puts their name behind. The rules engine is pure Python with no model in the loop, so every pass/fail can be traced to a line of code.

The design leaves the door open: `app/ocr.py` exposes one function, `extract_text(bytes) -> OcrResult`. A vision-model backend could be swapped in behind that function (or used only as a fallback when Tesseract confidence is low) without touching the rules or the UI.

## Matching rules

All comparisons run on normalised text: NFKD-decomposed, accents stripped, curly quotes and dashes unified, whitespace collapsed, upper-cased.

**Brand name, class/type, producer, country.** Exact case-sensitive containment → pass. Case-insensitive containment → pass with a note (TTB does not require case to match on these fields). Containment ignoring punctuation and spacing → *needs a look* (OCR routinely drops apostrophes). Otherwise the best-matching window of the OCR text is scored with RapidFuzz's Levenshtein ratio: ≥92 % *needs a look* as likely OCR noise, ≥80 % *needs a look* as partial, below that a failure. Class/type and producer use slightly lower thresholds because they are longer strings.

**Alcohol content.** Percentages are parsed from both sides; the label side prefers a percentage that sits within a few characters of "ALC", "VOL" or "ABV" so that "100% corn" is not mistaken for ABV. Exact numeric match passes. If a proof figure is also on the label it is checked against 2×ABV and inconsistency is flagged. A miss of ≤1 point is *needs a look*, because Tesseract reads 5/6 and drops decimal points.

**Net contents.** Parsed to millilitres so 750 mL, 75 cl and 25.4 fl oz all compare. Exact match passes; anything else fails.

**Government warning.** The 27 CFR 16.21 text is compiled into the app; nobody types it. The header is located case-insensitively so that a title-case header can be reported as a specific failure ("must be in capital letters"). The statement body is then compared word-for-word. If every letter and digit matches but punctuation differs, it passes with a note that punctuation should be confirmed by eye, because Tesseract drops a comma at almost every line wrap. A difference that consists only of same-length word swaps where each word is a near-miss of the original (machinery → machinory) is *needs a look*. Any inserted, deleted, or substantively different word fails, however high the overall character similarity. I found that last rule the hard way: "should not drink" → "should avoid drinking" is 97 % character-similar, and my first version waved it through as noise. The test suite now pins it.

## OCR pipeline

Tesseract 5 with page-segmentation mode 3 (automatic layout). Mode 6, which I tried first, silently dropped the oversized brand-name line on every sample; this is measured, not assumed. Images are upscaled to at least 1200 px wide (Tesseract wants roughly 30 px x-height) and downscaled from above 2400 px so a 12-megapixel phone photo does not take 6 seconds.

The escalation is cheap-first: one pass with Otsu binarisation. Only if mean word confidence is below 55 does the app run orientation detection on a half-size copy (~0.35 s) and retry rotated, then retry with CLAHE and adaptive thresholding for uneven lighting. The best-scoring attempt wins. On the samples, 8 of 9 finish in one attempt.

## What I deliberately left out

- **Bold detection on "GOVERNMENT WARNING:".** Stroke-width analysis on the header's bounding box would be the approach; it is fragile across fonts and photo quality, and a wrong "not bold" verdict is worse than an honest "check by eye". The UI says so on every warning result.
- **Font-size and contrast rules (27 CFR 16.22).** Minimum type sizes depend on container volume and would need a physical-scale reference in the image. Out of scope for a prototype.
- **Beverage-type-specific rules** (e.g. wines under 7 % ABV, malt beverages where ABV is optional). The rules engine skips a field when the application leaves it blank, which covers the common cases without encoding the full regulation.
- **Persistence, accounts, audit log.** Marcus asked for none of it and the retention rules for a real deployment would drive the design; guessing now would be wasted.
- **A front-end framework.** One HTML file with ~250 lines of JavaScript, no build step. Easier to hand to a .NET shop than a Node toolchain.

## Known limitations

- Real photos of curved bottles, heavy glare, or embossed/foil text will produce low-confidence OCR. The app says so and asks for a better image rather than guessing; that is the same thing an agent does today, just faster.
- Highly stylised brand typography (script fonts, stacked letters, outlined text) may not OCR at all. A missing brand name is reported as a failure, which the agent will recognise as an OCR miss because the label preview is right there.
- Batch processing is sequential on one CPU: a 300-label dump takes roughly 8 minutes end to end, streamed in chunks of ten so the agent sees results immediately. Horizontal scaling is just more uvicorn workers or more containers.
- The sample labels are generated, not real. They were built to exercise each rule, not to represent the full variety of commercial artwork. Real COLA images are the obvious next test set.

## Tools used

Python 3.11, FastAPI, Tesseract 5 via pytesseract, OpenCV (preprocessing), RapidFuzz (Levenshtein), Pillow (sample generation), pytest, Docker.
