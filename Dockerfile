FROM python:3.11-slim

# Tesseract + the English and orientation (osd) data are the only system deps.
# libgl/libglib are needed by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY samples ./samples

ENV PORT=8000
EXPOSE 8000
# Single worker: Tesseract is CPU-bound and the free tiers on Render/Fly/Railway have one vCPU.
# Bump --workers on a bigger box; each worker handles one label at a time.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
