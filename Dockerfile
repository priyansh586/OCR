FROM python:3.11-slim

# Install the tesseract-ocr system binary (not available via pip alone)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render sets $PORT at runtime; gunicorn binds to it
CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app
