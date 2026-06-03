FROM python:3.11-slim

# Install system dependencies (libgl1 replaces deprecated libgl1-mesa-glx on Debian Trixie+)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Production dependencies only (no llama-cpp-python — Gemini LLM in prod)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Application code
COPY app/ ./app/

# Bake YOLO weights into image (avoids runtime download)
RUN mkdir -p /app/models \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && mv yolo11n.pt /app/models/yolo11n.pt

# Bake EasyOCR weights into image (avoids runtime download on first request)
RUN mkdir -p /app/models/easyocr \
    && python -c "import easyocr; easyocr.Reader(['vi','en'], gpu=False, model_storage_directory='/app/models/easyocr')"

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV AI_HOST=0.0.0.0
ENV AI_PORT=8000
ENV AI_ENVIRONMENT=production
ENV AI_DEBUG=false
ENV AI_LOG_LEVEL=INFO
ENV AI_LOG_FORMAT=json
ENV AI_LLM_PROVIDER=gemini
ENV AI_YOLO_MODEL_PATH=/app/models/yolo11n.pt
ENV AI_OCR_MODEL_PATH=/app/models/easyocr
ENV AI_ALLOWED_ORIGINS=

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
