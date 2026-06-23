# SimCity API image (FastAPI + WebSockets + GraphQL).
# CPU-only inference image, sized for the Oracle Cloud Always-Free tier.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# System deps occasionally needed by scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise layer caching.
COPY requirements.txt .
# Install the CPU build of torch to keep the image small and free of CUDA.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Copy the application source.
COPY . .

EXPOSE 8000

# Container-level health check used by Docker/Kubernetes probes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
