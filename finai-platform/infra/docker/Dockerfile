# syntax=docker/dockerfile:1
# ==========================================================================
# FinAI Platform - production image
# Multi-stage: wheels are built once, the runtime layer stays slim and non-root.
# ==========================================================================

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt


# --------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r finai && useradd -r -g finai -d /app finai

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/data/{cache,models,artifacts} && chown -R finai:finai /app

USER finai
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
