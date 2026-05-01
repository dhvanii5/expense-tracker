# Expense Tracker backend image

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY backend/requirements.deploy.txt .

# Install to a prefix dir so we can COPY it cleanly to final stage
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --prefix=/install \
        -r requirements.deploy.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime-only system libs (no build tools, no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=9002 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    CHROMA_DB_PATH=/data/finance_memory

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/ ./

VOLUME ["/data"]

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && mkdir -p /data /app/.cache/huggingface \
    && chown -R appuser:appgroup /app /data
USER appuser

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9002/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9002", "--workers", "1"]
