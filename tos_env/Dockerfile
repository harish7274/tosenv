# ── ToS Risk Analyzer — OpenEnv Server Dockerfile ──────────────────────────
# Compatible with Hugging Face Spaces (runs as non-root on port 7860)
# and local development (port 8000).
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# HF Spaces expects the app to listen on 7860 by default.
# Override with PORT env var for local dev.
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (required by HF Spaces) ────────────────────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ──────────────────────────────────────────────────────
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──────────────────────────────────────────────────────────────
COPY . .

# ── Fix permissions ──────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app

USER appuser

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE $PORT

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start server ─────────────────────────────────────────────────────────────
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
