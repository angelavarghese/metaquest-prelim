FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal OS packages for validation/debug flows.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Run as non-root user for safer deployment.
RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-7860}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
