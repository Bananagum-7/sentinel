FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY sentinel ./sentinel
COPY config ./config
COPY dashboard.py .
RUN useradd -m -u 1000 app && mkdir -p /app/data && chown -R app /app
USER app
ENV PORT=8000 SENTINEL_DB=/app/data/requests.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
# One worker: the rate limiter and cache are in-process. Scale out with Redis-backed versions (see README).
CMD ["sh", "-c", "uvicorn sentinel.app:app --host 0.0.0.0 --port $PORT --workers 1"]
