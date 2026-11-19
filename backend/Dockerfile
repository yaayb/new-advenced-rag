FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app:app /data
USER app
ENV EVIDENCELAB_DB_PATH=/data/evidencelab.db \
    EVIDENCELAB_EVAL_DATA_DIR=/eval_data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status==200 else 1)"
CMD ["uvicorn", "evidencelab.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
