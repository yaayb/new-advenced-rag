FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV EVIDENCELAB_DB_PATH=/data/evidencelab.db \
    EVIDENCELAB_EVAL_DATA_DIR=/eval_data
EXPOSE 8000
CMD ["uvicorn", "evidencelab.api:app", "--host", "0.0.0.0", "--port", "8000"]
