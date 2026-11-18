FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 FEAST_USAGE=False
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-runtime.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY scripts /app/scripts
COPY feature_repo/*.py feature_repo/*.yaml /app/feature_repo/
COPY config.yaml /app/config.yaml
COPY docker/app.sql /app/docker/app.sql
RUN mkdir -p /app/data/processed /app/data/raw /app/feature_repo/data
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS graph
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-geometric==2.5.3
