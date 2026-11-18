FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN pip install --no-cache-dir psycopg2-binary
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1
