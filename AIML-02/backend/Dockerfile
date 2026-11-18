FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
WORKDIR /app
COPY backend/requirements-lock.txt backend/requirements-lock.txt
RUN pip install --no-cache-dir -r backend/requirements-lock.txt
ARG INSTALL_ML=false
COPY ml/requirements-serving-lock.txt ml/requirements-serving-lock.txt
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r ml/requirements-serving-lock.txt; fi
COPY backend backend
COPY ml ml
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
