# syntax=docker/dockerfile:1
# One Dockerfile, two images:
#   docker build -t heart-api .                       -> API (default: the last stage)
#   docker build --target train -t heart-train .      -> training pipeline (full deps)
# See the Makefile for ready-made build/run commands.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- shared base
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
# libgomp1: OpenMP runtime needed by xgboost (and loading xgboost models).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app

# ---------------------------------------------------------------- training image
FROM base AS train
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY config.yaml ./
COPY src/ src/
COPY scripts/ scripts/
USER app
# Extra args go to train.py, e.g. `docker run heart-train --model-type xgboost`.
ENTRYPOINT ["python", "-m", "src.train"]

# ---------------------------------------------------------------- API image (default)
FROM base AS api
# Dependencies first, code second: code changes don't reinstall packages.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY config.yaml ./
COPY src/ src/
USER app
# Cloud Run sets PORT (8080 by default); keep the same default locally.
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health')"
# sh -c so $PORT is expanded; exec so uvicorn receives stop signals directly.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
