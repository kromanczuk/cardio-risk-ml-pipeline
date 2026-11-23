# Streamlit front-end image (built from the frontend/ folder).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .streamlit/ .streamlit/
COPY app.py .
USER app
# Cloud Run sets PORT (8080 by default). API_URL must point at the deployed API.
ENV PORT=8080 \
    API_URL=http://127.0.0.1:8000
EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
