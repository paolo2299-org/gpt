FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WEIGHTS_DIR=/weights \
    MODEL_DEVICE=cpu \
    WEB_CONCURRENCY=1

WORKDIR /app

COPY requirements-web.txt .
# Install CPU-only torch first so pip doesn't pull the ~2GB CUDA wheel from PyPI
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-web.txt

COPY app ./app
COPY llm ./llm

CMD ["sh", "-c", "exec gunicorn --bind \":${PORT}\" --workers \"${WEB_CONCURRENCY}\" --threads 1 --timeout 300 \"app:create_app()\""]
