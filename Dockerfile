FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_WEIGHTS_PATH=/models/model.jane-austen-5.pth \
    MODEL_PRESET=book-124M \
    MODEL_DEVICE=cpu \
    WEB_CONCURRENCY=1

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app ./app
COPY llm_demo ./llm_demo

CMD ["sh", "-c", "exec gunicorn --bind \":${PORT}\" --workers \"${WEB_CONCURRENCY}\" --threads 1 --timeout 300 \"app:create_app()\""]
