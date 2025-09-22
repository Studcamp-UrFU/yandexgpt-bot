FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/telegram-bot/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/telegram-bot/app.py /app/app.py

ENV TELEGRAM_TOKEN="" \
    GATEWAY_URL="http://api-gateway:8080"

CMD ["python", "app.py"]
