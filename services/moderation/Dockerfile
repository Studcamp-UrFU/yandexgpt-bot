FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/moderation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/moderation/app.py /app/app.py

# токены/ключи задашь в .env или в compose
ENV FOLDER_ID="" SERVICE_ACCOUNT_ID="" KEY_ID="" PRIVATE_KEY=""

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
