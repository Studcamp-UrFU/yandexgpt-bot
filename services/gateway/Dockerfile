FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/gateway/requirements.txt /app/requirements.txt
# ускоряем установку колёс
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY services/gateway/app.py /app/app.py

# адреса сервисов и параметры LLM можно переопределять на деплое
ENV SECURITY_URL="http://security-svc:8080" \
    MODERATION_URL="http://moderation-svc:8080" \
    RAG_URL="http://rag-svc:8080" \
    MODEL_URI="" \
    TEMPERATURE="0.6" \
    MAX_TOKENS="1000" \
    MAX_QUESTION_CHARS="8000" \
    FOLDER_ID="" \
    SERVICE_ACCOUNT_ID="" \
    KEY_ID="" \
    PRIVATE_KEY=""

# более безопасный запуск не от root
RUN useradd -m appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
