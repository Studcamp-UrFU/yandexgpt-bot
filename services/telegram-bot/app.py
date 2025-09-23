import os
import logging
import asyncio

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("telegram-bot")

GATEWAY_URL = "http://api-gateway:8080"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

MAX_QUESTION_CHARS = 8000
REQUEST_TIMEOUT_S = 60
RETRIES = 1

def _chunks(s: str, n: int = 4096):
    for i in range(0, len(s), n):
        yield s[i:i + n]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Просто напиши вопрос.")

async def _ask_gateway(question: str) -> str:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        last_err = None
        for attempt in range(RETRIES + 1):
            try:
                r = await client.post(f"{GATEWAY_URL}/ask", json={"question": question})
                r.raise_for_status()
                return r.json().get("answer", "Упс, пустой ответ")
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.3 * (attempt + 1))
        raise last_err

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        await update.message.reply_text("Пустой запрос :(")
        return

    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
        log.info("trimmed user input to %d chars", MAX_QUESTION_CHARS)

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        ans = await _ask_gateway(q)
        for part in _chunks(ans, 4096):
            await update.message.reply_text(part)
    except Exception as e:
        log.exception("gateway error: %s", e)
        await update.message.reply_text("Не получилось, попробуй позже")

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
