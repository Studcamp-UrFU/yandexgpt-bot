import os
import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tg-bot")

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://api-gateway:8080")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("привет! просто напиши вопрос.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        await update.message.reply_text("пустой запрос :(")
        return
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        r = requests.post(f"{GATEWAY_URL}/ask", json={"question": q}, timeout=60)
        r.raise_for_status()
        ans = r.json().get("answer", "упс, пустой ответ")
        await update.message.reply_text(ans)
    except Exception as e:
        log.exception("gateway error")
        await update.message.reply_text("не получилось, попробуй позже")

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN not set")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, poll_interval=1.5)

if __name__ == "__main__":
    main()
