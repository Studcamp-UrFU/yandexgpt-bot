import httpx
import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


GATEWAY_URL = "http://api-gateway:8080"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        await update.message.reply_text("Пустой запрос :(")
        return
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{GATEWAY_URL}/ask", json={"question": q})
            r.raise_for_status()
            ans = r.json().get("answer", "упс, пустой ответ")
        await update.message.reply_text(ans)
    except Exception:
        log.exception("gateway error")
        await update.message.reply_text("Не получилось, попробуй позже")


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
