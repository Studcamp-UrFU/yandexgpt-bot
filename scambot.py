import logging
import os
import time

from dotenv import load_dotenv
import jwt
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag import RAG
from security import detect_injection, get_detected_pattern
from moderator import init_moderator, is_malicious_prompt


load_dotenv()

SERVICE_ACCOUNT_ID = os.getenv("SERVICE_ACCOUNT_ID")
KEY_ID = os.getenv("KEY_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
FOLDER_ID = os.getenv("FOLDER_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

_rag = RAG().open()

class YandexGPTBot:
    def __init__(self):
        self.iam_token = None
        self.token_expires = 0

    def get_iam_token(self):
        """Получение IAM-токена (с кэшированием на 3500 секунд)."""
        if self.iam_token and time.time() < self.token_expires:
            return self.iam_token

        try:
            now = int(time.time())
            payload = {
                'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                'iss': SERVICE_ACCOUNT_ID,
                'iat': now,
                'exp': now + 360 
            }

            pk = PRIVATE_KEY.replace("\\n", "\n") if PRIVATE_KEY else PRIVATE_KEY
            encoded_token = jwt.encode(
                payload,
                pk,
                algorithm='PS256',
                headers={'kid': KEY_ID}
)

            response = requests.post(
                'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                json={'jwt': encoded_token},
                timeout=10
            )

            if response.status_code != 200:
                raise Exception(f"Ошибка генерации токена: {response.text}")

            token_data = response.json()
            self.iam_token = token_data['iamToken']
            self.token_expires = now + 3500 

            logger.info("IAM token generated successfully")
            return self.iam_token

        except Exception as e:
            logger.error(f"Error generating IAM token: {str(e)}")
            raise

    def ask_gpt(self, question):
        """
        Запрос к Yandex GPT API с системным промптом с RAG-контекстом.
        """
        try:
            iam_token = self.get_iam_token()

            try:
                context_text = _rag.context(question, k=4, max_chars=2500)
            except Exception:
                context_text = ""

            system_prompt = (
                "Ты — корпоративный ассистент. Отвечай строго по документам. "
                "Если информации нет — скажи 'В документах не указано'.\n\n"
                "Контекст из документов:\n{context_text}"
            ).format(context_text=context_text)

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {iam_token}',
                'x-folder-id': FOLDER_ID
            }

            data = {
                "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.6,
                    "maxTokens": 1000
                },
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user",   "text": question}
                ]
            }

            response = requests.post(
                'https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Yandex GPT API error: {response.text}")
                raise Exception(f"Ошибка API: {response.status_code}")

            return response.json()['result']['alternatives'][0]['message']['text']

        except Exception as e:
            logger.error(f"Error in ask_gpt: {str(e)}")
            raise


yandex_bot = YandexGPTBot()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для работы с Yandex GPT. Просто напиши мне свой вопрос"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text or ""

    if not user_message.strip():
        await update.message.reply_text("Пожалуйста, введите вопрос")
        return

    if detect_injection(user_message):
        detected_pattern = get_detected_pattern(user_message)
        logger.warning(
            f"Обнаружена попытка промпт-инъекции: '{detected_pattern}' в сообщении: '{user_message}'"
        )
        await update.message.reply_text("ban")
        return

    if is_malicious_prompt(user_message):
        logger.warning(
            f"Модель-модератор заблокировала запрос от {update.effective_user.id} ({update.effective_user.username}): '{user_message[:100]}...'")
        await update.message.reply_text(
            "ban2"
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        response = yandex_bot.ask_gpt(user_message)
        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        await update.message.reply_text(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте позже."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "Произошла ошибка. Пожалуйста, попробуйте позже."
        )


def main():
    try:
        yandex_bot.get_iam_token()
        logger.info("IAM token test successful")

        init_moderator(FOLDER_ID, yandex_bot.get_iam_token)
        logger.info("Moderator initialized")

        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_error_handler(error_handler)

        logger.info("Бот запускается...")
        application.run_polling()

    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")


if __name__ == "__main__":
    main()
