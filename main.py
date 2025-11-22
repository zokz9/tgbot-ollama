# main.py
import os
import asyncio
import tempfile
import logging
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes
)
from core import OllamaProcessor
from config import TELEGRAM_TOKEN, MAX_IMAGE_SIZE

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

user_contexts = {}
context_lock = asyncio.Lock()

ollama = OllamaProcessor()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Привет! Я бот с интеграцией Ollama.\n\n"
        "📝 Что умею:\n"
        "• Просто пишите сообщения - общение с Kimi-K2\n"
        "• Отправляйте изображения - я их проанализирую\n"
        "• /search <запрос> - поиск в интернете\n"
        "• /clear - очистить контекст беседы\n\n"
        "💡 Поддерживается контекст беседы!"
    )
    await update.message.reply_text(welcome_text)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ Укажите поисковый запрос.\n"
            "Пример: /search последние новости в мире ИИ"
        )
        return
    
    query = " ".join(context.args)
    status_msg = await update.message.reply_text("🔍 Поиск в интернете...")
    
    result = await ollama.search_internet(query)
    
    # Убраны *, parse_mode и другая Markdown-разметка
    response_text = f"🔎 Результаты поиска: \"{query}\"\n\n{result}"
    
    if len(response_text) > 4096:
        parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
        for i, part in enumerate(parts):
            if i == 0:
                await status_msg.edit_text(part)
            else:
                await update.message.reply_text(part)
    else:
        await status_msg.edit_text(response_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""
    
    async with context_lock:
        if user_id not in user_contexts:
            user_contexts[user_id] = []
        
        user_contexts[user_id].append({'role': 'user', 'content': text})
        
        if len(user_contexts[user_id]) > 10:
            user_contexts[user_id] = user_contexts[user_id][-10:]
        
        messages = user_contexts[user_id].copy()
    
    status_msg = await update.message.reply_text("🤔 Думаю...")
    
    try:
        response = await ollama.chat_with_main(messages, stream=False)
        
        async with context_lock:
            user_contexts[user_id].append({'role': 'assistant', 'content': response})
        
        if len(response) > 4096:
            parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    await status_msg.edit_text(part)
                else:
                    await update.message.reply_text(part)
        else:
            await status_msg.edit_text(response)
            
    except Exception as e:
        logger.error(f"❌ Ошибка обработки сообщения: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    caption = update.message.caption or ""
    photo = update.message.photo[-1]
    
    if photo.file_size > MAX_IMAGE_SIZE:
        await update.message.reply_text("⚠️ Изображение слишком большое. Максимум 10МБ.")
        return
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    status_msg = await update.message.reply_text("👁️ Анализирую изображение...")
    
    try:
        photo_file = await photo.get_file()
        await photo_file.download_to_drive(tmp_path)
        
        logger.info(f"📥 Изображение загружено: {tmp_path} ({photo.file_size} bytes)")
        
        context_text = await ollama.analyze_image(tmp_path, caption)
        
        async with context_lock:
            if user_id not in user_contexts:
                user_contexts[user_id] = []
            user_contexts[user_id].append({'role': 'user', 'content': context_text})
        
        response = await ollama.chat_with_main([{'role': 'user', 'content': context_text}], stream=False)
        
        async with context_lock:
            user_contexts[user_id].append({'role': 'assistant', 'content': response})
        
        await status_msg.edit_text(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа изображения: {e}")
        await status_msg.edit_text(f"❌ Ошибка при анализе: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"🗑️ Временный файл удален: {tmp_path}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    async with context_lock:
        if user_id in user_contexts:
            del user_contexts[user_id]
    
    await update.message.reply_text("🗑️ Контекст беседы очищен!")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ Ошибка в боте: {context.error}", exc_info=True)
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуйте еще раз."
        )

def main():
    logger.info("🚀 Запуск Telegram-бота...")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    logger.info("✅ Бот запущен и работает!")
    application.run_polling()

if __name__ == "__main__":
    main()