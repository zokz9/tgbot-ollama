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
from config import (
    TELEGRAM_TOKEN, MAX_IMAGE_SIZE, MAX_HISTORY_SIZE, MAX_SEARCH_CONTEXT,
    OLLAMA_MAIN_MODEL, OLLAMA_SEARCH_MODEL, OLLAMA_VISION_MODEL
)

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
        "👋 Привет! Я бот ZoKzGPT с тремя ИИ-моделями.\n\n"
        "📝 Что умею:\n"
        "• Общение с Kimi-K2\n"
        "• Анализ изображений Qwen3\n"
        "• Поиск в интернете GPT-4\n\n"
        "Задай любой вопрос я отвечу!"
    )
    await update.message.reply_text(welcome_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""
    
    logger.info(f"📩 Сообщение: '{text[:50]}...'")
    
    async with context_lock:
        current_context = user_contexts.get(user_id, []).copy()
    
    need_search = await ollama.should_search_internet(text, current_context)
    logger.info(f"🔍 Искать? {need_search}")
    
    if need_search:
        status_msg = await update.message.reply_text("🔍 Ищу в интернете...")
        
        result, used_context = await ollama.search_internet(text, current_context)
        
        # ✅ Сохраняем в историю с пометками, но НЕ показываем их в чате
        async with context_lock:
            if user_id not in user_contexts:
                user_contexts[user_id] = []
            
            # Внутри истории - с пометками
            user_contexts[user_id].append({
                'role': 'user', 
                'content': f"[ВОПРОС] {text}"
            })
            
            user_contexts[user_id].append({
                'role': 'assistant', 
                'content': f"[ИНТЕРНЕТ] {result}"
            })
            
            if len(user_contexts[user_id]) > MAX_HISTORY_SIZE:
                user_contexts[user_id] = user_contexts[user_id][-MAX_HISTORY_SIZE:]
        
        # В чат отправляем чистый результат
        if len(result) > 4096:
            parts = [result[i:i+4000] for i in range(0, len(result), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    await status_msg.edit_text(part)
                else:
                    await update.message.reply_text(part)
        else:
            await status_msg.edit_text(result)
            
    else:
        async with context_lock:
            if user_id not in user_contexts:
                user_contexts[user_id] = []
            
            user_contexts[user_id].append({'role': 'user', 'content': text})
            
            if len(user_contexts[user_id]) > MAX_HISTORY_SIZE:
                user_contexts[user_id] = user_contexts[user_id][-MAX_HISTORY_SIZE:]
            
            messages = user_contexts[user_id].copy()
        
        status_msg = await update.message.reply_text("🤔 Думаю...")
        
        try:
            response = await ollama.chat_with_main(messages, stream=False)
            
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
            logger.error(f"❌ Ошибка: {e}")
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
        
        # Анализ изображения
        context_text = await ollama.analyze_image(tmp_path, caption)
        
        # ✅ Сохраняем промежуточное описание в историю
        async with context_lock:
            if user_id not in user_contexts:
                user_contexts[user_id] = []
            
            user_contexts[user_id].append({
                'role': 'user', 
                'content': f"[ФОТО] {caption}"
            })
        
        # Генерируем финальный ответ
        response = await ollama.chat_with_main([{'role': 'user', 'content': context_text}], stream=False)
        
        # ✅ Сохраняем ответ в историю
        async with context_lock:
            user_contexts[user_id].append({
                'role': 'assistant', 
                'content': f"[ОТВЕТ] {response}"
            })
            
            if len(user_contexts[user_id]) > MAX_HISTORY_SIZE:
                user_contexts[user_id] = user_contexts[user_id][-MAX_HISTORY_SIZE:]
        
        # В чат отправляем только финальный ответ
        await status_msg.edit_text(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        await status_msg.edit_text(f"❌ Ошибка при анализе: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"🗑️ Файл удален: {tmp_path}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    async with context_lock:
        if user_id in user_contexts:
            del user_contexts[user_id]
    
    await update.message.reply_text("🗑️ Контекст беседы очищен!")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ Ошибка: {context.error}", exc_info=True)
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла ошибка. Попробуйте еще раз."
        )

def main():
    logger.info("🚀 Запуск Telegram-бота...")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    logger.info("✅ Бот запущен и работает!")
    logger.info(f"📊 Настройки: история={MAX_HISTORY_SIZE}, контекст поиска={MAX_SEARCH_CONTEXT}")
    logger.info(f"🤖 Модели: {OLLAMA_MAIN_MODEL} | {OLLAMA_SEARCH_MODEL} | {OLLAMA_VISION_MODEL}")
    application.run_polling()

if __name__ == "__main__":
    main()