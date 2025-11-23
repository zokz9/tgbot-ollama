# core.py
import os
import base64
import asyncio
import logging
import re
from ollama import Client
from config import (
    OLLAMA_API_KEY, OLLAMA_HOST, 
    OLLAMA_MAIN_MODEL, OLLAMA_SEARCH_MODEL, OLLAMA_VISION_MODEL,
    MAX_SEARCH_CONTEXT, MAX_HISTORY_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    """Удаляет Markdown-разметку"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    return text

class OllamaProcessor:
    def __init__(self):
        self.client = Client(
            host=OLLAMA_HOST,
            headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
        )
        self.main_model = OLLAMA_MAIN_MODEL
        self.search_model = OLLAMA_SEARCH_MODEL
        self.vision_model = OLLAMA_VISION_MODEL

    async def should_search_internet(self, query: str, context: list) -> bool:
        """Определяет, нужен ли поиск с учетом контекста"""
        try:
            # ✅ ПРОВЕРКА КЛЮЧЕВЫХ СЛОВ (быстро и надежно)
            search_keywords = [
                'найди', 'поищи', 'ищи', 'search', 'найти',
                'какие новости', 'что нового', 'последние', 'новости',
                'актуальная информация', 'в интернете', 'online',
                'кто такой', 'что такое', 'где найти', 'когда будет',
                'обращайся к интернету', 'посмотри в сети', 'загугли',
                'для актуальности', 'последние данные', 'свежие новости',
                'сколько стоит', 'цена', 'купить', 'где купить'
            ]
            
            query_lower = query.lower().strip()
            if any(keyword in query_lower for keyword in search_keywords):
                logger.info(f"✅ Ключевое слово найдено: '{query[:50]}...'")
                return True
            
            # Если ключевых слов нет, спрашиваем модель
            context_text = ""
            if context:
                last_messages = context[-MAX_SEARCH_CONTEXT:]
                context_text = "Контекст беседы:\n" + "\n".join([
                    f"- {msg['content'][:80]}..." if len(msg['content']) > 80 else f"- {msg['content']}"
                    for msg in last_messages
                ]) + "\n\n"
            
            # ✅ Ясный промпт для модели
            check_messages = [{
                'role': 'user',
                'content': f'{context_text}Пользователь спрашивает: "{query}"\n\n' \
                          'Если для ответа НУЖНА актуальная информация из интернета, новости или свежие данные, ответь "ДА". ' \
                          'Если можно ответить без интернета, ответь "НЕТ". Ответь только одним словом.'
            }]
            
            logger.info(f"🔍 Проверка Kimi-K2: '{query[:50]}...'")
            response = self.client.chat(
                model=self.main_model,
                messages=check_messages,
                stream=False
            )
            
            result = response['message']['content'].strip().upper()
            should_search = "ДА" in result
            
            logger.info(f"🔍 Решение Kimi-K2: {'Искать' if should_search else 'Не искать'}")
            return should_search
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки поиска: {e}")
            return False

    async def chat_with_main(self, messages: list, stream: bool = False):
        """Общение с Kimi-K2"""
        try:
            system_prompt = {
                'role': 'system',
                'content': 'Ты полезный ассистент. Отвечай КРАТКО и по делу.'
            }
            
            messages_with_prompt = [system_prompt] + messages
            
            logger.info(f"📤 Запрос к Kimi-K2: {messages[-1]['content'][:50]}...")
            response = self.client.chat(
                model=self.main_model,
                messages=messages_with_prompt,
                stream=stream
            )
            
            if stream:
                full_response = ""
                for part in response:
                    content = part['message']['content']
                    full_response += content
                return clean_markdown(full_response)
            else:
                result = response['message']['content']
                logger.info(f"📥 Ответ Kimi-K2 ({len(result)} символов)")
                return clean_markdown(result)
                
        except Exception as e:
            logger.error(f"❌ Ошибка Kimi-K2: {e}")
            return f"❌ Ошибка при обращении к Kimi-K2: {str(e)}"

    async def analyze_image(self, image_path: str, user_text: str = "") -> str:
        """Анализ изображения через Qwen3-VL"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            vision_messages = [{
                'role': 'user',
                'content': 'Опиши это изображение КРАТКО и по делу (максимум 4-5 предложений). ' \
                          'Опиши только главные объекты, действия и важные детали.',
                'images': [base64_image]
            }]
            
            logger.info(f"📤 Анализ изображения через Qwen3-VL")
            vision_response = self.client.chat(
                model=self.vision_model,
                messages=vision_messages,
                stream=False
            )
            
            description = vision_response['message']['content']
            logger.info(f"📥 Описание Qwen3-VL ({len(description)} символов)")
            description = clean_markdown(description)
            
            context = f"Пользователь отправил изображение и написал: \"{user_text}\"\n\n" \
                     f"Описание изображения: {description}\n\n" \
                     f"Ответь КРАТКО на вопрос пользователя с учетом изображения."
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Ошибка Qwen3-VL: {e}")
            return f"❌ Ошибка при анализе изображения: {str(e)}"

    async def search_internet(self, query: str, context: list) -> tuple:
        """Поиск в интернете через GPT-OSS с учетом контекста"""
        try:
            context_text = ""
            used_context = []
            if context:
                last_messages = context[-MAX_SEARCH_CONTEXT:]
                used_context = last_messages
                context_text = "Контекст беседы:\n" + "\n".join([
                    f"- {msg['content'][:80]}..." if len(msg['content']) > 80 else f"- {msg['content']}"
                    for msg in last_messages
                ]) + "\n\n"
            
            search_prompt = f'{context_text}Вопрос: "{query}"\n\n' \
                           f'Используя контекст и интернет, найди актуальную информацию и дай КРАТКИЙ, информативный ответ (не более 5 предложений).'
            
            search_messages = [{
                'role': 'user',
                'content': search_prompt
            }]
            
            logger.info(f"🔍 Поиск через GPT-OSS: {query[:50]}...")
            search_response = self.client.chat(
                model=self.search_model,
                messages=search_messages,
                stream=False,
                tools=[{'type': 'search'}]
            )
            
            search_result = search_response['message']['content']
            logger.info(f"📥 Результат GPT-OSS ({len(search_result)} символов)")
            
            return search_result, used_context
            
        except Exception as e:
            logger.error(f"❌ Ошибка GPT-OSS: {e}")
            return f"❌ Ошибка при поиске в интернете: {str(e)}", []