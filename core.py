# core.py
import os
import base64
import asyncio
import logging
import re
from ollama import Client
from config import (
    OLLAMA_API_KEY, OLLAMA_HOST, 
    OLLAMA_MAIN_MODEL, OLLAMA_VISION_MODEL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    """Удаляет Markdown-разметку из текста нейросети"""
    # Удаляем жирный, курсив, код, ссылки, заголовки
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # списки
    return text

class OllamaProcessor:
    def __init__(self):
        self.client = Client(
            host=OLLAMA_HOST,
            headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
        )
        self.main_model = OLLAMA_MAIN_MODEL
        self.vision_model = OLLAMA_VISION_MODEL

    async def chat_with_main(self, messages: list, stream: bool = False):
        try:
            logger.info(f"📤 Запрос к {self.main_model}: {messages[-1]['content'][:50]}...")
            response = self.client.chat(
                model=self.main_model,
                messages=messages,
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
                logger.info(f"📥 Ответ от {self.main_model}: {result[:50]}...")
                return clean_markdown(result)
                
        except Exception as e:
            logger.error(f"❌ Ошибка основной модели: {e}")
            return f"❌ Ошибка при обращении к основной модели: {str(e)}"

    async def analyze_image(self, image_path: str, user_text: str = "") -> str:
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            vision_messages = [{
                'role': 'user',
                'content': 'Опиши это изображение максимально подробно и детально. '
                          'Опиши все объекты, действия, тексты, эмоции, композицию, '
                          'цвета, стиль и любые важные детали.',
                'images': [base64_image]
            }]
            
            logger.info(f"📤 Анализ изображения через {self.vision_model}")
            vision_response = self.client.chat(
                model=self.vision_model,
                messages=vision_messages,
                stream=False
            )
            
            description = vision_response['message']['content']
            logger.info(f"📥 Описание изображения: {description[:100]}...")
            
            description = clean_markdown(description)
            
            context = f"Пользователь отправил изображение и написал: \"{user_text}\"\n\n" \
                     f"Описание изображения: {description}\n\n" \
                     f"Пожалуйста, ответь на вопрос пользователя с учетом изображения."
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа изображения: {e}")
            return f"❌ Ошибка при анализе изображения: {str(e)}"

    async def search_internet(self, query: str) -> str:
        try:
            search_messages = [{
                'role': 'user',
                'content': f'Используя доступ к интернету, найди актуальную информацию по: "{query}". '
                          'Предоставь краткий и структурированный ответ с ключевыми фактами.'
            }]
            
            logger.info(f"🔍 Поиск: {query}")
            search_response = self.client.chat(
                model=self.vision_model,
                messages=search_messages,
                stream=False,
                tools=[{'type': 'search'}]
            )
            
            search_result = search_response['message']['content']
            
            final_messages = [{
                'role': 'user',
                'content': f'Пользователь запросил: "{query}"\n\n'
                          f'Результаты поиска: {search_result}\n\n'
                          f'Пожалуйста, предоставь понятный и структурированный ответ на русском языке.'
            }]
            
            return await self.chat_with_main(final_messages)
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return f"❌ Ошибка при поиске в интернете: {str(e)}"