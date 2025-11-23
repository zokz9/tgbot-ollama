# config.py
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")

# Разные модели для разных задач
OLLAMA_MAIN_MODEL = "kimi-k2-thinking:cloud"           # Обычное общение
OLLAMA_SEARCH_MODEL = "gpt-oss:120b-cloud"       # Поиск в интернете
OLLAMA_VISION_MODEL = "qwen3-vl:235b-cloud"  # Анализ изображений

MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Настройки контекста
MAX_HISTORY_SIZE = 10
MAX_SEARCH_CONTEXT = 5