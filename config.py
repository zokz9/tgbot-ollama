# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Ollama API
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MAIN_MODEL = "kimi-k2:1t-cloud"
OLLAMA_VISION_MODEL = "qwen3-vl:235b-instruct-cloud"

# Внутренние настройки
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB