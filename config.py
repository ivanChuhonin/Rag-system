# -*- coding: utf-8 -*-
"""
Конфигурация LLM-части проекта (ask.py). Секреты и настраиваемые параметры
берутся из .env (см. .env.example) — токенов и ключей в коде нет.
"""
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class OllamaConfig:
    # Локальный Ollama Desktop: base_url = http://localhost:11434, api_key не нужен.
    # Облачные модели ollama.com: base_url = https://ollama.com, api_key обязателен.
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    api_key: str | None = os.getenv("OLLAMA_API_KEY") or None
    model: str = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "400"))
    top_k: int = int(os.getenv("OLLAMA_TOP_K", "20"))
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))


@dataclass
class GigaChatConfig:
    # Authorization key из личного кабинета GigaChat API (base64-строка).
    credentials: str = os.getenv("GIGACHAT_CREDENTIALS", "")
    scope: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    model: str = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
    temperature: float = float(os.getenv("GIGACHAT_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("GIGACHAT_MAX_TOKENS", "400"))
    request_timeout: float = float(os.getenv("GIGACHAT_TIMEOUT", "60"))
    # У GigaChat сертификат от российского Минцифры, обычно не в системном
    # доверенном хранилище — без этого запросы падают с SSL-ошибкой.
    verify_ssl_certs: bool = _get_bool("GIGACHAT_VERIFY_SSL", False)


@dataclass
class VKConfig:
    access_token: str = os.getenv("VK_ACCESS_TOKEN", "")


ollama_cfg = OllamaConfig()
gigachat_cfg = GigaChatConfig()
vk_cfg = VKConfig()
