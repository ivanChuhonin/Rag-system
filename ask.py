"""
/ask (MVP, шаг 4 — финальный ответ с источниками).

Берёт топ-чанки под вопрос через ретривер из search.py (по умолчанию
hybrid BM25+FAISS, 70/30) и либо:

  --llm none      (по умолчанию) — экстрактивный ответ: сами топ-фрагменты
                  с источниками, без генерации текста. Работает без каких-
                  либо LLM-зависимостей и credentials.
  --llm ollama    — отправляет контекст+вопрос в Ollama. Настройки — в .env
                  (см. .env.example): локальный Ollama Desktop
                  (OLLAMA_BASE_URL=http://localhost:11434, без api_key) или
                  облачные модели ollama.com (OLLAMA_BASE_URL=https://ollama.com
                  + OLLAMA_API_KEY). Модель по умолчанию — gemma4:cloud
                  (переопределяется OLLAMA_MODEL в .env или флагом --model).
  --llm gigachat  — отправляет в GigaChat API через официальный SDK
                  (gigachat), не сырыми запросами: он сам делает OAuth,
                  обновление токена и по умолчанию не проверяет SSL-сертификат
                  (у GigaChat он не из обычного доверенного хранилища —
                  без этого голые requests падают с SSL-ошибкой, что и
                  было причиной проблемы). Нужен GIGACHAT_CREDENTIALS в .env
                  (Authorization key из личного кабинета GigaChat API).

Запуск:
    python ask.py "какой фильм с юрой борисовым ожидается"
    python ask.py "вопрос" --llm ollama
    python ask.py "вопрос" --llm ollama --model qwen3.5:9b
    python ask.py "вопрос" --llm gigachat
    python ask.py --llm ollama                # REPL

Зависимости: requests, python-dotenv, gigachat + всё, что тянет search.py
при --engine faiss/hybrid.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

import requests
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from config import ollama_cfg, gigachat_cfg
from search import make_retriever, open_db

SYSTEM_PROMPT = (
    "Ты — ассистент, который отвечает на вопросы о кино, опираясь ТОЛЬКО на "
    "приведённые ниже фрагменты постов. Не придумывай факты, которых нет в "
    "контексте. Если ответа в контексте нет — прямо скажи, что не нашёл. "
    "В ответе ссылайся на источники их номерами в квадратных скобках, "
    "например [1], [2]."
)


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Контекст из найденных чанков
# --------------------------------------------------------------------------- #

def build_context(results: list[dict]) -> str:
    """Полный текст чанка, не UI-сниппет — обрезанный до 220 символов
    snippet прятал от модели хвост чанка (реальный случай: имя режиссёра
    было именно в этом хвосте — модель "угадала" по своим знаниям и
    подписала правдоподобной, но фактически необоснованной ссылкой)."""
    parts = []
    for i, r in enumerate(results, 1):
        date = (r["date"] or "")[:10]
        parts.append(f"[{i}] ({r['source']}, {date}) {r['text']}")
    return "\n\n".join(parts)


def print_sources(results: list[dict]) -> None:
    print("\nИсточники:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title'][:80]} — {r['url']}")


# --------------------------------------------------------------------------- #
# LLM-бэкенды
# --------------------------------------------------------------------------- #

def ollama_chat(messages: list[dict], model: str | None = None, num_predict: int | None = None) -> str:
    """Низкоуровневый вызов Ollama /api/chat — переиспользуется ask_ollama()
    и внешними скриптами (например eval_recall.py для роли судьи)."""
    payload = {
        "model": model or ollama_cfg.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": ollama_cfg.temperature,
            "num_predict": num_predict or ollama_cfg.num_predict,  # ограничивает длину (и время) генерации
            "top_k": ollama_cfg.top_k,
        },
    }
    headers = {"Authorization": f"Bearer {ollama_cfg.api_key}"} if ollama_cfg.api_key else {}
    try:
        r = requests.post(f"{ollama_cfg.base_url}/api/chat", json=payload, headers=headers,
                          timeout=ollama_cfg.request_timeout)
    except requests.RequestException as e:
        raise LLMError(f"не достучался до Ollama ({ollama_cfg.base_url}): {type(e).__name__} {e}")
    if r.status_code != 200:
        raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError):
        raise LLMError(f"неожиданный формат ответа Ollama: {data}")


def ask_ollama(question: str, context: str, model: str) -> str:
    return ollama_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ],
        model=model,
    )


def ask_gigachat(question: str, context: str) -> str:
    chat = Chat(
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
            Messages(role=MessagesRole.USER, content=f"Контекст:\n{context}\n\nВопрос: {question}"),
        ],
        temperature=gigachat_cfg.temperature,
        max_tokens=gigachat_cfg.max_tokens,
    )
    try:
        with GigaChat(
            credentials=gigachat_cfg.credentials,
            scope=gigachat_cfg.scope,
            model=gigachat_cfg.model,
            verify_ssl_certs=gigachat_cfg.verify_ssl_certs,
            timeout=gigachat_cfg.request_timeout,
        ) as giga:
            response = giga.chat(chat)
    except Exception as e:
        raise LLMError(f"GigaChat: {type(e).__name__} {e}")
    try:
        return response.choices[0].message.content.strip()
    except (AttributeError, IndexError):
        raise LLMError(f"неожиданный формат ответа GigaChat: {response}")


# --------------------------------------------------------------------------- #
# Экстрактивный режим (без LLM)
# --------------------------------------------------------------------------- #

def extractive_answer(results: list[dict]) -> str:
    if not results:
        return "По этому вопросу ничего не нашлось."
    lines = ["Ничего не генерировал (LLM выключен) — вот что нашлось по вопросу, самое релевантное первым:\n"]
    for i, r in enumerate(results, 1):
        date = (r["date"] or "")[:10]
        lines.append(f"[{i}] {r['title']} ({date})\n    {r['snippet']}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Основной цикл
# --------------------------------------------------------------------------- #

def answer(conn: sqlite3.Connection, run_query_topk, question: str, top_k: int,
          llm: str, model: str | None) -> None:
    results = run_query_topk(question, top_k)
    if not results:
        print("По этому вопросу ничего не нашлось.")
        return

    if llm == "none":
        print(extractive_answer(results))
        return

    context = build_context(results)
    try:
        if llm == "ollama":
            text = ask_ollama(question, context, model or ollama_cfg.model)
        else:  # gigachat
            if not gigachat_cfg.credentials:
                raise LLMError(
                    "не задан GIGACHAT_CREDENTIALS. Возьми Authorization key в личном кабинете "
                    "GigaChat API и впиши в .env (см. .env.example)."
                )
            text = ask_gigachat(question, context)
    except LLMError as e:
        print(f"! LLM недоступен ({e}) — показываю экстрактивный ответ вместо него.\n")
        print(extractive_answer(results))
        return

    print(text)
    print_sources(results)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="/ask — вопрос -> топ-чанки -> ответ (LLM или экстрактивно)")
    ap.add_argument("question", nargs="?", default=None, help="вопрос; без него — интерактивный режим")
    ap.add_argument("--llm", choices=["none", "ollama", "gigachat"], default="none", help="источник ответа")
    ap.add_argument("--model", default=None, help="(--llm ollama) модель, иначе OLLAMA_MODEL из .env")
    ap.add_argument("--engine", choices=["bm25", "faiss", "hybrid"], default="hybrid", help="движок ретривера")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite")
    ap.add_argument("--index-path", default=None, help="путь к FAISS-индексу, по умолчанию <db>.faiss")
    ap.add_argument("--top-k", type=int, default=8, help="сколько чанков подать в контекст")
    ap.add_argument("--source", default=None, help="фильтр по source, например vk:kinoprotebya")
    ap.add_argument("--bm25-weight", type=float, default=0.7, help="(--engine hybrid) вес BM25 в RRF")
    args = ap.parse_args(argv)

    conn = open_db(args.db)
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if n_chunks == 0:
        sys.exit("В chunks 0 строк — нечего искать. Запусти vk_parse.py, потом chunk.py.")

    run_query_topk = make_retriever(conn, args.engine, args.db, args.index_path,
                                    args.source, args.bm25_weight, verbose=True)

    if args.question:
        answer(conn, run_query_topk, args.question, args.top_k, args.llm, args.model)
        return

    print("готово. Пустая строка или Ctrl+C — выход.\n")
    while True:
        try:
            q = input("вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        answer(conn, run_query_topk, q, args.top_k, args.llm, args.model)
        print()


if __name__ == "__main__":
    main()
