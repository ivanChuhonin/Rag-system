"""
Общее хранилище (Postgres + pgvector) для всех источников — сайтов и
ВК-пабликов. Раньше это был файл SQLite; переехали на Postgres, чтобы
документы/чанки/векторы жили в одном месте с нормальным сервером, а не
в одном хрупком файле (см. docker-compose.yml — локальный контейнер).

Схема одна на весь проект: таблица documents(title, url, text, date, ...).
Чанкинг (chunk.py) создаёт свою таблицу chunks со ссылкой на documents.id,
не зная, откуда взялись сами документы.

Подключение настраивается через .env (POSTGRES_HOST/PORT/USER/PASSWORD/DB,
см. .env.example и rag/config.py) — секретов в коде нет.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import psycopg
from pgvector.psycopg import register_vector

from rag.config import pg_cfg

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    url        TEXT    NOT NULL UNIQUE,
    text       TEXT    NOT NULL,
    date       TEXT,
    text_hash  TEXT    NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents(text_hash);
CREATE INDEX IF NOT EXISTS idx_documents_date   ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
"""


def get_connection() -> psycopg.Connection:
    """Новое соединение с Postgres + создание схемы documents, если её ещё
    нет (CREATE TABLE IF NOT EXISTS — безопасно звать на каждый запуск).
    register_vector делает psycopg понятным тип vector(...) из pgvector —
    нужно на каждом соединении, где читаются/пишутся эмбеддинги (chunks)."""
    conn = psycopg.connect(pg_cfg.conninfo(), autocommit=False)
    register_vector(conn)
    conn.execute(DDL)
    conn.commit()
    return conn


def clear_db(conn: psycopg.Connection) -> int:
    """Удаляет всё содержимое documents (данные с прошлых запусков).
    CASCADE тащит за собой chunks (там FK ON DELETE CASCADE) — отдельно
    чистить chunks/эмбеддинги не нужно, Postgres сам разберётся по FK."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        n = cur.fetchone()[0]
        cur.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    conn.commit()
    return n


def save_document(conn: psycopg.Connection, source: str, doc: dict) -> str:
    """doc: {'title','url','text','date'}. Возвращает 'new' | 'dup-url' | 'dup-text'."""
    text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE url = %s", (doc["url"],))
        if cur.fetchone():
            return "dup-url"
        cur.execute("SELECT 1 FROM documents WHERE text_hash = %s", (text_hash,))
        if cur.fetchone():
            return "dup-text"
        cur.execute(
            "INSERT INTO documents (source, title, url, text, date, text_hash, fetched_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (source, doc["title"], doc["url"], doc["text"], doc["date"], text_hash,
             datetime.now(timezone.utc)),
        )
    conn.commit()
    return "new"
