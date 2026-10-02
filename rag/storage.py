"""
Общее хранилище (SQLite) для всех источников — сайтов и ВК-пабликов.

Схема одна на весь проект: таблица documents(title, url, text, date, ...).
Чанкинг (chunk.py) читает из неё же, не зная, откуда взялись данные.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    url        TEXT    NOT NULL UNIQUE,
    text       TEXT    NOT NULL,
    date       TEXT,
    text_hash  TEXT    NOT NULL,
    fetched_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents(text_hash);
CREATE INDEX IF NOT EXISTS idx_documents_date   ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    conn.commit()
    return conn


def clear_db(conn: sqlite3.Connection) -> int:
    """Удаляет всё содержимое documents (данные с прошлых запусков). Возвращает сколько удалено."""
    n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.execute("DELETE FROM documents")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'").fetchone():
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'documents'")
    conn.commit()
    return n


def save_document(conn: sqlite3.Connection, source: str, doc: dict) -> str:
    """doc: {'title','url','text','date'}. Возвращает 'new' | 'dup-url' | 'dup-text'."""
    text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM documents WHERE url = ?", (doc["url"],))
    if cur.fetchone():
        return "dup-url"
    cur.execute("SELECT 1 FROM documents WHERE text_hash = ?", (text_hash,))
    if cur.fetchone():
        return "dup-text"
    cur.execute(
        "INSERT INTO documents (source, title, url, text, date, text_hash, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, doc["title"], doc["url"], doc["text"], doc["date"], text_hash,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return "new"
