"""
Чанкинг (MVP, шаг 2 — разбиение текста на фрагменты).

Читает статьи из таблицы `documents` (её заполняет vk_parse.py, через storage.py), режет text на
куски по ~300-800 слов с перехлёстом 50-100 слов, режет по границам
предложений/абзацев (а не как попало по словам), и складывает в таблицу
`chunks` со ссылкой на doc_id — чтобы потом при поиске подтягивать
title/url/date исходной статьи джойном.

Запуск:
    python chunk.py                          # news.db -> chunks по умолчанию
    python chunk.py --chunk-size 400 --overlap 60
    python chunk.py --keep-chunks             # не чистить chunks, пропускать уже нарезанные doc_id

Зависимостей кроме stdlib нет.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Конфиг
# --------------------------------------------------------------------------- #

CHUNK_SIZE_WORDS = 500     # целевой размер чанка, слов (диапазон MVP: 300-800)
CHUNK_OVERLAP_WORDS = 75   # перехлёст между соседними чанками (диапазон MVP: 50-100)
MIN_CHUNK_WORDS = 50       # чанк короче — приклеиваем к предыдущему, а не храним отдельно
PROGRESS_EVERY = 10_000    # печатать промежуточный итог каждые N созданных чанков

# граница предложения (после .!?…) или пустая строка между абзацами
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")
_WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
# Чистая логика чанкинга — не зависит от БД, легко тестируется на любой строке
# --------------------------------------------------------------------------- #

def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _split_long(sentence: str, max_words: int) -> list[str]:
    """Разбивает предложение без пунктуации/аномально длинное на куски по словам."""
    words = _WORD_RE.findall(sentence)
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _tail_words(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Последние `overlap` слов из хвоста sentences, при необходимости режет
    по словам последнее (самое раннее из взятых) предложение — так перехлёст
    никогда не раздувает следующий чанк сверх ожидаемого, даже если одно
    предложение само длиннее overlap."""
    if overlap <= 0:
        return [], 0
    words: list[str] = []
    for s in reversed(sentences):
        words = _WORD_RE.findall(s) + words
        if len(words) >= overlap:
            words = words[-overlap:]
            break
    return ([" ".join(words)], len(words)) if words else ([], 0)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[str]:
    """Режет текст на чанки по ~chunk_size слов с overlap слов перехлёста,
    не разрывая предложения (кроме аномально длинных без пунктуации)."""
    if not text or not text.strip():
        return []

    raw_sentences = _split_sentences(text)
    sentences: list[str] = []
    for s in raw_sentences:
        sentences.extend(_split_long(s, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        w = len(_WORD_RE.findall(sent))
        # min_words тут не участвует: цель — не превысить chunk_size, а не
        # накопить минимум перед split. Раньше несколько коротких предложений
        # (< min_words суммарно), за которыми шёл большой кусок, склеивались
        # в один чанк далеко за chunk_size — min_words решает только, стоит
        # ли ПОСЛЕ формирования чанка держать его отдельно (см. ниже).
        if current and current_words + w > chunk_size:
            chunks.append(" ".join(current))
            # перехлёст: не больше overlap слов из хвоста текущего чанка
            current, current_words = _tail_words(current, overlap)
        current.append(sent)
        current_words += w

    if current:
        chunks.append(" ".join(current))

    # хвостовой чанк короче min_words — приклеить к предыдущему, но только
    # если это не раздует его далеко за chunk_size (иначе пусть остаётся
    # отдельным маленьким чанком — это безопаснее, чем чанк, который потом
    # обрежется при эмбеддинге)
    if len(chunks) >= 2:
        tail_words = len(_WORD_RE.findall(chunks[-1]))
        prev_words = len(_WORD_RE.findall(chunks[-2]))
        if tail_words < min_words and prev_words + tail_words <= chunk_size:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

    return chunks


# --------------------------------------------------------------------------- #
# БД
# --------------------------------------------------------------------------- #

DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    word_count  INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    UNIQUE(doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    has_documents = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
    ).fetchone()
    if not has_documents:
        print(
            f"В базе {path} нет таблицы documents — сначала запусти сборщик данных "
            "(vk_parse.py), чтобы собрать новости. Чанкинг делать не из чего, выхожу.",
            file=sys.stderr,
        )
        sys.exit(1)
    conn.executescript(DDL)
    conn.commit()
    return conn


def clear_chunks(conn: sqlite3.Connection) -> int:
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute("DELETE FROM chunks")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'").fetchone():
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'chunks'")
    # chunks.id — autoincrement, после сброса новые чанки получат старые id заново.
    # Если есть FAISS-индекс (embed_index.py), его bookkeeping по старым id
    # станет враньём — почистим и его, иначе --keep-index потом молча пропустит
    # заново пронумерованные, но на деле новые чанки.
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_index_meta'").fetchone():
        conn.execute("DELETE FROM vector_index_meta")
    conn.commit()

    # тот же резон: файл индекса по умолчанию (<db>.faiss) хранит сами старые
    # векторы, не только bookkeeping — если его не убрать, --keep-index потом
    # домешает новые векторы поверх старых с задублированными id. Кастомный
    # --index-path (нестандартный путь) сюда не попадает — это уже на совести
    # того, кто им пользуется.
    db_file = next(
        (row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main" and row[2]),
        None,
    )
    if db_file:
        default_faiss_path = os.path.splitext(db_file)[0] + ".faiss"
        if os.path.exists(default_faiss_path):
            os.remove(default_faiss_path)
            print(f"также удалил устаревший FAISS-индекс: {default_faiss_path}")

    return n


def already_chunked_doc_ids(conn: sqlite3.Connection) -> set[int]:
    return {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM chunks")}


def save_chunks(conn: sqlite3.Connection, doc_id: int, pieces: list[str]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO chunks (doc_id, chunk_index, text, word_count, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (doc_id, i, piece, len(_WORD_RE.findall(piece)), now)
            for i, piece in enumerate(pieces)
        ],
    )
    return len(pieces)


# --------------------------------------------------------------------------- #
# Основной цикл
# --------------------------------------------------------------------------- #

def run(db_path: str, chunk_size: int, overlap: int, min_words: int,
        limit: int | None, verbose: bool, keep_chunks: bool) -> None:
    conn = open_db(db_path)

    if keep_chunks:
        skip_ids = already_chunked_doc_ids(conn)
        print(f"--keep-chunks: пропущу {len(skip_ids)} уже нарезанных документов.\n")
    else:
        removed = clear_chunks(conn)
        skip_ids = set()
        print(f"очистил chunks: удалено {removed} старых фрагментов.\n")

    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if total_docs == 0:
        print("В documents пока 0 строк — сначала запусти сборщик данных (vk_parse.py). Нарезать нечего.")
        conn.close()
        return

    query = "SELECT id, title, text FROM documents ORDER BY id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    stats = {"docs_seen": 0, "docs_skipped": 0, "docs_chunked": 0, "chunks_created": 0, "docs_empty": 0}
    next_milestone = PROGRESS_EVERY
    since_commit = 0

    for doc_id, title, text in conn.execute(query):
        stats["docs_seen"] += 1
        if doc_id in skip_ids:
            stats["docs_skipped"] += 1
            continue

        pieces = chunk_text(text, chunk_size, overlap, min_words)
        if not pieces:
            stats["docs_empty"] += 1
            if verbose:
                print(f"  - пусто после чанкинга: doc_id={doc_id} {title[:70]!r}", file=sys.stderr)
            continue

        n = save_chunks(conn, doc_id, pieces)
        stats["docs_chunked"] += 1
        stats["chunks_created"] += n
        since_commit += 1
        if verbose:
            print(f"  + doc_id={doc_id} -> {n} чанков | {title[:70]}")

        if since_commit >= 200:
            conn.commit()
            since_commit = 0

        if stats["chunks_created"] >= next_milestone:
            print(f"\n>>> создано чанков: {stats['chunks_created']} (обработано документов: {stats['docs_seen']}/{total_docs})\n")
            next_milestone += PROGRESS_EVERY

    conn.commit()
    _report(stats, db_path)
    conn.close()


def _report(stats: dict, db_path: str) -> None:
    avg = stats["chunks_created"] / stats["docs_chunked"] if stats["docs_chunked"] else 0
    print("\n" + "-" * 60)
    print(f"документов просмотрено : {stats['docs_seen']}")
    print(f"  нарезано             : {stats['docs_chunked']}")
    print(f"  пропущено (keep)     : {stats['docs_skipped']}")
    print(f"  пустых после чанкинга: {stats['docs_empty']}")
    print(f"чанков создано         : {stats['chunks_created']} (~{avg:.1f} на документ)")
    print(f"база                   : {db_path}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Чанкинг documents -> chunks")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite (тот же, что у vk_parse.py)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_WORDS, help="целевой размер чанка, слов")
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_WORDS, help="перехлёст между чанками, слов")
    ap.add_argument("--min-words", type=int, default=MIN_CHUNK_WORDS, help="минимальный размер чанка, слов")
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые K документов")
    ap.add_argument("--verbose", action="store_true", help="подробный лог по каждому документу")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="не очищать chunks перед запуском, пропускать уже нарезанные документы")
    args = ap.parse_args(argv)
    run(args.db, args.chunk_size, args.overlap, args.min_words, args.limit, args.verbose, args.keep_chunks)


if __name__ == "__main__":
    main()
