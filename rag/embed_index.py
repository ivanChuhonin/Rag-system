"""
Векторный индекс (pgvector поверх chunks — раньше тут был FAISS-файл).

Считает эмбеддинги текста чанков локальной моделью через fastembed (ONNX,
без torch, без Ollama/GigaChat — работает офлайн) и пишет их прямо в колонку
chunks.embedding (vector(384), добавляет её сюда же миграцией при первом
запуске) — отдельного файла-индекса больше нет, вектор живёт в той же строке,
что текст чанка. Поиск по ним — HNSW-индекс (vector_cosine_ops), см. search.py.

Модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384 измерения, ~225 МБ, многоязычная — годится для русского). Веса
скачиваются один раз и кешируются в .fastembed_cache/ рядом с этим файлом.

Какие chunk_id уже заэмбеддены — видно по chunks.embedding IS NOT NULL,
это даёт инкрементальность через --keep-index без отдельной таблицы учёта.

Запуск (из корня репозитория):
    python -m rag.embed_index                     # (пере)считать эмбеддинги по всем chunks
    python -m rag.embed_index --keep-index         # доэмбеддить только новые chunks
    python -m rag.search "запрос" --engine faiss   # искать через эти векторы

Зависимости: fastembed, numpy (тянется вместе с fastembed), psycopg+pgvector
(через storage.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import psycopg
from fastembed import TextEmbedding

from rag.storage import get_connection

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache")

BATCH_SIZE = 64
PROGRESS_EVERY = 2_000  # эмбеддинг медленнее, чем чанкинг/save — печатаем чаще

# HNSW — приближённый ближайший поиск, на 47k векторов практической разницы
# с полным перебором нет, но индекс готов расти вместе с корпусом без
# переделки запросов. vector_cosine_ops — векторы L2-нормированы ниже, так
# что cosine distance здесь корректная мера.
DDL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_embedder: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)
    return _embedder


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # пустой/нулевой вектор — не делить на 0
    return (vecs / norms).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Векторы для текста чанков (то, что кладём в индекс)."""
    vecs = np.array(list(get_embedder().embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32)
    return _normalize(vecs)


def embed_query(text: str) -> np.ndarray:
    """Вектор для поискового запроса — та же модель, один текст."""
    return embed_passages([text])[0]


# --------------------------------------------------------------------------- #
# БД
# --------------------------------------------------------------------------- #

def open_db() -> psycopg.Connection:
    conn = get_connection()
    has_chunks = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks'"
    ).fetchone()
    if not has_chunks:
        sys.exit("В базе нет таблицы chunks — сначала запусти chunk.py.")
    conn.execute(DDL)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# Построение / обновление эмбеддингов
# --------------------------------------------------------------------------- #

def build(keep_index: bool, batch_size: int, verbose: bool) -> None:
    conn = open_db()

    if not keep_index:
        with conn.cursor() as cur:
            cur.execute("UPDATE chunks SET embedding = NULL")
        conn.commit()
        print("сбросил старые эмбеддинги — считаю заново.\n")

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total_chunks == 0:
        print("В chunks 0 строк — сначала запусти chunk.py. Индексировать нечего.")
        conn.close()
        return

    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    already = total_chunks - len(rows)
    print(f"всего чанков: {total_chunks}, уже с эмбеддингом: {already}, к расчёту: {len(rows)}")
    if not rows:
        conn.close()
        return

    print(f"гружу модель {EMBED_MODEL}... (первый раз — скачает ~225 МБ, дальше из кеша)")
    get_embedder()  # форсируем загрузку здесь, а не молча внутри первого батча

    added = 0
    t_start = time.time()
    next_milestone = PROGRESS_EVERY

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]

        vecs = embed_passages(texts)
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s WHERE id = %s",
                [(vec, cid) for cid, vec in zip(ids, vecs)],
            )
        conn.commit()
        added += len(batch)

        if verbose:
            print(f"  + {len(batch)} чанков embedded (batch {i // batch_size + 1})")
        if added >= next_milestone:
            rate = added / (time.time() - t_start)
            print(f"\n>>> заэмбеддено чанков: {added}/{len(rows)} (~{rate:.0f}/сек)\n")
            next_milestone += PROGRESS_EVERY

    dt = time.time() - t_start
    n_total = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    print("\n" + "-" * 60)
    print(f"добавлено векторов : {added}")
    print(f"всего в базе       : {n_total}")
    print(f"время              : {dt:.1f}с (~{added / dt if dt else 0:.0f}/сек)")
    conn.close()


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Считает эмбеддинги chunks -> chunks.embedding (pgvector)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="размер батча эмбеддинга")
    ap.add_argument("--keep-index", action="store_true",
                    help="не пересчитывать эмбеддинги с нуля, досчитать только новые чанки")
    ap.add_argument("--verbose", action="store_true", help="лог по каждому батчу")
    args = ap.parse_args(argv)

    build(args.keep_index, args.batch_size, args.verbose)


if __name__ == "__main__":
    main()
