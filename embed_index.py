"""
Векторный индекс (MVP, шаг 3b — FAISS поверх chunks).

Считает эмбеддинги текста чанков локальной моделью через fastembed (ONNX,
без torch, без Ollama/GigaChat — работает офлайн) и складывает в FAISS
(IndexFlatIP поверх L2-нормированных векторов = косинусная близость).

Модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384 измерения, ~225 МБ, многоязычная — годится для русского). Веса
скачиваются один раз и кешируются в .fastembed_cache/ рядом с этим файлом.

Соответствие chunk_id <-> вектор хранит сам FAISS (IndexIDMap: id чанка
и есть id вектора) — отдельный файл-мэппинг не нужен. Какие chunk_id уже
попали в индекс — отслеживается в таблице vector_index_meta (в той же
news.db), это даёт инкрементальность через --keep-index, как у chunk.py.

Запуск:
    python embed_index.py                     # (пере)строить индекс по всем chunks
    python embed_index.py --keep-index         # доиндексировать только новые chunks
    python search.py "запрос" --engine faiss   # искать через этот индекс

Зависимости: fastembed, faiss-cpu, numpy (тянется вместе с ними).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

import faiss
import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache")

BATCH_SIZE = 64
PROGRESS_EVERY = 2_000  # эмбеддинг медленнее, чем чанкинг/save — печатаем чаще

DDL = """
CREATE TABLE IF NOT EXISTS vector_index_meta (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE
);
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
# БД / индекс на диске
# --------------------------------------------------------------------------- #

def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    has_chunks = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    if not has_chunks:
        sys.exit(f"В базе {path} нет таблицы chunks — сначала запусти chunk.py.")
    conn.executescript(DDL)
    conn.commit()
    return conn


def default_index_path(db_path: str) -> str:
    base, _ = os.path.splitext(db_path)
    return base + ".faiss"


def load_index(index_path: str) -> faiss.Index:
    """Загружает существующий индекс. Кидает понятную ошибку, если его нет
    или размерность не совпадает с текущей моделью (значит модель менялась)."""
    if not os.path.exists(index_path):
        sys.exit(f"Индекс {index_path} не найден — сначала запусти embed_index.py.")
    index = faiss.read_index(index_path)
    if index.d != EMBED_DIM:
        sys.exit(
            f"У индекса {index_path} размерность {index.d}, а у текущей модели {EMBED_DIM}. "
            "Модель поменялась — пересобери индекс без --keep-index."
        )
    return index


def _fresh_index() -> faiss.Index:
    return faiss.IndexIDMap(faiss.IndexFlatIP(EMBED_DIM))


# --------------------------------------------------------------------------- #
# Построение / обновление индекса
# --------------------------------------------------------------------------- #

def build(db_path: str, index_path: str, keep_index: bool, batch_size: int, verbose: bool) -> None:
    conn = open_db(db_path)

    if keep_index and os.path.exists(index_path):
        index = load_index(index_path)
        already = {row[0] for row in conn.execute("SELECT chunk_id FROM vector_index_meta")}
        print(f"--keep-index: в индексе уже {index.ntotal} векторов, пропущу {len(already)} чанков.\n")
    else:
        index = _fresh_index()
        already = set()
        conn.execute("DELETE FROM vector_index_meta")
        conn.commit()
        if os.path.exists(index_path):
            os.remove(index_path)
        print("строю индекс заново.\n")

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total_chunks == 0:
        print("В chunks 0 строк — сначала запусти chunk.py. Индексировать нечего.")
        conn.close()
        return

    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    todo = [(cid, text) for cid, text in rows if cid not in already]
    print(f"всего чанков: {total_chunks}, к индексации: {len(todo)}")
    if not todo:
        conn.close()
        return

    print(f"гружу модель {EMBED_MODEL}... (первый раз — скачает ~225 МБ, дальше из кеша)")
    get_embedder()  # форсируем загрузку здесь, а не молча внутри первого батча

    added = 0
    t_start = time.time()
    next_milestone = PROGRESS_EVERY

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        ids = np.array([cid for cid, _ in batch], dtype=np.int64)
        texts = [text for _, text in batch]

        vecs = embed_passages(texts)
        index.add_with_ids(vecs, ids)
        conn.executemany(
            "INSERT OR IGNORE INTO vector_index_meta (chunk_id) VALUES (?)",
            [(int(cid),) for cid in ids],
        )
        conn.commit()
        added += len(batch)

        if verbose:
            print(f"  + {len(batch)} чанков embedded (batch {i // batch_size + 1})")
        if added >= next_milestone:
            rate = added / (time.time() - t_start)
            print(f"\n>>> заэмбеддено чанков: {added}/{len(todo)} (~{rate:.0f}/сек)\n")
            next_milestone += PROGRESS_EVERY

    faiss.write_index(index, index_path)
    dt = time.time() - t_start
    print("\n" + "-" * 60)
    print(f"добавлено векторов : {added}")
    print(f"всего в индексе    : {index.ntotal}")
    print(f"время              : {dt:.1f}с (~{added / dt if dt else 0:.0f}/сек)")
    print(f"индекс             : {os.path.abspath(index_path)}")
    conn.close()


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Строит/обновляет FAISS-индекс по chunks")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite")
    ap.add_argument("--index-path", default=None,
                    help="путь к файлу индекса (по умолчанию <db>.faiss рядом с базой)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="размер батча эмбеддинга")
    ap.add_argument("--keep-index", action="store_true",
                    help="не пересобирать индекс с нуля, доиндексировать только новые чанки")
    ap.add_argument("--verbose", action="store_true", help="лог по каждому батчу")
    args = ap.parse_args(argv)

    index_path = args.index_path or default_index_path(args.db)
    build(args.db, index_path, args.keep_index, args.batch_size, args.verbose)


if __name__ == "__main__":
    main()
