"""
Поиск (MVP, шаг 3 — "Индекс" + "/search"). Три движка на выбор:

  --engine bm25    (по умолчанию) — точное совпадение слов, индекс в памяти,
                   без внешних зависимостей (rank-bm25/whoosh сознательно не
                   ставим — вариант "быстро, без внешних ключей" из плана).
  --engine faiss   — семантический поиск по векторам (embed_index.py должен
                   быть запущен заранее). Ловит смысловые совпадения без
                   общих слов, но на практике слабее BM25 на именных
                   запросах (имя актёра/режиссёра) — маленькая многоязычная
                   модель плохо различает конкретные имена собственные.
  --engine hybrid  — BM25 + FAISS вместе через Reciprocal Rank Fusion:
                   берём топ-кандидатов от обоих, элемент получает
                   1/(60+ранг) очков от каждого списка, где встретился,
                   очки суммируются. Не требует нормализации score (BM25 и
                   косинус на разных шкалах — сравнивать напрямую нельзя).
                   Рекомендуемый режим по умолчанию для реальных запросов.

Индекс строится в памяти при старте (для тысяч-десятков тысяч чанков это
доли секунды/пара секунд) и живёт, пока работает процесс — при интерактивном
использовании (REPL) строится один раз, а не на каждый запрос.

Ограничение BM25: токенизация — просто нижний регистр + разбиение по словам,
без стемминга/лемматизации. Для русского это значит, что "фильм" и "фильма"
это разные термы — совпадения только по точной форме слова. Если понадобится
точнее — сюда позже добавляется pymorphy3 (или снежный стеммер) внутри
tokenize(), сам BM25 менять не придётся.

Запуск:
    python search.py "лучшие фильмы про космос" --engine hybrid
    python search.py "лучшие фильмы про космос" --top-k 10 --source vk:kinoprotebya
    python search.py --engine hybrid              # REPL: вводишь запросы построчно

Зависимостей кроме stdlib нет для --engine bm25; faiss/hybrid тянут
embed_index.py (fastembed, faiss-cpu, numpy).
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict

from stopwords_ru import RU_STOPWORDS

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75
SNIPPET_LEN = 220


def tokenize(text: str) -> list[str]:
    """Только для BM25 — не переиспользовать для эмбеддингов (там нужен
    связный текст, а не мешок значимых слов, см. text_clean.py)."""
    return [
        t for t in TOKEN_RE.findall(text.lower())
        if t not in RU_STOPWORDS and (not t.isdigit() or len(t) <= 4)
    ]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class BM25Index:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []       # doc_idx -> chunk_id
        self.doc_len: list[int] = []       # doc_idx -> число токенов
        self.postings: dict[str, dict[int, int]] = {}  # term -> {doc_idx: freq}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.n_docs = 0

    def build(self, chunk_rows: list[tuple[int, str]]) -> None:
        """chunk_rows: [(chunk_id, text), ...]"""
        self.doc_ids = [cid for cid, _ in chunk_rows]
        self.n_docs = len(chunk_rows)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        doc_len: list[int] = []

        for idx, (_, text) in enumerate(chunk_rows):
            tokens = tokenize(text)
            doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                postings[term][idx] = freq

        self.doc_len = doc_len
        self.avgdl = (sum(doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings = dict(postings)
        self.idf = {
            term: math.log((self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5) + 1)
            for term, plist in self.postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Возвращает [(chunk_id, score), ...] по убыванию score, лучшие top_k."""
        if not self.n_docs:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for idx, freq in plist.items():
                dl = self.doc_len[idx] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * freq * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in ranked]


# --------------------------------------------------------------------------- #
# БД
# --------------------------------------------------------------------------- #

def load_chunks(conn: sqlite3.Connection, source: str | None = None) -> list[tuple[int, str]]:
    if source:
        rows = conn.execute(
            "SELECT c.id, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.source = ?", (source,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    return rows


def fetch_results(conn: sqlite3.Connection, hits: list[tuple[int, float]],
                  source: str | None = None, limit: int | None = None) -> list[dict]:
    out = []
    for chunk_id, score in hits:
        row = conn.execute(
            "SELECT c.text, c.chunk_index, d.title, d.url, d.date, d.source "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            continue
        text, chunk_index, title, url, date, doc_source = row
        if source and doc_source != source:
            continue
        snippet = text[:SNIPPET_LEN] + ("…" if len(text) > SNIPPET_LEN else "")
        out.append({
            "chunk_id": chunk_id, "score": score, "chunk_index": chunk_index,
            "title": title, "url": url, "date": date, "source": doc_source,
            "snippet": snippet,  # для UI/консоли — коротко, читабельно
            "text": text,        # полный текст чанка — для LLM-контекста (ask.py), не обрезать
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    has_chunks = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    if not has_chunks:
        sys.exit(f"В базе {path} нет таблицы chunks — сначала запусти chunk.py.")
    return conn


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_index(conn: sqlite3.Connection, source: str | None) -> BM25Index:
    rows = load_chunks(conn, source)
    idx = BM25Index()
    idx.build(rows)
    return idx


def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], weights: list[float] | None = None,
            k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion — комбинирует несколько ранжированных списков
    без нормализации сырых score (BM25 и косинус из FAISS на разных
    шкалах, сравнивать их напрямую нельзя). Каждый список даёт элементу
    weight/(k+ранг) очков, суммируем по всем спискам, где элемент встретился.
    weights по умолчанию — поровну; передай, например, [0.7, 0.3], чтобы
    доверять первому списку сильнее (у нас BM25 надёжнее на именных
    запросах — см. rag-news-project)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[int, float] = defaultdict(float)
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (item_id, _) in enumerate(ranked, start=1):
            scores[item_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def faiss_search(index, query: str, top_k: int, oversample_for_filter: bool) -> list[tuple[int, float]]:
    """oversample_for_filter=True — тянуть больше кандидатов, если потом
    будем отсекать по source (у FAISS один общий индекс на все источники,
    без --source не отфильтровать заранее)."""
    from embed_index import embed_query  # тяжёлый импорт — только когда реально нужен

    k = top_k
    if oversample_for_filter:
        k = min(max(top_k * 8, 50), index.ntotal)
    qvec = embed_query(query).reshape(1, -1)
    scores, ids = index.search(qvec, k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def print_results(results: list[dict]) -> None:
    if not results:
        print("(ничего не найдено)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['score']:.3f}] {r['title']}")
        print(f"   {r['url']}  |  {r['date'] or 'no-date'}  |  {r['source']}  |  чанк #{r['chunk_index']}")
        print(f"   {r['snippet']}")


# --------------------------------------------------------------------------- #
# Переиспользуемый ретривер — им пользуется и этот CLI, и ask.py
# --------------------------------------------------------------------------- #

def make_retriever(conn: sqlite3.Connection, engine: str, db_path: str, index_path: str | None,
                   source: str | None, bm25_weight: float, verbose: bool = True):
    """Возвращает run_query(query, top_k) -> list[dict] под выбранный engine.
    verbose=True печатает, что именно построено/загружено (для CLI-режима;
    ask.py вызывает с verbose=False, чтобы не засорять вывод ответа)."""
    index = None
    bm25 = None

    if engine in ("faiss", "hybrid"):
        import embed_index as ei
        index = ei.load_index(index_path or ei.default_index_path(db_path))
    if engine in ("bm25", "hybrid"):
        bm25 = build_index(conn, source)

    if engine == "faiss":
        if verbose:
            print(f"загрузил FAISS-индекс: {index.ntotal} векторов"
                 f"{' (фильтр source=' + source + ')' if source else ''}...")

        def run_query(q: str, top_k: int) -> list[dict]:
            hits = faiss_search(index, q, top_k, oversample_for_filter=bool(source))
            return fetch_results(conn, hits, source=source, limit=top_k)

    elif engine == "hybrid":
        w_bm25 = bm25_weight
        w_faiss = 1.0 - w_bm25
        if verbose:
            print(f"гибрид готов: BM25(вес={w_bm25:.2f}) + FAISS(вес={w_faiss:.2f}, {index.ntotal} векторов)"
                 f"{' (source=' + source + ')' if source else ''}...")

        def run_query(q: str, top_k: int) -> list[dict]:
            candidate_k = max(top_k * 6, 30)
            bm25_hits = bm25.search(q, candidate_k)
            faiss_hits = faiss_search(index, q, candidate_k, oversample_for_filter=bool(source))
            fused = rrf_fuse([bm25_hits, faiss_hits], weights=[w_bm25, w_faiss])
            return fetch_results(conn, fused, source=source, limit=top_k)

    else:  # bm25
        def run_query(q: str, top_k: int) -> list[dict]:
            hits = bm25.search(q, top_k)
            return fetch_results(conn, hits)

    return run_query


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Поиск по chunks (BM25 или FAISS)")
    ap.add_argument("query", nargs="?", default=None, help="запрос; без него — интерактивный режим")
    ap.add_argument("--engine", choices=["bm25", "faiss", "hybrid"], default="bm25", help="движок поиска")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite")
    ap.add_argument("--index-path", default=None,
                    help="(только --engine faiss/hybrid) путь к файлу индекса, по умолчанию <db>.faiss")
    ap.add_argument("--top-k", type=int, default=5, help="сколько чанков вернуть")
    ap.add_argument("--source", default=None, help="фильтр по source, например vk:kinoprotebya")
    ap.add_argument("--bm25-weight", type=float, default=0.7,
                    help="(только --engine hybrid) вес BM25 в RRF, 0..1 — остальное достаётся FAISS. "
                         "BM25 надёжнее на именных запросах (актёры/режиссёры), поэтому по умолчанию 0.7")
    args = ap.parse_args(argv)

    conn = open_db(args.db)
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if n_chunks == 0:
        sys.exit("В chunks 0 строк — нечего искать. Запусти vk_parse.py, потом chunk.py.")

    run_query_topk = make_retriever(conn, args.engine, args.db, args.index_path,
                                    args.source, args.bm25_weight, verbose=True)

    def run_query(q: str) -> list[dict]:
        return run_query_topk(q, args.top_k)

    if args.query:
        print_results(run_query(args.query))
        return

    print("готово. Пустая строка или Ctrl+C — выход.\n")
    while True:
        try:
            q = input("запрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print_results(run_query(q))


if __name__ == "__main__":
    main()
