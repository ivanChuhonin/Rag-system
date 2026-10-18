"""
FastAPI-обёртка над search.py/ask.py (MVP, шаг 5 — HTTP-эндпоинты).

GET /search?q=...    -> топ-чанки со score и ссылками на источники.
GET /ask?q=...        -> ответ (LLM или экстрактивно) + источники.
GET /                 -> простой чат-UI (static/index.html), источники справа.

BM25-индекс строится один раз при старте сервера (lifespan), не на каждый
запрос — иначе на 47k+ чанков это было бы секунды на каждый HTTP-запрос
впустую. Векторный поиск (pgvector) живёт в самой БД — его не нужно грузить
в память отдельно, каждый запрос просто шлёт SQL. Postgres-соединение —
новое на каждый HTTP-запрос: открытие дёшево, а гонять одно соединение между
потоками uvicorn (FastAPI крутит sync-эндпоинты в threadpool) — та же логика,
что была с sqlite3, только теперь без ограничения "один писатель за раз".

Запуск (из корня репозитория):
    uvicorn rag.api:app --reload
    curl "http://127.0.0.1:8000/search?q=фильмы+про+космос&top_k=5"
    curl "http://127.0.0.1:8000/ask?q=какой+фильм+с+юрой+борисовым&llm=ollama"

Зависимости: fastapi, uvicorn + всё, что тянут search.py/ask.py/embed_index.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag.ask as ask_mod
import rag.search as se
from rag.storage import get_connection

DEFAULT_BM25_WEIGHT = 0.7


# --------------------------------------------------------------------------- #
# Схемы ответов
# --------------------------------------------------------------------------- #

class SearchResult(BaseModel):
    chunk_id: int
    score: float
    chunk_index: int
    title: str
    url: str
    date: str | None
    source: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    engine: str
    results: list[SearchResult]


class AskResponse(BaseModel):
    query: str
    llm: str
    answer: str
    sources: list[SearchResult]
    warning: str | None = None


# --------------------------------------------------------------------------- #
# Индексы — строим один раз при старте
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = get_connection()
    has_chunks = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks'"
    ).fetchone()
    if not has_chunks:
        conn.close()
        raise RuntimeError("В базе нет таблицы chunks — запусти vk_parse.py, потом chunk.py.")

    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_vectors = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    print(f"строю BM25-индекс: {n_chunks} чанков...")
    app.state.bm25 = se.build_index(conn, None)  # без source-фильтра — глобальный индекс
    app.state.n_vectors = n_vectors
    print(f"готово: BM25 {app.state.bm25.n_docs} чанков, pgvector {n_vectors} векторов")

    conn.close()
    yield


app = FastAPI(title="Кино-RAG API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Ретривер — переиспользует уже построенный BM25-индекс, векторный поиск
# фильтруется по source прямо в SQL (см. se.vector_search).
# --------------------------------------------------------------------------- #

def retrieve(query: str, top_k: int, engine: str, source: str | None, bm25_weight: float) -> list[dict]:
    conn = get_connection()
    try:
        bm25 = app.state.bm25

        if engine == "bm25":
            hits = bm25.search(query, top_k)
            return se.fetch_results(conn, hits, source=source, limit=top_k)

        if engine == "vector":
            hits = se.vector_search(conn, query, top_k, source=source)
            return se.fetch_results(conn, hits, source=source, limit=top_k)

        # hybrid
        w_bm25 = bm25_weight
        w_vector = 1.0 - w_bm25
        candidate_k = max(top_k * 6, 30)
        bm25_hits = bm25.search(query, candidate_k)
        vector_hits = se.vector_search(conn, query, candidate_k, source=source)
        fused = se.rrf_fuse([bm25_hits, vector_hits], weights=[w_bm25, w_vector])
        return se.fetch_results(conn, fused, source=source, limit=top_k)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Эндпоинты
# --------------------------------------------------------------------------- #

@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., min_length=1, description="поисковый запрос"),
    top_k: int = Query(5, ge=1, le=50),
    engine: str = Query("hybrid", pattern="^(bm25|vector|hybrid)$"),
    source: str | None = Query(None, description="фильтр по source, например vk:kinoprotebya"),
    bm25_weight: float = Query(DEFAULT_BM25_WEIGHT, ge=0.0, le=1.0, description="только для hybrid"),
):
    results = retrieve(q, top_k, engine, source, bm25_weight)
    return SearchResponse(query=q, engine=engine, results=results)


@app.get("/ask", response_model=AskResponse)
def ask_endpoint(
    q: str = Query(..., min_length=1, description="вопрос"),
    top_k: int = Query(8, ge=1, le=20, description="сколько чанков подать в контекст"),
    llm: str = Query("none", pattern="^(none|ollama|gigachat)$"),
    model: str | None = Query(None, description="только для llm=ollama"),
    engine: str = Query("hybrid", pattern="^(bm25|vector|hybrid)$"),
    source: str | None = Query(None),
    bm25_weight: float = Query(DEFAULT_BM25_WEIGHT, ge=0.0, le=1.0),
):
    results = retrieve(q, top_k, engine, source, bm25_weight)
    if not results:
        return AskResponse(query=q, llm=llm, answer="По этому вопросу ничего не нашлось.", sources=[])

    if llm == "none":
        return AskResponse(query=q, llm=llm, answer=ask_mod.extractive_answer(results), sources=results)

    context = ask_mod.build_context(results)
    warning = None
    try:
        if llm == "ollama":
            from rag.config import ollama_cfg
            text = ask_mod.ask_ollama(q, context, model or ollama_cfg.model)
        else:
            from rag.config import gigachat_cfg
            if not gigachat_cfg.credentials:
                raise ask_mod.LLMError("не задан GIGACHAT_CREDENTIALS в .env")
            text = ask_mod.ask_gigachat(q, context)
    except ask_mod.LLMError as e:
        warning = f"LLM недоступен ({e}) — показан экстрактивный ответ вместо него."
        text = ask_mod.extractive_answer(results)

    return AskResponse(query=q, llm=llm, answer=text, sources=results, warning=warning)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bm25_chunks": app.state.bm25.n_docs,
        "vectors": app.state.n_vectors,
    }


# Статика монтируется последней: /search, /ask, /health уже зарегистрированы
# выше и матчатся первыми, а html=True отдаёт static/index.html на "/".
app.mount("/", StaticFiles(directory="static", html=True), name="static")
