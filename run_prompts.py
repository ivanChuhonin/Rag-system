"""
Раннер для test_prompts.txt — прогоняет все промпты через ask.answer(),
строя ретривер (BM25+FAISS) и подключение к LLM только один раз, а не на
каждый вопрос заново. Результаты складывает в один текстовый файл.

Запуск:
    python run_prompts.py --llm ollama
    python run_prompts.py --llm none --top-k 8
    python run_prompts.py --prompts test_prompts.txt --out results.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout

import ask
from search import make_retriever, open_db


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Массовый прогон промптов через ask.answer()")
    ap.add_argument("--prompts", default="test_prompts.txt")
    ap.add_argument("--out", default="test_results.txt")
    ap.add_argument("--llm", choices=["none", "ollama", "gigachat"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--engine", choices=["bm25", "faiss", "hybrid"], default="hybrid")
    ap.add_argument("--db", default="news.db")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--bm25-weight", type=float, default=0.7)
    args = ap.parse_args(argv)

    prompts = load_prompts(args.prompts)
    if not prompts:
        sys.exit(f"В {args.prompts} нет промптов (строки без # и непустые).")
    print(f"промптов: {len(prompts)}, llm={args.llm}, engine={args.engine}, top_k={args.top_k}")

    conn = open_db(args.db)
    run_query_topk = make_retriever(conn, args.engine, args.db, None,
                                    None, args.bm25_weight, verbose=True)

    with open(args.out, "w", encoding="utf-8") as out:
        out.write(f"llm={args.llm} model={args.model or '(default)'} engine={args.engine} "
                 f"top_k={args.top_k} bm25_weight={args.bm25_weight}\n")
        for i, q in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] {q}")
            t0 = time.time()
            buf = io.StringIO()
            with redirect_stdout(buf):
                ask.answer(conn, run_query_topk, q, args.top_k, args.llm, args.model)
            dt = time.time() - t0

            out.write("\n" + "=" * 100 + "\n")
            out.write(f"[{i}] ВОПРОС: {q}\n")
            out.write(f"({dt:.1f}s)\n\n")
            out.write(buf.getvalue())
            out.write("\n")
            out.flush()
            print(f"  -> {dt:.1f}s")

    print(f"\nготово: {args.out}")


if __name__ == "__main__":
    main()
