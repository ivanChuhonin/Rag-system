"""
Автогенерация тестовых вопросов для scripts.eval_recall — масштабирует
test_prompts.txt (20 вопросов руками) до любого N, не отрываясь от того,
что реально есть в корпусе.

Идея: берём стратифицированную выборку documents (поровну по каждому
source — иначе крупные VK-каналы перетянут выборку, а kinopoisk/маленькие
каналы почти не попадут), просим LLM (тот же ollama_chat, что и у судьи в
eval_recall.py) придумать ОДИН естественный вопрос пользователя, на который
текст отвечает по существу — не пересказ и не цитата, а то, как реальный
человек спросил бы в поиске/чате.

Хэндрайтед test_prompts.txt трогать не нужно (это стабильный baseline для
сравнения "было/стало") — результат сюда, в отдельный файл, дальше
scripts.eval_recall запускается с --prompts test_prompts_auto.txt.

Запуск (из корня репозитория, нужен настроенный Ollama — см. .env.example):
    python -m scripts.gen_test_prompts --n 120
    python -m scripts.gen_test_prompts --n 60 --out test_prompts_auto_small.txt
"""

from __future__ import annotations

import argparse
import random
import sys

from rag.ask import ollama_chat, LLMError
from rag.storage import get_connection

GEN_SYSTEM_PROMPT = (
    "Ты помогаешь составить тестовый набор вопросов для RAG-системы о кино. "
    "Тебе дан текст одного поста/статьи. Придумай ОДИН вопрос, который реальный "
    "пользователь мог бы задать в поиске или чате про кино, и на который этот "
    "текст по существу отвечает. Переформулируй своими словами, не цитируй текст "
    "напрямую и не упоминай, что у тебя есть какой-то конкретный пост/источник. "
    "Ответь СТРОГО одним вопросом на русском языке, одна строка, без кавычек, "
    "без нумерации, без пояснений до или после."
)

MAX_DOC_CHARS = 2000  # текста обычно хватает на вопрос и без полного поста


def sample_doc_ids(conn, n: int, seed: int | None) -> list[tuple[int, str]]:
    """Стратифицированная выборка: поровну doc_id с каждого source, пока не
    наберём n (источников может быть меньше/больше квоты — остаток
    доливается из самых больших по счёту, чтобы не терять N при неравных
    источниках)."""
    if seed is not None:
        conn.execute("SELECT setseed(%s)", (seed / 2147483647,))  # setseed хочет [-1, 1]

    sources = [row[0] for row in conn.execute("SELECT DISTINCT source FROM documents ORDER BY source")]
    if not sources:
        return []
    quota = max(1, n // len(sources))

    picked: list[tuple[int, str]] = []
    for source in sources:
        rows = conn.execute(
            "SELECT id, source FROM documents WHERE source = %s ORDER BY random() LIMIT %s",
            (source, quota),
        ).fetchall()
        picked.extend(rows)

    # не хватило из-за неравных источников — доливаем случайными из всей базы
    if len(picked) < n:
        have_ids = {doc_id for doc_id, _ in picked}
        extra = conn.execute(
            "SELECT id, source FROM documents WHERE id NOT IN %s ORDER BY random() LIMIT %s",
            (tuple(have_ids) or (-1,), n - len(picked)),
        ).fetchall()
        picked.extend(extra)

    random.shuffle(picked)
    return picked[:n]


def generate_question(text: str, model: str | None) -> str | None:
    messages = [
        {"role": "system", "content": GEN_SYSTEM_PROMPT},
        {"role": "user", "content": text[:MAX_DOC_CHARS]},
    ]
    try:
        raw = ollama_chat(messages, model=model, num_predict=120)
    except LLMError as e:
        print(f"  ! LLM не ответил: {e}", file=sys.stderr)
        return None

    question = raw.strip().strip('"').strip("«»").split("\n")[0].strip()
    if len(question) < 10 or "?" not in question:
        return None  # не похоже на вопрос — пропускаем, не чиним руками
    return question


def run(n: int, out_path: str, model: str | None, seed: int | None, verbose: bool) -> None:
    conn = get_connection()
    docs = sample_doc_ids(conn, n, seed)
    print(f"выбрано документов: {len(docs)} (цель: {n})")

    by_source: dict[str, int] = {}
    for _, source in docs:
        by_source[source] = by_source.get(source, 0) + 1
    for source, cnt in sorted(by_source.items()):
        print(f"  {source}: {cnt}")
    print()

    questions: list[str] = []
    seen: set[str] = set()

    for i, (doc_id, source) in enumerate(docs, 1):
        text = conn.execute("SELECT text FROM documents WHERE id = %s", (doc_id,)).fetchone()[0]
        q = generate_question(text, model)
        if not q or q in seen:
            print(f"[{i}/{len(docs)}] ! пропуск (doc_id={doc_id}, {source})")
            continue
        seen.add(q)
        questions.append(q)
        if verbose:
            print(f"[{i}/{len(docs)}] {q}")
        elif i % 20 == 0:
            print(f"[{i}/{len(docs)}] сгенерировано вопросов: {len(questions)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Автогенерация scripts.gen_test_prompts (n={n}, seed={seed}).\n")
        f.write("# Не ручная разметка — вопросы придуманы LLM по реальным документам корпуса.\n")
        for q in questions:
            f.write(q + "\n")

    print("\n" + "-" * 60)
    print(f"запрошено документов : {len(docs)}")
    print(f"вопросов сгенерировано: {len(questions)}")
    print(f"файл                 : {out_path}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Автогенерация тестовых вопросов из корпуса для eval_recall")
    ap.add_argument("--n", type=int, default=120, help="сколько вопросов сгенерировать (по умолчанию 120)")
    ap.add_argument("--out", default="test_prompts_auto.txt", help="куда писать результат")
    ap.add_argument("--model", default=None, help="модель Ollama, иначе OLLAMA_MODEL из .env")
    ap.add_argument("--seed", type=int, default=None, help="сид для воспроизводимой выборки документов")
    ap.add_argument("--verbose", action="store_true", help="печатать каждый сгенерированный вопрос")
    args = ap.parse_args(argv)

    run(args.n, args.out, args.model, args.seed, args.verbose)


if __name__ == "__main__":
    main()
