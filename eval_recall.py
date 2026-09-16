"""
Оценка ретривера: Recall@K / Hit@K через LLM-as-judge (MVP-расширение
"Оценка качества" из плана).

Настоящий Recall@K нужен эталон — заранее известно, какой документ реально
релевантен вопросу. Размечать вручную дорого, поэтому эталон строим
LLM-судьёй: на каждый вопрос берём пул топ-N кандидатов от ретривера (шире,
чем K, которые сравниваем) и просим модель разметить каждый — релевантен
вопросу по существу или нет. Это не человеческая разметка, а прокси через
LLM; метрика ровно настолько честная, насколько адекватен судья — поэтому
он обязан вернуть короткое обоснование к каждой оценке, чтобы разметку
можно было выборочно перепроверить в JSON-выгрузке.

Запуск:
    python eval_recall.py --prompts test_prompts.txt --pool-size 10
    python eval_recall.py --ks 1,3,5 --pool-size 5 --out eval_quick.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from ask import ollama_chat, LLMError
from search import make_retriever, open_db

JUDGE_SYSTEM_PROMPT = (
    "Ты — асессор поисковой системы про кино. Тебе дан вопрос пользователя и "
    "пронумерованный список найденных фрагментов постов. Для каждого фрагмента "
    "реши: содержит ли он информацию, которая реально помогает ответить на "
    "вопрос по существу (relevant=true), или нет (relevant=false) — фрагмент "
    "на смежную тему без ответа по существу тоже relevant=false.\n"
    "Ответь СТРОГО JSON-массивом, без текста до/после и без markdown-разметки, "
    "формат:\n"
    '[{"n": 1, "relevant": true, "reason": "..."}, {"n": 2, "relevant": false, "reason": "..."}]\n'
    "reason — не больше 12 слов, на русском."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def judge_pool(question: str, pool: list[dict]) -> list[dict]:
    """pool — результат ретривера (нужен r['text'], полный текст чанка).
    Возвращает список {'n','chunk_id','relevant','reason'} по каждому кандидату,
    в исходном порядке ранжирования."""
    listing = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(pool))
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Вопрос: {question}\n\nФрагменты:\n{listing}"},
    ]
    raw = ollama_chat(messages, num_predict=max(800, len(pool) * 60))
    cleaned = _CODE_FENCE_RE.sub("", raw.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"судья вернул не-JSON: {e}\n{raw[:300]}")

    by_n = {item.get("n"): item for item in parsed if isinstance(item, dict)}
    out = []
    for i, r in enumerate(pool):
        item = by_n.get(i + 1, {})
        out.append({
            "n": i + 1,
            "chunk_id": r["chunk_id"],
            "relevant": bool(item.get("relevant", False)),
            "reason": item.get("reason", "(судья не дал оценку для этого номера)"),
        })
    return out


def compute_metrics(judged_questions: list[dict], ks: list[int]) -> dict:
    n_questions = len(judged_questions)
    hit = {k: 0 for k in ks}
    recall_sum = {k: 0.0 for k in ks}
    recall_count = {k: 0 for k in ks}

    for q in judged_questions:
        judgments = q["judgments"]
        total_relevant = sum(1 for j in judgments if j["relevant"])
        for k in ks:
            found = sum(1 for j in judgments[:k] if j["relevant"])
            if found > 0:
                hit[k] += 1
            if total_relevant > 0:
                recall_sum[k] += found / total_relevant
                recall_count[k] += 1

    by_k = {}
    for k in ks:
        by_k[k] = {
            "hit_rate": hit[k] / n_questions if n_questions else 0.0,
            "mean_recall": (recall_sum[k] / recall_count[k]) if recall_count[k] else None,
            "questions_with_relevant": recall_count[k],
        }
    return {"n_questions": n_questions, "by_k": by_k}


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Recall@K / Hit@K ретривера через LLM-as-judge")
    ap.add_argument("--prompts", default="test_prompts.txt")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--pool-size", type=int, default=10, help="сколько кандидатов судить на вопрос")
    ap.add_argument("--ks", default="1,3,5,10", help="через запятую, максимум не больше --pool-size")
    ap.add_argument("--engine", choices=["bm25", "faiss", "hybrid"], default="hybrid")
    ap.add_argument("--db", default="news.db")
    ap.add_argument("--bm25-weight", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="оценить только первые N вопросов (для смоук-теста)")
    args = ap.parse_args(argv)

    ks = sorted(int(x) for x in args.ks.split(","))
    if ks[-1] > args.pool_size:
        sys.exit(f"максимальный K ({ks[-1]}) не может быть больше --pool-size ({args.pool_size})")

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        sys.exit(f"в {args.prompts} нет промптов")

    conn = open_db(args.db)
    run_query = make_retriever(conn, args.engine, args.db, None, None, args.bm25_weight, verbose=True)

    judged_questions = []
    t_start = time.time()
    for i, q in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {q}")
        pool = run_query(q, args.pool_size)
        if not pool:
            print("  ! ретривер ничего не нашёл, пропускаю")
            continue
        t0 = time.time()
        try:
            judgments = judge_pool(q, pool)
        except LLMError as e:
            print(f"  ! судья не ответил: {e}")
            continue
        dt = time.time() - t0
        n_rel = sum(1 for j in judgments if j["relevant"])
        print(f"  -> {dt:.1f}s, релевантных по мнению судьи: {n_rel}/{len(pool)}")
        judged_questions.append({"question": q, "judgments": judgments})

    metrics = compute_metrics(judged_questions, ks)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "questions": judged_questions}, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 60)
    print(f"вопросов оценено : {metrics['n_questions']}  (за {time.time() - t_start:.0f}с)")
    for k in ks:
        m = metrics["by_k"][k]
        recall_str = f"{m['mean_recall']:.2f}" if m["mean_recall"] is not None else "—"
        print(f"  K={k:<3} hit_rate={m['hit_rate']:.2f}  mean_recall={recall_str}  "
             f"(вопросов, где судья нашёл хоть что-то релевантное в пуле: {m['questions_with_relevant']})")
    print(f"\nподробности с обоснованиями: {args.out}")


if __name__ == "__main__":
    main()
