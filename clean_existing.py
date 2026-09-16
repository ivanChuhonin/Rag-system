"""
Разовая (но реюзабельная) миграция: применяет text_clean.clean_text() к уже
сохранённым documents.text — на случай правил чистки, которые появились
после того, как данные уже собраны (наш случай: убрали разметку ВК/ссылки
из текста, который парсили раньше).

Пересчитывает text_hash под новый текст. Раз текст документов меняется —
существующие chunks (нарезаны из старого текста) становятся неактуальными,
поэтому чистит их же (через chunk.clear_chunks — заодно инвалидирует и
FAISS-индекс, если он был). После этого нужно перезапустить chunk.py и
embed_index.py.

Запуск:
    python clean_existing.py --db news.db
    python clean_existing.py --db news.db --dry-run    # только показать, что изменится
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys

import chunk as ck
from text_clean import clean_text


def run(db_path: str, dry_run: bool, sample: int) -> None:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, text FROM documents").fetchall()

    changed: list[tuple[int, str, str]] = []
    for doc_id, text in rows:
        cleaned = clean_text(text)
        if cleaned != text:
            changed.append((doc_id, text, cleaned))

    print(f"всего документов: {len(rows)}")
    print(f"будет изменено   : {len(changed)}")

    empty_after = [doc_id for doc_id, _, cleaned in changed if not cleaned.strip()]
    if empty_after:
        print(f"! станут пустыми после чистки: {len(empty_after)} (id: {empty_after[:10]}{'...' if len(empty_after) > 10 else ''})")

    for doc_id, before, after in changed[:sample]:
        print(f"\n--- id={doc_id} ---")
        print("до    :", before[:160].replace("\n", " \\n "))
        print("после :", after[:160].replace("\n", " \\n "))

    if dry_run:
        print("\n--dry-run: ничего не записано.")
        conn.close()
        return

    if not changed:
        print("\nнечего обновлять.")
        conn.close()
        return

    conn.executemany(
        "UPDATE documents SET text = ?, text_hash = ? WHERE id = ?",
        [(after, hashlib.sha256(after.encode("utf-8")).hexdigest(), doc_id)
         for doc_id, _, after in changed],
    )
    conn.commit()
    print(f"\nобновил {len(changed)} документов.")

    has_chunks = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    if has_chunks:
        removed = ck.clear_chunks(conn)
        print(f"текст изменился -> старые чанки неактуальны, удалил {removed} шт. "
             "Перезапусти chunk.py, потом embed_index.py.")

    conn.close()


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Применить clean_text() к уже сохранённым documents")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite")
    ap.add_argument("--dry-run", action="store_true", help="только показать, ничего не менять")
    ap.add_argument("--sample", type=int, default=5, help="сколько примеров изменений напечатать")
    args = ap.parse_args(argv)
    run(args.db, args.dry_run, args.sample)


if __name__ == "__main__":
    main()
