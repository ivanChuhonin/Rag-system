"""
Парсер постов со стен ВК-пабликов через VK API (MVP, шаг 1 — "Сбор данных").

wall.get отдаёт всю историю стены постранично через offset — глубина
ограничена только объёмом паблика, так что для набора большого корпуса
(десятки тысяч постов) это надёжнее, чем постраничный листинг сайта

Нужен токен доступа VK API (пользовательский или сервисный, с правом читать
стену — обычно достаточно дефолтных прав для открытых пабликов). Впиши его
в VK_ACCESS_TOKEN в .env (см. .env.example, читает config.py) — в коде
токен не хранится.

Список пабликов — в константе GROUPS ниже (впиши screen_name или id), либо
передай через --groups при запуске (он переопределяет GROUPS).

Запуск:
    python vk_parse.py                                   # берёт паблики из GROUPS
    python vk_parse.py --groups championat --limit-per-group 200 --verbose

Пишет в таблицу documents (storage.py), источник помечается как
"vk:<screen_name>". Дальше chunk.py работает без изменений.

Зависимости: только `requests`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

from config import vk_cfg
from storage import open_db, clear_db, save_document
from text_clean import clean_text

# --------------------------------------------------------------------------- #
# Конфиг
# --------------------------------------------------------------------------- #

# Впиши сюда пабликы, которые парсим: screen_name (как в vk.com/<это>) или
# числовой id группы. Можно переопределить флагом --groups при запуске.
GROUPS: list[str] = [
    "kinoprotebya",
    "itfollowspub",
    "kinoartmag",
    "ponimai_kino",
    "off_rkvt",
]

VK_API_VERSION = "5.199"
VK_API_URL = "https://api.vk.com/method/{method}"

REQUEST_TIMEOUT = 25
POSTS_PER_CALL = 100        # максимум, который отдаёт wall.get за один вызов
MIN_REQUEST_INTERVAL = 0.35  # ВК: не больше ~3 запросов/сек на токен
MAX_RETRIES = 4
MIN_TEXT_LEN = 60           # посты короче — не тащим (у ВК они куда компактнее статей)
PROGRESS_EVERY = 10_000     # промежуточный итог каждые N собранных постов

RATE_LIMIT_CODE = 6         # "Too many requests per second"
FATAL_ERROR_CODES = {5, 27, 28, 100, 113, 15, 203}  # токен/доступ — ретраить бессмысленно


class VKFatalError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# HTTP / VK API
# --------------------------------------------------------------------------- #

class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._next_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._next_at - now
        self._next_at = max(now, self._next_at) + self.min_interval
        if delay > 0:
            time.sleep(delay)


def vk_call(session: requests.Session, limiter: RateLimiter, method: str,
            params: dict, access_token: str) -> dict:
    """Вызывает VK API method, ретраит сетевые ошибки и rate-limit. Кидает
    VKFatalError на ошибках токена/доступа (ретраить их бессмысленно)."""
    payload = dict(params)
    payload["access_token"] = access_token
    payload["v"] = VK_API_VERSION

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            r = session.get(VK_API_URL.format(method=method), params=payload, timeout=REQUEST_TIMEOUT)
            data = r.json()
        except requests.RequestException as e:
            print(f"  ! попытка {attempt}/{MAX_RETRIES}: {type(e).__name__} ({method})", file=sys.stderr)
            time.sleep(1.0 * attempt)
            continue
        except ValueError:
            print(f"  ! попытка {attempt}/{MAX_RETRIES}: не-JSON ответ ({method})", file=sys.stderr)
            time.sleep(1.0 * attempt)
            continue

        if "error" in data:
            err = data["error"]
            code, msg = err.get("error_code"), err.get("error_msg", "")
            if code == RATE_LIMIT_CODE:
                time.sleep(1.0 * attempt)  # словили rate-limit — подождать и повторить
                continue
            if code in FATAL_ERROR_CODES:
                raise VKFatalError(f"VK API {method}: [{code}] {msg}")
            print(f"  ! VK API {method}: [{code}] {msg}", file=sys.stderr)
            time.sleep(1.0 * attempt)
            continue

        return data["response"]

    raise VKFatalError(f"VK API {method}: не удалось получить ответ за {MAX_RETRIES} попыток")


def resolve_owner_id(session: requests.Session, limiter: RateLimiter,
                     group_ref: str, access_token: str) -> tuple[int, str]:
    """group_ref: screen_name ('championat') или числовой id (с минусом или без).
    Возвращает (owner_id для wall.get — отрицательный, читаемое имя)."""
    ref = group_ref.strip().lstrip("@")
    if re.fullmatch(r"-?\d+", ref):
        gid = abs(int(ref))
        return -gid, ref
    resp = vk_call(session, limiter, "groups.getById", {"group_id": ref}, access_token)
    groups = resp["groups"] if isinstance(resp, dict) else resp
    if not groups:
        raise VKFatalError(f"группа '{ref}' не найдена")
    g = groups[0]
    return -int(g["id"]), g.get("screen_name") or ref


# --------------------------------------------------------------------------- #
# Разбор поста
# --------------------------------------------------------------------------- #

def post_to_doc(post: dict, owner_id: int) -> dict | None:
    text = (post.get("text") or "").strip()

    # репост без своего комментария — берём текст оригинала
    if not text:
        for orig in post.get("copy_history") or []:
            orig_text = (orig.get("text") or "").strip()
            if orig_text:
                text = orig_text
                break

    text = clean_text(text)
    if len(text) < MIN_TEXT_LEN:
        return None

    title = text.split("\n", 1)[0].strip()
    title = re.sub(r"\s+", " ", title)[:120] or f"Пост {post['id']}"

    url = f"https://vk.com/wall{owner_id}_{post['id']}"
    date = datetime.fromtimestamp(post["date"], tz=timezone.utc).isoformat()

    return {"title": title, "url": url, "text": text, "date": date}


# --------------------------------------------------------------------------- #
# Обход одной группы
# --------------------------------------------------------------------------- #

def crawl_group(session: requests.Session, limiter: RateLimiter, access_token: str,
                conn, group_ref: str, limit_per_group: int | None,
                verbose: bool, stats: dict, next_milestone: list[int],
                skip_existing: bool = False) -> None:
    owner_id, name = resolve_owner_id(session, limiter, group_ref, access_token)
    print(f"\n[группа] {group_ref} -> owner_id={owner_id}")

    if skip_existing:
        already = conn.execute(
            "SELECT 1 FROM documents WHERE source = ? LIMIT 1", (f"vk:{name}",)
        ).fetchone()
        if already:
            print(f"  уже есть в базе (source=vk:{name}) — пропускаю, не трогаю.")
            return

    offset = 0
    total = None
    got_for_group = 0

    while True:
        if limit_per_group is not None and got_for_group >= limit_per_group:
            break
        count = POSTS_PER_CALL
        if limit_per_group is not None:
            count = min(count, limit_per_group - got_for_group)

        resp = vk_call(session, limiter, "wall.get",
                       {"owner_id": owner_id, "offset": offset, "count": count}, access_token)
        if total is None:
            total = resp["count"]
            print(f"  всего постов на стене: {total}")

        items = resp["items"]
        if not items:
            break

        for post in items:
            stats["posts_seen"] += 1
            got_for_group += 1
            doc = post_to_doc(post, owner_id)
            if not doc:
                stats["skipped_empty"] += 1
                continue
            status = save_document(conn, f"vk:{name}", doc)
            stats[status] += 1
            if verbose:
                mark = {"new": "+", "dup-url": "=", "dup-text": "~"}[status]
                print(f"  {mark} [{doc['date'][:10]}] {doc['title'][:90]}")

            if stats["new"] >= next_milestone[0]:
                print(f"\n>>> собрано новых постов: {stats['new']}\n")
                next_milestone[0] += PROGRESS_EVERY

        offset += len(items)
        print(f"  {group_ref}: обработано {offset}/{total}")

        if offset >= total or len(items) < count:
            break


# --------------------------------------------------------------------------- #
# Основной цикл
# --------------------------------------------------------------------------- #

def run(groups: list[str], db_path: str, limit_per_group: int | None,
        verbose: bool, keep_db: bool, access_token: str, skip_existing: bool = False) -> None:
    if not access_token:
        sys.exit(
            "Не задан VK_ACCESS_TOKEN. Получи токен доступа VK API (с правом читать стену) "
            "и впиши его в VK_ACCESS_TOKEN в .env (см. .env.example). Либо передай --token напрямую."
        )

    session = requests.Session()
    limiter = RateLimiter(MIN_REQUEST_INTERVAL)
    conn = open_db(db_path)

    if keep_db:
        print("--keep-db: старые записи в базе сохранены.\n")
    else:
        removed = clear_db(conn)
        print(f"очистил базу: удалено {removed} старых записей.\n")

    stats = {"posts_seen": 0, "new": 0, "dup-url": 0, "dup-text": 0, "skipped_empty": 0}
    next_milestone = [PROGRESS_EVERY]

    for group_ref in groups:
        try:
            crawl_group(session, limiter, access_token, conn, group_ref,
                       limit_per_group, verbose, stats, next_milestone, skip_existing)
        except VKFatalError as e:
            print(f"  ! пропускаю группу '{group_ref}': {e}", file=sys.stderr)
            continue

    _report(stats, db_path)
    conn.close()


def _report(stats: dict, db_path: str) -> None:
    print("\n" + "-" * 60)
    print(f"постов просмотрено : {stats['posts_seen']}")
    print(f"  новых            : {stats['new']}")
    print(f"  дубль url        : {stats['dup-url']}")
    print(f"  дубль текст      : {stats['dup-text']}")
    print(f"  пустых/коротких  : {stats['skipped_empty']}")
    print(f"база               : {os.path.abspath(db_path)}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Парсер постов ВК-пабликов -> SQLite")
    ap.add_argument("--groups", default=None,
                    help="список пабликов через запятую: screen_name или числовой id "
                         "(например: championat,footballtoday,sports_ru). "
                         "Если не задан — берётся список GROUPS в начале файла.")
    ap.add_argument("--db", default="news.db", help="путь к файлу SQLite (общий с chunk.py)")
    ap.add_argument("--limit-per-group", type=int, default=None,
                    help="максимум постов с одной группы (по умолчанию — вся стена)")
    ap.add_argument("--verbose", action="store_true", help="подробный лог по каждому посту")
    ap.add_argument("--keep-db", action="store_true",
                    help="не очищать базу перед запуском (по умолчанию старые записи стираются)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="пропускать пабликы, у которых в базе уже есть посты (не перекачивать заново); "
                         "имеет смысл вместе с --keep-db, когда GROUPS расширили новыми пабликами")
    ap.add_argument("--token", default=None, help="токен VK API (иначе берётся из VK_ACCESS_TOKEN)")
    args = ap.parse_args(argv)

    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    else:
        groups = [g.strip() for g in GROUPS if g.strip()]
    if not groups:
        sys.exit(
            "Не заданы паблики. Впиши screen_name/id в список GROUPS в начале vk_parse.py, "
            "либо передай --groups championat,footballtoday,..."
        )

    token = args.token or vk_cfg.access_token
    run(groups, args.db, args.limit_per_group, args.verbose, args.keep_db, token, args.skip_existing)


if __name__ == "__main__":
    main()
