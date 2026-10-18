"""
Парсер новостей с сайта kinopoisk.ru/media/news/ (доп. источник данных,
расширяет корпус поверх ВК-пабликов — см. vk_parse.py).

Страница новостей отрендерена на сервере (SSR): обычный GET без JS уже
отдаёт тело статьи в HTML, так что requests + BeautifulSoup достаточно,
без headless-браузера.

Схема обхода:
  1. Листинг /media/news/ (и ?page=N) отдаёт карточки со ссылками на
     статьи (`a.post-feature-card__link`, href вида /media/news/<id>/).
     Листаем страницы по порядку, пока не кончатся новые ссылки или не
     упрёмся в --max-pages.
  2. Каждая статья скачивается отдельным GET; текст вытаскивается из
     `div.media-news__body-content[itemprop="articleBody"]` по
     блочным тегам (p/h2/h3/h4/li/blockquote), заголовок — из
     `h1 span[itemprop="headline"]`, дата — из атрибута `content` на
     `span[itemprop="datePublished"]` (уже ISO-дата, без времени).

Запуск (из корня репозитория):
    python -m scripts.kinopoisk_parse --max-pages 50
    python -m scripts.kinopoisk_parse --max-pages 5 --verbose

Пишет в таблицу documents (storage.py), источник помечается как
"kinopoisk:news". Дальше chunk.py работает без изменений.

Зависимости: requests, beautifulsoup4 (html.parser — без lxml).
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

from rag.storage import get_connection, clear_db, save_document
from rag.text_clean import clean_text

BASE_URL = "https://www.kinopoisk.ru"
LISTING_URL = f"{BASE_URL}/media/news/"

REQUEST_TIMEOUT = 20
MIN_REQUEST_INTERVAL = 1.0   # вежливая пауза между запросами — не долбим сайт
MAX_RETRIES = 3
MIN_TEXT_LEN = 200           # статьи короче — скорее всего анонс без текста

ARTICLE_HREF_RE = re.compile(r"^/media/news/(\d+)/$")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

BLOCK_TAGS = ("p", "h2", "h3", "h4", "li", "blockquote")


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


def fetch(session: requests.Session, limiter: RateLimiter, url: str) -> str | None:
    """GET с ретраями на сетевые ошибки/5xx. Возвращает None, если сайт
    так и не ответил — вызывающий код сам решает, что с этим делать
    (пропустить страницу/статью, не обрушивая весь прогон)."""
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            print(f"  ! попытка {attempt}/{MAX_RETRIES}: {type(e).__name__} ({url})", file=sys.stderr)
            time.sleep(1.5 * attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            print(f"  ! попытка {attempt}/{MAX_RETRIES}: HTTP {r.status_code} ({url})", file=sys.stderr)
            time.sleep(1.5 * attempt)
            continue
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    print(f"  ! сдаюсь после {MAX_RETRIES} попыток: {url}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Листинг -> ссылки на статьи
# --------------------------------------------------------------------------- #

def extract_article_links(html: str) -> list[str]:
    """Берём ссылки по href-паттерну, а не по классу карточки: на листинге
    вперемешку несколько видов карточек (feature/обычная), часть из них
    держится на CSS-module классах без стабильного имени — только сам
    href вида /media/news/<id>/ общий для всех."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ARTICLE_HREF_RE.match(href):
            links.append(BASE_URL + href)
    return links


def discover_links(session: requests.Session, limiter: RateLimiter,
                    max_pages: int, verbose: bool) -> list[str]:
    """Листает /media/news/?page=N по порядку, собирая ссылки на статьи.
    Останавливается, когда страница не добавила ни одной новой ссылки
    (конец листинга), или по достижении max_pages.

    Страница 1 (голый /media/news/ без ?page=) — особый случай: по ней
    иногда отдаётся не обычный листинг, а антибот-заглушка/кэш-заглушка
    (бывает у Яндекса на корневых урлах), тогда как /media/news/?page=2
    и далее по-прежнему рендерятся нормально. Поэтому если именно страница
    1 не загрузилась или не дала ни одной карточки — это не считается
    "концом листинга", а просто пропускается: обход продолжается со
    страницы 2."""
    seen: dict[str, None] = {}  # dict, не set — сохраняем порядок обнаружения
    for page in range(1, max_pages + 1):
        url = LISTING_URL if page == 1 else f"{LISTING_URL}?page={page}"
        html = fetch(session, limiter, url)
        if html is None:
            if page == 1:
                print("  ! страница 1 не загрузилась (похоже на блокировку именно "
                      "корневого урла) — пропускаю её, продолжаю со страницы 2.", file=sys.stderr)
                continue
            print(f"  ! страница {page} не загрузилась — останавливаю листинг здесь.", file=sys.stderr)
            break
        links = extract_article_links(html)
        before = len(seen)
        for link in links:
            seen.setdefault(link, None)
        added = len(seen) - before
        print(f"  листинг стр. {page}: найдено {len(links)} карточек, новых ссылок {added}, всего {len(seen)}")
        if verbose:
            for link in links:
                print(f"    {link}")
        if added == 0:
            if page == 1:
                print("  страница 1 без карточек (похоже на антибот/заглушку именно "
                      "на корневом урле) — пропускаю её, продолжаю со страницы 2.")
                continue
            print("  новых ссылок на этой странице нет — конец листинга.")
            break
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# Статья -> doc
# --------------------------------------------------------------------------- #

def parse_article(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one('h1.media-news__title span[itemprop="headline"]')
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    date_el = soup.select_one('span.media-news__published-date[itemprop="datePublished"]')
    date = date_el.get("content") if date_el else None

    body_el = soup.select_one('div.media-news__body-content[itemprop="articleBody"]')
    if not body_el:
        return None

    for junk in body_el.select('[class*="adfox"], [class*="foxad"], script, style'):
        junk.decompose()

    parts = [
        t for el in body_el.find_all(BLOCK_TAGS)
        if (t := el.get_text(" ", strip=True))
    ]
    text = clean_text("\n\n".join(parts))
    if len(text) < MIN_TEXT_LEN:
        return None

    return {"title": title, "url": url, "text": text, "date": date}


# --------------------------------------------------------------------------- #
# Основной цикл
# --------------------------------------------------------------------------- #

def run(max_pages: int, limit: int | None, verbose: bool,
        keep_db: bool, skip_existing_urls: bool) -> None:
    session = requests.Session()
    limiter = RateLimiter(MIN_REQUEST_INTERVAL)
    conn = get_connection()

    if keep_db:
        print("--keep-db: старые записи в базе сохранены.\n")
    else:
        removed = clear_db(conn)
        print(f"очистил базу: удалено {removed} старых записей.\n")

    existing_urls: set[str] = set()
    if skip_existing_urls:
        existing_urls = {
            row[0] for row in conn.execute(
                "SELECT url FROM documents WHERE source = %s", ("kinopoisk:news",)
            ).fetchall()
        }
        print(f"уже в базе (source=kinopoisk:news): {len(existing_urls)} статей — их пропущу.\n")

    print(f"[листинг] {LISTING_URL} (до {max_pages} стр.)")
    links = discover_links(session, limiter, max_pages, verbose)
    print(f"\nвсего ссылок на статьи: {len(links)}\n")

    stats = {"seen": 0, "new": 0, "dup-url": 0, "dup-text": 0, "skipped_short": 0, "skipped_parse": 0}

    for url in links:
        if limit is not None and stats["new"] >= limit:
            break
        if url in existing_urls:
            continue

        stats["seen"] += 1
        html = fetch(session, limiter, url)
        if html is None:
            stats["skipped_parse"] += 1
            continue

        doc = parse_article(html, url)
        if not doc:
            stats["skipped_parse"] += 1
            if verbose:
                print(f"  ! не удалось распарсить (нет текста/структуры): {url}")
            continue
        if len(doc["text"]) < MIN_TEXT_LEN:
            stats["skipped_short"] += 1
            continue

        status = save_document(conn, "kinopoisk:news", doc)
        stats[status] += 1
        if verbose:
            mark = {"new": "+", "dup-url": "=", "dup-text": "~"}[status]
            print(f"  {mark} [{doc['date'] or '????-??-??'}] {doc['title'][:90]}")

        if stats["new"] and stats["new"] % 100 == 0:
            print(f"\n>>> собрано новых статей: {stats['new']}\n")

    _report(stats)
    conn.close()


def _report(stats: dict) -> None:
    print("\n" + "-" * 60)
    print(f"статей просмотрено : {stats['seen']}")
    print(f"  новых            : {stats['new']}")
    print(f"  дубль url        : {stats['dup-url']}")
    print(f"  дубль текст      : {stats['dup-text']}")
    print(f"  слишком коротких : {stats['skipped_short']}")
    print(f"  не распарсилось  : {stats['skipped_parse']}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Парсер новостей kinopoisk.ru/media/news/ -> Postgres")
    ap.add_argument("--max-pages", type=int, default=50,
                     help="максимум страниц листинга /media/news/?page=N (по умолчанию 50)")
    ap.add_argument("--limit", type=int, default=None,
                     help="максимум НОВЫХ статей за прогон (по умолчанию — все найденные)")
    ap.add_argument("--verbose", action="store_true", help="подробный лог по каждой странице/статье")
    ap.add_argument("--keep-db", action="store_true",
                     help="не очищать базу перед запуском (по умолчанию старые записи стираются)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="пропускать статьи, чей url уже есть в базе с source=kinopoisk:news "
                          "(не скачивать их заново); имеет смысл вместе с --keep-db для инкрементальных дозагрузок")
    args = ap.parse_args(argv)

    run(args.max_pages, args.limit, args.verbose, args.keep_db, args.skip_existing)


if __name__ == "__main__":
    main()
