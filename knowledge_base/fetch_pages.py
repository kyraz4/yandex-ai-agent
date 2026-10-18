#!/usr/bin/env python3
"""
Выгружает указанные страницы с matrix.fandom.com через MediaWiki API
и сохраняет очищенный текст в папку raw_data/.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# ─── Настройки ────────────────────────────────────────────────────────────────
WIKI = "https://matrix.fandom.com"
API_URL = f"{WIKI}/api.php"
RAW_DIR = Path("raw_data")
RAW_DIR.mkdir(exist_ok=True)

PAGES = [
    # Персонажи
    "Neo",
    "Trinity",
    "Morpheus",
    "Agent Smith",
    "The Oracle",
    "The Architect",
    "Cypher",
    "Niobe",
    "Commander Lock",
    "The Merovingian",
    "Persephone",
    "Seraph",
    "The Keymaker",
    "Link",
    "Tank",
    "Dozer",
    "Apoc",
    "Switch",
    "Mouse",
    "Agent Brown",
    "Agent Jones",
    "The Kid",
    "Bane",
    # Локации, технологии, организации
    "Zion",
    "Machine City",
    "The Matrix",
    "Nebuchadnezzar",
    "Construct",
    "Hardline",
    "Redpill",
    "Bluepill",
    "The Source",
    "The Trainman",
    "Mobil Avenue",
    "Club Hel",
]

HEADERS = {
    "User-Agent": "RAG-Knowledge-Base-Builder/1.0 (educational project)"
}


# ─── Вспомогательные функции ──────────────────────────────────────────────────
def slugify(title: str) -> str:
    """Превращает название страницы в безопасное имя файла."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug


def clean_html(html: str) -> str:
    """Удаляет из HTML разметку, infobox'ы, скрипты и оставляет основной текст."""
    soup = BeautifulSoup(html, "html.parser")

    # Удаляем служебные блоки, которые не несут полезного текста
    for selector in [
        "script", "style", "noscript",
        ".mw-editsection",        # ссылки «править»
        ".navbox", ".infobox",    # навигационные таблицы и карточки
        ".toc", "#toc",           # оглавление
        ".reference", ".references",
        "sup.reference",
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Заменяем переносы строк и множественные пробелы
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_page(title: str) -> str | None:
    """Запрашивает HTML страницы через action=parse и возвращает очищенный текст."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "disablelimitreport": "1",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f"  ⚠️  API error для «{title}»: {data['error'].get('info')}")
            return None

        html = data.get("parse", {}).get("text", "")
        if not html:
            print(f"  ⚠️  Пустой ответ для «{title}»")
            return None

        return clean_html(html)

    except requests.RequestException as e:
        print(f"  ❌ Сетевая ошибка для «{title}»: {e}")
        return None
    except json.JSONDecodeError:
        print(f"  ❌ Некорректный JSON для «{title}»")
        return None


# ─── Основной цикл ────────────────────────────────────────────────────────────
def main() -> None:
    success, failed = 0, 0

    for i, title in enumerate(PAGES, 1):
        print(f"[{i:02d}/{len(PAGES)}] {title} …")
        text = fetch_page(title)

        if text:
            slug = slugify(title)
            out_path = RAW_DIR / f"{slug}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"  ✅ сохранено → {out_path} ({len(text)} символов)")
            success += 1
        else:
            failed += 1

        # Вежливая пауза, чтобы не перегружать Fandom API
        time.sleep(0.7)

    print(f"\nГотово: успешно {success}, ошибок {failed}")


if __name__ == "__main__":
    main()
