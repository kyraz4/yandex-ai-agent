#!/usr/bin/env python3
"""
Читает тексты из raw_data/, применяет замены из terms_map.json
и сохраняет обезличенные документы в prepared_data/.
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("raw_data")
PREPARED_DIR = Path("prepared_data")
TERMS_MAP_PATH = Path("terms_map.json")  # можно положить в prepared_data/

PREPARED_DIR.mkdir(exist_ok=True)


def load_terms_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_pattern(terms: dict[str, str]) -> re.Pattern:
    """
    Строит одно регулярное выражение из всех ключей словаря.
    Сортируем по убыванию длины, чтобы «Agent Smith» заменился раньше, чем «Smith».
    Используем границы слов, чтобы не портить «neon» при замене «Neo».
    """
    keys = sorted(terms.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in keys]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def replace_terms(text: str, terms: dict[str, str], pattern: re.Pattern) -> str:
    def _sub(match: re.Match) -> str:
        original = match.group(0)
        # Ищем ключ без учёта регистра
        for key, value in terms.items():
            if key.lower() == original.lower():
                # Пытаемся сохранить регистр первой буквы
                if original[0].isupper() and value:
                    return value[0].upper() + value[1:]
                return value
        return original  # fallback

    return pattern.sub(_sub, text)


def main() -> None:
    terms = load_terms_map(TERMS_MAP_PATH)
    pattern = build_pattern(terms)

    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        print(f"❌ В {RAW_DIR}/ нет .txt файлов. Сначала запустите fetch_pages.py")
        return

    print(f"Найдено {len(files)} документов. Применяю замены…\n")
    for src in files:
        text = src.read_text(encoding="utf-8")
        new_text = replace_terms(text, terms, pattern)
        dst = PREPARED_DIR / src.name
        dst.write_text(new_text, encoding="utf-8")
        print(f"  ✅ {src.name} → {dst}")

    # Копируем словарь замен в prepared_data/, чтобы он был «рядом» с базой
    dst_map = PREPARED_DIR / "terms_map.json"
    dst_map.write_text(
        json.dumps(terms, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n📄 Словарь замен скопирован → {dst_map}")
    print("Готово.")


if __name__ == "__main__":
    main()
