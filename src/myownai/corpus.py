from __future__ import annotations

import re
import unicodedata


_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def normalize_article_text(text: str) -> str:
    """Normalize whitespace without rewriting article wording."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\x00", " ")
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r" *\\n *", "\\n", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text.strip()


def language_character_ratio(text: str, language: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    if language == "ar":
        def is_arabic(char: str) -> bool:
            codepoint = ord(char)
            return any(start <= codepoint <= end for start, end in _ARABIC_RANGES)
        matching = sum(is_arabic(char) for char in letters)
    elif language == "en":
        matching = sum(char.isascii() and char.isalpha() for char in letters)
    else:
        raise ValueError(f"Unsupported language: {language}")
    return matching / len(letters)


def clean_article_text(
    text: str,
    language: str,
    min_chars: int = 250,
    max_chars: int = 12000,
    min_language_ratio: float = 0.2,
) -> str | None:
    """Return a bounded article if it passes basic size and script checks."""
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("Require 1 <= min_chars <= max_chars")
    cleaned = normalize_article_text(text)
    if len(cleaned) < min_chars:
        return None
    if sum(char.isalpha() for char in cleaned) < min_chars // 2:
        return None
    if language_character_ratio(cleaned, language) < min_language_ratio:
        return None
    return cleaned[:max_chars].strip()
