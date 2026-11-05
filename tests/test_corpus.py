from pathlib import Path

import pytest

from myownai.corpus import (
    clean_article_text,
    language_character_ratio,
    normalize_article_text,
)


def test_normalize_article_text_handles_tabs_nulls_and_blank_lines() -> None:
    value = "Alpha\t beta\x00\n\n\nGamma"
    assert normalize_article_text(value) == "Alpha beta\n\nGamma"


def test_language_character_ratio_for_arabic_and_english() -> None:
    assert language_character_ratio("مرحبا بالعالم", "ar") == 1.0
    assert language_character_ratio("Hello world", "en") == 1.0


def test_clean_article_text_rejects_short_or_wrong_script_text() -> None:
    assert clean_article_text("قصير", "ar", min_chars=20) is None
    assert clean_article_text("This is a sufficiently long English sentence. " * 5, "ar", min_chars=100) is None


def test_clean_article_text_caps_length_and_normalizes_whitespace() -> None:
    value = ("هذا نص عربي للتجربة والاختبار. " * 20) + "\n\n"
    cleaned = clean_article_text(value, "ar", min_chars=100, max_chars=120)
    assert cleaned is not None
    assert len(cleaned) <= 120
    assert "\n\n" not in cleaned


def test_clean_article_text_validates_limits() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        clean_article_text("text", "en", min_chars=20, max_chars=10)
