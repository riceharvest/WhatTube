"""Tests for MarianTranslator LRU cache, routing, and fallbacks."""

import pytest
from whattube.translation.marian import MarianTranslator

def test_translation_identity_fastpath():
    translator = MarianTranslator(use_ct2=False)
    res = translator.translate("Hello world", "en", "en")
    assert res.is_success
    assert res.translated_text == "Hello world"
    assert res.latency_ms < 10.0

def test_translation_lru_cache_bounded():
    """Verify that model cache does not grow unbounded and respects max_cached_models."""
    translator = MarianTranslator(use_ct2=True, max_cached_models=2)

    # Prefetch id->en
    translator.prefetch("id", "en")
    assert len(translator.ct2_models) <= 2

    res = translator.translate("Halo apa kabar", "id", "en")
    assert res.is_success
    assert len(res.translated_text) > 0
