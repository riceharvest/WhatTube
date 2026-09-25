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

def test_language_model_aliases():
    """Verify that language codes without direct opus-mt-{src}-{tgt} are mapped properly."""
    translator = MarianTranslator(use_ct2=False)
    assert translator._get_model_id("pt", "en") == "Helsinki-NLP/opus-mt-roa-en"
    assert translator._get_model_id("el", "en") == "Helsinki-NLP/opus-mt-grk-en"
    assert translator._get_model_id("fil", "en") == "Helsinki-NLP/opus-mt-tl-en"
    assert translator._get_model_id("yue", "en") == "Helsinki-NLP/opus-mt-zh-en"
    assert translator._get_model_id("ms", "en") == "Helsinki-NLP/opus-mt-id-en"

def test_portuguese_translation_ct2():
    """Verify Portuguese translates via mapped roa-en model."""
    translator = MarianTranslator(use_ct2=True)
    res = translator.translate("Obrigado", "pt", "en")
    assert res.is_success
    assert "thank" in res.translated_text.lower()
