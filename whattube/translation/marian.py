import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import torch
from transformers import MarianMTModel, MarianTokenizer

from whattube.translation.base import BaseTranslator, TranslationResult

logger = logging.getLogger("WhatTube.Marian")

try:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    HAS_CTRANSLATE2 = True
except ImportError:
    HAS_CTRANSLATE2 = False

LANGUAGE_MODEL_ALIASES = {
    "pt": "roa",       # Portuguese maps to Helsinki-NLP Romance-to-English
    "fil": "tl",      # Filipino maps to Tagalog
    "tagalog": "tl",
    "yue": "zh",      # Cantonese maps to Chinese
    "cantonese": "zh",
    "ms": "id",       # Malay maps to Indonesian
    "malay": "id",
    "jw": "id",       # Javanese maps to Indonesian
    "jv": "id",       # Javanese maps to Indonesian
    "su": "id",       # Sundanese maps to Indonesian
    "el": "grk",      # Greek maps to Helsinki-NLP Greek-to-English (opus-mt-grk-en)
    "greek": "grk",
}

class MarianTranslator(BaseTranslator):
    """Translates text on CPU using CTranslate2 INT8 accelerated Helsinki-NLP/opus-mt models,
    with bounded LRU caching and automatic fallback to PyTorch MarianMTModel.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: str = "models",
        use_ct2: bool = True,
        max_cached_models: int = 5,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.use_ct2 = use_ct2 and HAS_CTRANSLATE2
        self.max_cached_models = max_cached_models

        # Bounded LRU caches: (src, tgt) -> (model/translator, tokenizer)
        self.models: OrderedDict[tuple[str, str], tuple[MarianMTModel, MarianTokenizer]] = OrderedDict()
        self.ct2_models: OrderedDict[tuple[str, str], tuple[any, MarianTokenizer]] = OrderedDict()
        self._lock = threading.Lock()

    def _get_model_id(self, src: str, tgt: str) -> str:
        mapped_src = LANGUAGE_MODEL_ALIASES.get(src.lower(), src.lower())
        mapped_tgt = LANGUAGE_MODEL_ALIASES.get(tgt.lower(), tgt.lower())
        return f"Helsinki-NLP/opus-mt-{mapped_src}-{mapped_tgt}"

    def _get_ct2_dir(self, src: str, tgt: str) -> str:
        mapped_src = LANGUAGE_MODEL_ALIASES.get(src.lower(), src.lower()).replace("-", "_")
        mapped_tgt = LANGUAGE_MODEL_ALIASES.get(tgt.lower(), tgt.lower()).replace("-", "_")
        return os.path.join(self.cache_dir, f"ct2_opus_mt_{mapped_src}_{mapped_tgt}")

    def prefetch(self, src: str, tgt: str) -> bool:
        """Prefetch and pre-convert a language pair into local INT8 cache."""
        if src.lower() == tgt.lower():
            return True
        return self._load_ct2_model(src.lower(), tgt.lower()) is not None

    def _load_ct2_model(self, src: str, tgt: str) -> tuple[Any, MarianTokenizer] | None:
        """Loads or converts a CTranslate2 INT8 model with bounded LRU eviction."""
        if not self.use_ct2:
            return None

        pair = (src, tgt)
        ct2_dir = self._get_ct2_dir(src, tgt)
        model_id = self._get_model_id(src, tgt)

        with self._lock:
            if pair in self.ct2_models:
                self.ct2_models.move_to_end(pair)
                return self.ct2_models[pair]

            try:
                tokenizer = MarianTokenizer.from_pretrained(model_id, local_files_only=True)
            except Exception:
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_id)
                except Exception as e:
                    logger.warning(f"[-] Could not load Marian tokenizer for {model_id}: {e}")
                    return None

            # Check if CT2 model directory exists
            if not (os.path.isdir(ct2_dir) and os.path.exists(os.path.join(ct2_dir, "model.bin"))):
                try:
                    logger.info(f"[*] Converting {model_id} to CTranslate2 INT8 in {ct2_dir}...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    converter = TransformersConverter(model_id, low_cpu_mem_usage=True)
                    converter.convert(ct2_dir, quantization="int8", force=True)
                    logger.info(f"[+] Converted {model_id} to CT2 INT8 successfully.")
                except Exception as e:
                    logger.error(f"[-] CT2 conversion failed for {model_id}: {e}")
                    return None

            try:
                ct2_translator = ctranslate2.Translator(
                    ct2_dir,
                    device=self.device,
                    compute_type="int8",
                    inter_threads=1,
                    intra_threads=4,
                )

                # Bounded LRU eviction
                if len(self.ct2_models) >= self.max_cached_models:
                    _evicted_pair, (evicted_trans, _) = self.ct2_models.popitem(last=False)
                    del evicted_trans

                self.ct2_models[pair] = (ct2_translator, tokenizer)
                return ct2_translator, tokenizer
            except Exception as e:
                logger.warning(f"[-] Failed to instantiate CTranslate2 Translator for {pair}: {e}")
                return None

    def _load_model(self, src: str, tgt: str) -> tuple[MarianMTModel, MarianTokenizer] | None:
        """Fallback PyTorch loader with bounded LRU eviction."""
        pair = (src, tgt)
        with self._lock:
            if pair in self.models:
                self.models.move_to_end(pair)
                return self.models[pair]

            model_id = self._get_model_id(src, tgt)
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_id, local_files_only=True)
                model = MarianMTModel.from_pretrained(model_id, local_files_only=True).to(self.device)
            except Exception:
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_id)
                    model = MarianMTModel.from_pretrained(model_id).to(self.device)
                except Exception as e:
                    logger.warning(f"[-] Could not load Marian PyTorch model for {model_id}: {e}")
                    return None

            model.eval()

            if len(self.models) >= self.max_cached_models:
                _evicted_pair, (evicted_model, _) = self.models.popitem(last=False)
                del evicted_model

            self.models[pair] = (model, tokenizer)
            return model, tokenizer

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "en",
    ) -> TranslationResult:
        t0 = time.perf_counter()
        clean_text = text.strip()

        if not clean_text or source_lang.lower() == target_lang.lower():
            return TranslationResult(
                original_text=clean_text,
                translated_text=clean_text,
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        src = source_lang.lower()
        tgt = target_lang.lower()

        # 1. Try ultra-fast CTranslate2 INT8 backend (<50ms)
        ct2_loaded = self._load_ct2_model(src, tgt)
        if ct2_loaded is not None:
            ct2_trans, tokenizer = ct2_loaded
            try:
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(clean_text))
                batch_res = ct2_trans.translate_batch([tokens])
                output_tokens = batch_res[0].hypotheses[0]
                translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens)).strip()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return TranslationResult(
                    original_text=clean_text,
                    translated_text=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    latency_ms=round(latency_ms, 2),
                    is_success=True,
                )
            except Exception as e:
                logger.warning(f"[-] CTranslate2 translate failed for {src}->{tgt}, trying PyTorch fallback: {e}")

        # 2. PyTorch Fallback
        loaded = self._load_model(src, tgt)
        if loaded is None:
            return TranslationResult(
                original_text=clean_text,
                translated_text=f"[{source_lang.upper()}]: {clean_text}",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                is_success=False,
                error_message=f"No translation model found for {source_lang}->{target_lang}",
            )

        model, tokenizer = loaded
        try:
            with torch.no_grad():
                inputs = tokenizer([clean_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                gen_tokens = model.generate(**inputs, max_length=128)
                translated = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return TranslationResult(
                original_text=clean_text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=round(latency_ms, 2),
                is_success=True,
            )
        except Exception as e:
            logger.error(f"[-] PyTorch Marian generation failed for {src}->{tgt}: {e}")
            return TranslationResult(
                original_text=clean_text,
                translated_text=f"[{source_lang.upper()}]: {clean_text}",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                is_success=False,
                error_message=str(e),
            )
