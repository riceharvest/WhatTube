"""MarianMT multilingual translation backend with on-demand model routing."""

import time
from typing import Dict, Tuple, Optional
import torch
from transformers import MarianMTModel, MarianTokenizer
from whattube.translation.base import BaseTranslator, TranslationResult

class MarianTranslator(BaseTranslator):
    """Translates text on CPU using Helsinki-NLP/opus-mt models."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        # Cache of (src, tgt) -> (model, tokenizer)
        self.models: Dict[Tuple[str, str], Tuple[MarianMTModel, MarianTokenizer]] = {}
        import threading
        self._lock = threading.Lock()

    def _get_model_id(self, src: str, tgt: str) -> str:
        # Common language code normalizations
        src_map = {"jw": "id", "ms": "id"} # Javanese/Malay often map well to id or direct
        src_clean = src_map.get(src, src)
        return f"Helsinki-NLP/opus-mt-{src_clean}-{tgt}"

    def _load_model(self, src: str, tgt: str) -> Optional[Tuple[MarianMTModel, MarianTokenizer]]:
        pair = (src, tgt)
        with self._lock:
            if pair in self.models:
                return self.models[pair]

            model_id = self._get_model_id(src, tgt)
            try:
                # First try loading strictly from local cache to avoid huggingface network roundtrip
                tokenizer = MarianTokenizer.from_pretrained(model_id, local_files_only=True)
                model = MarianMTModel.from_pretrained(model_id, local_files_only=True).to(self.device)
            except Exception:
                try:
                    print(f"[*] Downloading translation model to cache: {model_id}")
                    tokenizer = MarianTokenizer.from_pretrained(model_id)
                    model = MarianMTModel.from_pretrained(model_id).to(self.device)
                except Exception as e:
                    print(f"[-] Could not load Marian model for {pair}: {e}")
                    return None

            model.eval()
            self.models[pair] = (model, tokenizer)
            return model, tokenizer

    def normalize_slang(self, text: str, lang: str) -> str:
        """Lightweight regex normalizer for common conversational/internet slang."""
        import re
        if lang.lower() == "id":
            # Indonesian TikTok live / street colloquialisms
            text = re.sub(r"\b(tetep|tag|ketek)\s+layarnya\b", "tekan layarnya", text, flags=re.IGNORECASE)
            text = re.sub(r"\b(ngeplok|nge-flop)\b", "nge-vlog", text, flags=re.IGNORECASE)
            text = re.sub(r"\bsampai\s+(petiak|berjalan)\b", "sampai pecah", text, flags=re.IGNORECASE)
        return text

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "en",
    ) -> TranslationResult:
        t0 = time.perf_counter()
        clean_text = self.normalize_slang(text.strip(), source_lang)

        if not clean_text or source_lang.lower() == target_lang.lower():
            return TranslationResult(
                original_text=clean_text,
                translated_text=clean_text,
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        loaded = self._load_model(source_lang.lower(), target_lang.lower())
        if loaded is None:
            # Fallback if specific pair model does not exist
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
