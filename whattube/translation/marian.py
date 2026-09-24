"""MarianMT multilingual translation backend with CTranslate2 INT8 acceleration and on-demand model routing."""

import os
import time
import threading
from typing import Dict, Tuple, Optional
import torch
from transformers import MarianMTModel, MarianTokenizer
from whattube.translation.base import BaseTranslator, TranslationResult

try:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    HAS_CTRANSLATE2 = True
except ImportError:
    HAS_CTRANSLATE2 = False

class MarianTranslator(BaseTranslator):
    """Translates text on CPU using CTranslate2 INT8 accelerated Helsinki-NLP/opus-mt models,

    with automatic fallback to PyTorch MarianMTModel.
    """

    def __init__(self, device: str = "cpu", cache_dir: str = "models", use_ct2: bool = True):
        self.device = device
        self.cache_dir = cache_dir
        self.use_ct2 = use_ct2 and HAS_CTRANSLATE2
        # PyTorch cache: (src, tgt) -> (model, tokenizer)
        self.models: Dict[Tuple[str, str], Tuple[MarianMTModel, MarianTokenizer]] = {}
        # CTranslate2 cache: (src, tgt) -> (ct2_translator, tokenizer)
        self.ct2_models: Dict[Tuple[str, str], Tuple[any, MarianTokenizer]] = {}
        self._lock = threading.Lock()

    def _get_model_id(self, src: str, tgt: str) -> str:
        # Common language code normalizations
        src_map = {"jw": "id", "ms": "id"}  # Javanese/Malay often map well to id or direct
        src_clean = src_map.get(src, src)
        return f"Helsinki-NLP/opus-mt-{src_clean}-{tgt}"

    def _get_ct2_dir(self, src: str, tgt: str) -> str:
        src_clean = src.replace("-", "_")
        tgt_clean = tgt.replace("-", "_")
        return os.path.join(self.cache_dir, f"ct2_opus_mt_{src_clean}_{tgt_clean}")

    def _load_ct2_model(self, src: str, tgt: str) -> Optional[Tuple[any, MarianTokenizer]]:
        """Loads or converts a CTranslate2 INT8 model for ultra-low latency CPU translation."""
        if not self.use_ct2:
            return None

        pair = (src, tgt)
        ct2_dir = self._get_ct2_dir(src, tgt)
        model_id = self._get_model_id(src, tgt)

        with self._lock:
            if pair in self.ct2_models:
                return self.ct2_models[pair]

            try:
                tokenizer = MarianTokenizer.from_pretrained(model_id, local_files_only=True)
            except Exception:
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_id)
                except Exception as e:
                    print(f"[-] Could not load Marian tokenizer for {pair}: {e}")
                    return None

            # Check if CT2 model directory exists
            if not (os.path.isdir(ct2_dir) and os.path.exists(os.path.join(ct2_dir, "model.bin"))):
                try:
                    print(f"[*] Converting {model_id} to CTranslate2 INT8 in {ct2_dir}...")
                    os.makedirs(self.cache_dir, exist_ok=True)
                    converter = TransformersConverter(model_id, low_cpu_mem_usage=True)
                    converter.convert(ct2_dir, quantization="int8", force=True)
                    print(f"[+] Converted {model_id} to CT2 INT8 successfully.")
                except Exception as e:
                    print(f"[-] CT2 conversion failed for {model_id}: {e}")
                    return None

            try:
                ct2_translator = ctranslate2.Translator(
                    ct2_dir,
                    device=self.device,
                    compute_type="int8",
                    inter_threads=1,
                    intra_threads=4,
                )
                self.ct2_models[pair] = (ct2_translator, tokenizer)
                return ct2_translator, tokenizer
            except Exception as e:
                print(f"[-] Failed to instantiate CTranslate2 Translator: {e}")
                return None

    def _load_model(self, src: str, tgt: str) -> Optional[Tuple[MarianMTModel, MarianTokenizer]]:
        """Fallback PyTorch loader."""
        pair = (src, tgt)
        with self._lock:
            if pair in self.models:
                return self.models[pair]

            model_id = self._get_model_id(src, tgt)
            try:
                # First try loading strictly from local cache
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
                print(f"[-] CTranslate2 translate failed, falling back to PyTorch: {e}")

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
