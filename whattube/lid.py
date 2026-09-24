"""Spoken Language Identification (LID) using quantized Whisper Tiny ONNX."""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Union
from transformers import WhisperFeatureExtractor

LANG_TOKENS = [
    50317, 50296, 50261, 50276, 50337, 50260, 50271, 50279, 50298, 50319,
    50301, 50272, 50268, 50352, 50333, 50322, 50330, 50347, 50341, 50303,
    50320, 50267, 50314, 50344, 50331, 50321, 50273, 50350, 50305, 50357,
    50355, 50286, 50353, 50309, 50262, 50270, 50292, 50312, 50289, 50325,
    50274, 50285, 50328, 50284, 50287, 50311, 50315, 50283, 50345, 50346,
    50332, 50338, 50313, 50354, 50299, 50295, 50297, 50281, 50349, 50327,
    50335, 50306, 50282, 50326, 50342, 50351, 50259, 50348, 50269, 50264,
    50304, 50291, 50334, 50275, 50280, 50340, 50288, 50310, 50316, 50323,
    50336, 50324, 50265, 50308, 50294, 50356, 50277, 50302, 50293, 50278,
    50329, 50343, 50318, 50290, 50263, 50266, 50300, 50339, 50307,
]

LANG_CODES = [
    "sq", "ml", "de", "hi", "uz", "zh", "nl", "he", "sk", "gl",
    "lv", "ar", "tr", "haw", "gu", "si", "be", "bo", "tk", "sr",
    "mr", "pt", "mn", "sa", "tg", "pa", "sv", "as", "sl", "su",
    "ba", "hu", "ln", "br", "es", "ca", "bg", "hy", "th", "yo",
    "it", "da", "oc", "ro", "ta", "is", "bs", "cs", "lb", "my",
    "sd", "fo", "ne", "ha", "te", "mi", "cy", "el", "mg", "af",
    "yi", "kn", "ms", "so", "nn", "tt", "en", "tl", "pl", "ko",
    "az", "hr", "am", "id", "uk", "ps", "no", "eu", "kk", "km",
    "lo", "sn", "fr", "mk", "la", "jw", "fi", "bn", "lt", "vi",
    "ka", "mt", "sw", "ur", "ru", "ja", "fa", "ht", "et",
]

EN_TOKEN_IDX = LANG_TOKENS.index(50259)

class WhisperTinyLID:
    """Sub-55ms CPU Spoken Language ID for filtering confident English."""

    def __init__(
        self,
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        lid_suspicious: float = 0.55,
        lid_immediate: float = 0.20,
        num_threads: int = 4,
    ):
        self.lid_suspicious = lid_suspicious
        self.lid_immediate = lid_immediate

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.enc = ort.InferenceSession(str(encoder_path), opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(decoder_path), opts, providers=["CPUExecutionProvider"])
        self.extractor = WhisperFeatureExtractor(feature_size=80)

        # Pre-allocated tensors
        self.tokens_sot = np.array([[50258]], dtype=np.int64)
        self.k_cache = np.zeros((4, 1, 448, 384), dtype=np.float32)
        self.v_cache = np.zeros((4, 1, 448, 384), dtype=np.float32)
        self.offset = np.array([0], dtype=np.int64)

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Predict language probabilities from audio chunk (typically 3s).
        Returns classification metrics and suspicion status.
        """
        t0 = time.perf_counter()

        # Compute Mel spectrogram and truncate to 1300 frames (~3s)
        mel = self.extractor(audio, sampling_rate=sample_rate, return_tensors="np").input_features
        mel_input = mel[:, :, :1300]

        enc_out = self.enc.run(None, {"mel": mel_input})
        cross_k, cross_v = enc_out[0], enc_out[1]

        dec_out = self.dec.run(
            None,
            {
                "tokens": self.tokens_sot,
                "in_n_layer_self_k_cache": self.k_cache,
                "in_n_layer_self_v_cache": self.v_cache,
                "n_layer_cross_k": cross_k,
                "n_layer_cross_v": cross_v,
                "offset": self.offset,
            },
        )
        logits = dec_out[0][0, 0]
        sub_logits = logits[LANG_TOKENS]
        sub_exp = np.exp(sub_logits - np.max(sub_logits))
        probs = sub_exp / np.sum(sub_exp)

        p_en = float(probs[EN_TOKEN_IDX])
        top_idx = int(np.argmax(probs))
        top_lang = LANG_CODES[top_idx]
        top_prob = float(probs[top_idx])

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        is_suspicious = p_en < self.lid_suspicious
        is_immediate = p_en < self.lid_immediate

        return {
            "top_lang": top_lang,
            "top_prob": top_prob,
            "p_en": p_en,
            "is_suspicious": is_suspicious,
            "is_immediate": is_immediate,
            "latency_ms": latency_ms,
        }
