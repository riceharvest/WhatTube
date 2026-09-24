"""Voice Activity Detection and Energy Gate."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Union, Optional

class EnergyAndSileroVAD:
    """Combines sub-millisecond RMS energy filter with Silero VAD v6."""

    def __init__(
        self,
        model_path: Union[str, Path],
        energy_threshold: float = 0.005,
        vad_threshold: float = 0.35,
    ):
        self.energy_threshold = energy_threshold
        self.vad_threshold = vad_threshold

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.reset_states()

    def reset_states(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)

    def calculate_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio**2)))

    def is_speech(self, audio: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if audio window contains active speech.
        Returns: (is_speech: bool, rms: float, max_vad_prob: float)
        """
        rms = self.calculate_rms(audio)
        if rms < self.energy_threshold:
            return False, rms, 0.0

        # Silero VAD evaluation (chunked in 576-sample windows / 36ms at 16kHz)
        self.reset_states()
        window_size = 576
        num_windows = len(audio) // window_size
        if num_windows == 0:
            return False, rms, 0.0

        chunks = audio[: num_windows * window_size].reshape(num_windows, window_size)
        probs, self._h, self._c = self.session.run(
            ["speech_probs", "hn", "cn"],
            {"input": chunks, "h": self._h, "c": self._c},
        )
        max_prob = float(np.max(probs))
        return (max_prob >= self.vad_threshold), rms, max_prob

    def find_split_point(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_split_sec: float = 1.5,
        max_split_sec: float = 4.0,
    ) -> Optional[float]:
        """Find natural acoustic breath pause dip (p < 0.30) to split mixed or multi-utterance bursts."""
        window_size = 576
        num_windows = len(audio) // window_size
        if num_windows < int(min_split_sec * sample_rate / window_size):
            return None

        chunks = audio[: num_windows * window_size].reshape(num_windows, window_size)
        self.reset_states()
        probs, _, _ = self.session.run(
            ["speech_probs", "hn", "cn"],
            {"input": chunks, "h": self._h, "c": self._c},
        )

        start_idx = int(min_split_sec * sample_rate / window_size)
        end_idx = min(int(max_split_sec * sample_rate / window_size), num_windows - 5)

        best_idx = None
        min_prob = 1.0
        for i in range(start_idx, end_idx):
            if probs[i] < 0.30 and probs[i] < min_prob:
                min_prob = probs[i]
                best_idx = i

        if best_idx is not None:
            return float(best_idx * window_size / sample_rate)
        return None
