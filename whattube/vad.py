"""Voice Activity Detection and Energy Gate."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class EnergyAndSileroVAD:
    """Combines sub-millisecond RMS energy filter with Silero VAD v6."""

    def __init__(
        self,
        model_path: str | Path,
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

    def is_speech(self, audio: np.ndarray) -> tuple[bool, float, float]:
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
        min_subslice_sec: float = 1.0,
    ) -> float | None:
        """Find natural acoustic breath pause dip (sustained valley >= 144ms, avg p < 0.25).
        Enforces that both resulting sub-slices are at least min_subslice_sec long.
        """
        window_size = 576
        num_windows = len(audio) // window_size
        min_sub_windows = int(min_subslice_sec * sample_rate / window_size)

        start_idx = min_sub_windows
        end_idx = num_windows - min_sub_windows
        if end_idx <= start_idx:
            return None

        chunks = audio[: num_windows * window_size].reshape(num_windows, window_size)
        self.reset_states()
        probs, _, _ = self.session.run(
            ["speech_probs", "hn", "cn"],
            {"input": chunks, "h": self._h, "c": self._c},
        )

        # Look for a sustained valley of low speech probability (4 consecutive windows ~144ms)
        valley_len = 4
        best_idx = None
        min_avg_prob = 0.25

        for i in range(start_idx, end_idx - valley_len):
            avg_prob = float(np.mean(probs[i : i + valley_len]))
            if avg_prob < min_avg_prob:
                min_avg_prob = avg_prob
                best_idx = i + valley_len // 2

        if best_idx is not None:
            return float(best_idx * window_size / sample_rate)
        return None
