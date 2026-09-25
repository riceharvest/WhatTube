"""Abstract ASR backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ASRResult:
    text: str
    language: str
    language_prob: float
    latency_ms: float
    is_discarded: bool = False
    discard_reason: str = ""

class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe audio chunk and detect primary spoken language."""
