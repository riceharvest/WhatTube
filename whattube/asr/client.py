"""Client for communicating with the Resident GPU ASR Daemon."""

import io
import time
import httpx
import numpy as np
import soundfile as sf
from typing import Optional
from whattube.asr.base import ASRBackend, ASRResult

class ResidentASRClient(ASRBackend):
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8766/transcribe", timeout: float = 10.0):
        self.endpoint_url = endpoint_url
        self.health_url = endpoint_url.replace("/transcribe", "/health")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            resp = self.client.get(self.health_url)
            return resp.status_code == 200 and resp.json().get("status") == "ready"
        except Exception:
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        t0 = time.perf_counter()

        # Pack to memory WAV container
        bio = io.BytesIO()
        sf.write(bio, audio, sample_rate, format="WAV", subtype="FLOAT")
        bio.seek(0)
        wav_bytes = bio.read()

        try:
            resp = self.client.post(
                self.endpoint_url,
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            )
            if resp.status_code != 200:
                return ASRResult(
                    text="",
                    language="error",
                    language_prob=0.0,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    is_discarded=True,
                    discard_reason=f"ASR daemon returned HTTP {resp.status_code}: {resp.text}",
                )

            data = resp.json()
            return ASRResult(
                text=data.get("text", ""),
                language=data.get("language", "unknown"),
                language_prob=data.get("language_prob", 1.0),
                latency_ms=data.get("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2)),
                is_discarded=data.get("is_discarded", False),
                discard_reason=data.get("discard_reason", ""),
            )
        except Exception as e:
            return ASRResult(
                text="",
                language="error",
                language_prob=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                is_discarded=True,
                discard_reason=f"Failed to communicate with ASR daemon: {str(e)}",
            )
