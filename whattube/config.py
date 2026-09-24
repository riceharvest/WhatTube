"""Configuration defaults for WhatTube streaming server and detection pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    # Audio capture
    sample_rate: int = 16000
    channels: int = 1
    analysis_window_sec: float = 3.0
    stride_sec: float = 1.0
    buffer_capacity_sec: float = 30.0

    # VAD & Energy filters
    energy_threshold_rms: float = 0.005
    vad_threshold: float = 0.35

    # Spoken Language ID (LID) thresholds
    lid_english_safe: float = 0.75       # p_en >= 0.75: Confident English (ignore)
    lid_suspicious: float = 0.55         # p_en < 0.55: Suspicious chatter overlap
    lid_immediate: float = 0.20          # p_en < 0.20: High-confidence non-English trigger
    consecutive_suspicious_req: int = 2  # 2 of 3 windows under 0.55 to open event

    # Dynamic Event Aggregator timing
    pre_roll_sec: float = 0.5            # Pre-roll padding before first suspicious window
    post_roll_sec: float = 0.5           # Post-roll padding after event end
    close_hangover_sec: float = 0.8      # Delay after clean English before closing
    min_event_sec: float = 3.0           # Minimum burst duration for linguistic coherence
    max_event_sec: float = 5.0           # Optimal Whisper Turbo context window (prevents host domination)

    # Service endpoints
    asr_endpoint: str = os.getenv("WHATTUBE_ASR_ENDPOINT", "http://127.0.0.1:8766/transcribe")
    ws_host: str = os.getenv("WHATTUBE_WS_HOST", "0.0.0.0")
    ws_port: int = int(os.getenv("WHATTUBE_WS_PORT", "8765"))
    target_language: str = os.getenv("WHATTUBE_TARGET_LANG", "en")

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = base_dir / "models"
    encoder_onnx: Path = models_dir / "tiny-encoder.int8.onnx"
    decoder_onnx: Path = models_dir / "tiny-decoder.int8.onnx"
    tokens_txt: Path = models_dir / "tiny-tokens.txt"
    silero_vad_onnx: Path = models_dir / "silero_vad_v6.onnx"
    log_file: Path = base_dir / "events.jsonl"

default_config = Config()
