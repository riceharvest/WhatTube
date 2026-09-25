import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def get_or_create_auth_token() -> str:
    """Retrieve existing auth token from env or ~/.whattube/token, or generate one."""
    env_token = os.getenv("WHATTUBE_AUTH_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    
    token_dir = Path.home() / ".whattube"
    token_file = token_dir / "token"
    if token_file.exists():
        try:
            tok = token_file.read_text().strip()
            if tok:
                return tok
        except (OSError, UnicodeDecodeError):
            pass

    try:
        token_dir.mkdir(parents=True, exist_ok=True)
        new_tok = secrets.token_hex(16)
        token_file.write_text(new_tok)
        token_file.chmod(0o600)
        return new_tok
    except OSError:
        return "whattube-default-token"

@dataclass
class Config:
    # Audio capture
    sample_rate: int = 16000
    channels: int = 1
    analysis_window_sec: float = 1.5
    stride_sec: float = 0.5
    buffer_capacity_sec: float = 30.0

    # VAD & Energy filters
    energy_threshold_rms: float = 0.005
    vad_threshold: float = 0.35

    # Spoken Language ID (LID) thresholds
    lid_english_safe: float = 0.75       # p_en >= 0.75: Confident English (ignore)
    lid_suspicious: float = 0.55         # p_en < 0.55: Suspicious chatter overlap
    lid_immediate: float = 0.25          # p_en < 0.25: High-confidence non-English trigger
    consecutive_suspicious_req: int = 2  # 2 of 3 windows under 0.55 to open event

    # Dynamic Event Aggregator timing
    pre_roll_sec: float = 0.2            # Focused pre-roll avoids swallowing host monologues
    post_roll_sec: float = 0.3           # Post-roll padding after event end
    close_hangover_sec: float = 0.6      # Delay after clean English before closing
    min_event_sec: float = 1.2           # Focused minimum burst for brief chatter
    max_event_sec: float = 6.0           # Maximum adaptive burst duration (prevents host context domination)

    # Service endpoints & security
    asr_endpoint: str = os.getenv("WHATTUBE_ASR_ENDPOINT", "http://127.0.0.1:8766/transcribe")
    ws_host: str = os.getenv("WHATTUBE_WS_HOST", "127.0.0.1")  # Safe localhost default
    ws_port: int = int(os.getenv("WHATTUBE_WS_PORT", "8765"))
    target_language: str = os.getenv("WHATTUBE_TARGET_LANG", "en")
    auth_token: str = field(default_factory=get_or_create_auth_token)
    enable_transcript_log: bool = os.getenv("WHATTUBE_ENABLE_LOG", "0") in ("1", "true", "True")  # Opt-in for privacy

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = base_dir / "models"
    encoder_onnx: Path = models_dir / "tiny-encoder.int8.onnx"
    decoder_onnx: Path = models_dir / "tiny-decoder.int8.onnx"
    tokens_txt: Path = models_dir / "tiny-tokens.txt"
    silero_vad_onnx: Path = models_dir / "silero_vad_v6.onnx"
    log_file: Path = base_dir / "events.jsonl"

default_config = Config()
