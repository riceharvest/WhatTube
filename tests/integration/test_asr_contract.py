"""Contract and validation tests for resident ASR daemon."""

import io
import json
import urllib.request

import numpy as np
import pytest
import soundfile as sf

DAEMON_URL = "http://127.0.0.1:8766"

def is_daemon_running():
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/health", timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ready"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False

@pytest.mark.skipif(not is_daemon_running(), reason="ASR daemon not running")
def test_asr_health_contract():
    req = urllib.request.Request(f"{DAEMON_URL}/health")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["status"] == "ready"
        assert "device" in data
        assert "model" in data

@pytest.mark.skipif(not is_daemon_running(), reason="ASR daemon not running")
def test_asr_transcribe_contract():
    # Generate 1.5 seconds of 16kHz audio
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 24000))).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, 16000, format="WAV")
    wav_bytes = bio.getvalue()

    req = urllib.request.Request(
        f"{DAEMON_URL}/transcribe",
        data=wav_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        assert "language" in res
        assert "language_prob" in res
        assert "text" in res
        assert "latency_ms" in res
        assert "is_discarded" in res
        assert isinstance(res["language_prob"], float)

@pytest.mark.skipif(not is_daemon_running(), reason="ASR daemon not running")
def test_asr_resamples_non_16k_wav():
    # Generate 44.1kHz audio
    audio_44k = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 66150))).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio_44k, 44100, format="WAV")
    wav_bytes = bio.getvalue()

    req = urllib.request.Request(
        f"{DAEMON_URL}/transcribe",
        data=wav_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode())
        assert "language" in res

@pytest.mark.skipif(not is_daemon_running(), reason="ASR daemon not running")
def test_asr_rejects_malformed_input():
    # Send unaligned arbitrary binary garbage that is neither WAV nor valid float32
    garbage = b"\x01\x02\x03"
    req = urllib.request.Request(
        f"{DAEMON_URL}/transcribe",
        data=garbage,
        headers={"Content-Type": "application/octet-stream"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400

from pathlib import Path

FIXTURE_WAV = Path("/mnt/ssd/hermes/cache/scratch/seal-test-P13/P13mMiIL_2I_full_16k.wav")

@pytest.mark.skipif(not is_daemon_running(), reason="ASR daemon not running")
@pytest.mark.skipif(not FIXTURE_WAV.exists(), reason="Fixture audio not found")
def test_asr_real_speech_language_prob():
    # 831.0s to 834.0s is known Indonesian street vendor speech
    audio, _sr = sf.read(str(FIXTURE_WAV), start=int(831.0 * 16000), stop=int(834.0 * 16000), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    bio = io.BytesIO()
    sf.write(bio, audio, 16000, format="WAV")
    wav_bytes = bio.getvalue()

    req = urllib.request.Request(
        f"{DAEMON_URL}/transcribe",
        data=wav_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        assert res["language"] == "id"
        assert 0.0 < res["language_prob"] < 1.0
        assert not res["is_discarded"]

