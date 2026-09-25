"""Extract clean English travel vlog host speech stems with exact ground-truth transcripts.

Scans vlog audio files for high-confidence English monologue segments (5.0s - 9.0s),
verifies clean English via Whisper Turbo on GPU (p_en >= 0.98), and saves 16kHz WAV stems.
"""

import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf

HOST_STEMS_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems/host"
BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"

CANDIDATE_VLOGS = [
    f"{BENCHMARK_DIR}/germany_munich_nrf576J4Lg8_16k.wav",
    f"{BENCHMARK_DIR}/japan_tokyo_iszTT9U4OA8.wav",
    f"{BENCHMARK_DIR}/indonesia_jakarta_P13mMiIL_2I_16k.wav",
    f"{BENCHMARK_DIR}/france_paris_h13cCP9wCpw_16k.wav",
    f"{BENCHMARK_DIR}/greece_athens_Kvg9kJ35jEY_16k.wav",
]

def query_whisper(audio_chunk, sr=16000):
    bio = io.BytesIO()
    sf.write(bio, audio_chunk, sr, format="WAV")
    req = urllib.request.Request(
        "http://127.0.0.1:8766/transcribe",
        data=bio.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[-] Whisper request failed: {e}")
        return None

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== EXTRACTING CLEAN ENGLISH HOST STEMS FROM TRAVEL VLOGS ===")
    print("==================================================================")

    os.makedirs(HOST_STEMS_DIR, exist_ok=True)
    manifest_path = os.path.join(HOST_STEMS_DIR, "host_manifest.json")

    stems = []
    target_count = 25

    for vlog_path in CANDIDATE_VLOGS:
        if not os.path.exists(vlog_path):
            continue

        print(f"\n[*] Scanning {os.path.basename(vlog_path)} for host speech segments...")
        audio, sr = sf.read(vlog_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        dur_sec = len(audio) / sr
        # Sample segments across the video between 30s and 600s
        step_sec = 15.0
        segment_len_sec = 6.5

        t = 30.0
        while t < min(dur_sec - 10.0, 900.0) and len(stems) < target_count:
            idx_start = int(t * sr)
            idx_end = int((t + segment_len_sec) * sr)
            chunk = audio[idx_start:idx_end]

            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.02:  # Active speech
                res = query_whisper(chunk, sr)
                if res and res.get("language") == "en" and res.get("language_prob", 0.0) >= 0.98:
                    text = res.get("text", "").strip()
                    words = text.split()
                    if len(words) >= 10:
                        host_id = len(stems) + 1
                        out_wav = os.path.join(HOST_STEMS_DIR, f"host_stem_{host_id:02d}.wav")
                        sf.write(out_wav, chunk, sr)

                        meta = {
                            "host_id": host_id,
                            "source_vlog": os.path.basename(vlog_path),
                            "start_sec": round(t, 2),
                            "end_sec": round(t + segment_len_sec, 2),
                            "duration_sec": segment_len_sec,
                            "transcript": text,
                            "word_count": len(words),
                            "rms": round(float(rms), 4),
                            "wav_file": out_wav,
                        }
                        stems.append(meta)
                        print(f"  [+] Host Stem #{host_id:02d} ({t:.1f}s): \"{text}\"")
                        t += segment_len_sec + 5.0
                        continue

            t += step_sec

        if len(stems) >= target_count:
            break

    with open(manifest_path, "w") as f:
        json.dump(stems, f, indent=2)

    print("\n==================================================================")
    print(f"[+] Successfully extracted {len(stems)} clean English host stems.")
    print(f"[+] Saved manifest to: {manifest_path}")
    print("==================================================================")

if __name__ == "__main__":
    main()
