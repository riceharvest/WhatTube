"""Extract clean ambient background noise stems (street, market, traffic) from travel vlogs.

Finds audio segments where VAD detects no speech (vad_prob < 0.1, RMS 0.005-0.03)
to serve as realistic background acoustic beds for synthetic mixtures.
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

from whattube.config import default_config
from whattube.vad import EnergyAndSileroVAD

AMBIENT_STEMS_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems/ambient"
BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"

VLOGS = [
    f"{BENCHMARK_DIR}/thailand_bangkok_dUEMXUMwiAw.wav",
    f"{BENCHMARK_DIR}/egypt_cairo_pIwCc7RuaQU_16k.wav",
    f"{BENCHMARK_DIR}/vietnam_hanoi_Feu_DN_UleE_16k.wav",
    f"{BENCHMARK_DIR}/turkey_istanbul_QoG1_v1xagU_16k.wav",
]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== EXTRACTING AMBIENT NOISE STEMS FROM TRAVEL VLOGS ===")
    print("==================================================================")

    os.makedirs(AMBIENT_STEMS_DIR, exist_ok=True)
    vad = EnergyAndSileroVAD(default_config.silero_vad_onnx)

    stems = []
    target_count = 10
    seg_len_sec = 8.0

    for vlog_path in VLOGS:
        if not os.path.exists(vlog_path):
            continue

        audio, sr = sf.read(vlog_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        dur_sec = len(audio) / sr
        t = 10.0
        while t < min(dur_sec - 15.0, 1200.0) and len(stems) < target_count:
            idx_start = int(t * sr)
            idx_end = int((t + seg_len_sec) * sr)
            chunk = audio[idx_start:idx_end]

            # Check if non-speech or ambient bed
            is_speech, rms, vad_prob = vad.is_speech(chunk)
            if not is_speech and 0.003 < rms < 0.08 and vad_prob < 0.25:
                amb_id = len(stems) + 1
                out_wav = os.path.join(AMBIENT_STEMS_DIR, f"ambient_{amb_id:02d}.wav")
                sf.write(out_wav, chunk, sr)

                meta = {
                    "ambient_id": amb_id,
                    "source_vlog": os.path.basename(vlog_path),
                    "start_sec": round(t, 2),
                    "duration_sec": seg_len_sec,
                    "rms": round(float(rms), 4),
                    "vad_prob": round(float(vad_prob), 4),
                    "wav_file": out_wav,
                }
                stems.append(meta)
                print(f"  [+] Ambient Stem #{amb_id:02d} ({t:.1f}s) from {os.path.basename(vlog_path)} (RMS={rms:.4f}, VAD={vad_prob:.3f})")
                t += 15.0
            else:
                t += 3.0

        if len(stems) >= target_count:
            break

    manifest_path = os.path.join(AMBIENT_STEMS_DIR, "ambient_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(stems, f, indent=2)

    print(f"\n[+] Extracted {len(stems)} ambient stems saved to {manifest_path}")

if __name__ == "__main__":
    main()
