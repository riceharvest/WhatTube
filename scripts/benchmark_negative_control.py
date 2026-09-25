"""English-Only Negative Control Benchmark.

Evaluates 1.0 hour of diverse English vlog audio and noisy English mixtures:
- Clear host monologues
- Host + loud street noise / traffic
- Host + crowd murmur / cafe chatter
- Host with strong accents & foreign place/person names (Tokyo, Munich, Medellín, Cairo)
- English background chatter

Measures:
Host False Caption Rate = emitted captions caused by English host / hour
"""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_vlog_suite import run_vlog_benchmark
from whattube.config import default_config
from whattube.lid import WhisperTinyLID
from whattube.translation.marian import MarianTranslator
from whattube.vad import EnergyAndSileroVAD

BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"
OUTPUT_DIR = "/mnt/ssd/scratch_benchmark/synthetic/negative"

NEGATIVE_SEGMENTS = [
    # 1. Munich vlog: Clear host speech + German place/food names + restaurant noise
    {"file": f"{BENCHMARK_DIR}/germany_munich_nrf576J4Lg8_16k.wav", "start": 30.0, "dur": 900.0, "name": "Munich_Host_Names_Restaurant"},
    # 2. Tokyo vlog: English host walking through Shibuya/trains + city noise + Japanese station names
    {"file": f"{BENCHMARK_DIR}/japan_tokyo_iszTT9U4OA8.wav", "start": 60.0, "dur": 900.0, "name": "Tokyo_Host_Street_Station_Names"},
    # 3. Colombia vlog: English host with Spanish place names & outdoor walking
    {"file": f"{BENCHMARK_DIR}/colombia_medellin_KrSpBWnBelE_16k.wav", "start": 30.0, "dur": 900.0, "name": "Medellin_Host_Place_Names_Outdoor"},
    # 4. Paris vlog: English host speaking with French bistro background music & murmur
    {"file": f"{BENCHMARK_DIR}/france_paris_h13cCP9wCpw_16k.wav", "start": 60.0, "dur": 900.0, "name": "Paris_Host_Music_Murmur"},
]


def run_negative_control():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== ENGLISH-ONLY NEGATIVE CONTROL BENCHMARK (1.0 HOUR) ===")
    print("==================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[*] Pre-loading pipeline models...")
    vad = EnergyAndSileroVAD(default_config.silero_vad_onnx)
    lid = WhisperTinyLID(default_config.encoder_onnx, default_config.decoder_onnx)
    translator = MarianTranslator(device="cpu", use_ct2=True)
    print("[+] Models ready.")

    total_eval_sec = 0.0
    total_false_captions = 0
    total_triggered_bursts = 0
    total_discarded_monologues = 0
    segment_results = []
    t0_suite = time.time()

    for idx, seg in enumerate(NEGATIVE_SEGMENTS, 1):
        wav_path = seg["file"]
        if not os.path.exists(wav_path):
            print(f"[!] Warning: Audio file not found: {wav_path}")
            continue

        print(f"\n[{idx}/4] Evaluating Negative Segment: {seg['name']}")
        print(f"  File: {os.path.basename(wav_path)} [{seg['start']:.1f}s - {seg['start']+seg['dur']:.1f}s] ({seg['dur']/60:.1f} min)")

        t0_seg = time.time()
        summary = run_vlog_benchmark(
            audio_path=wav_path,
            title_hints=["en"],  # Title language is English
            target_lang="en",
            start_sec=seg["start"],
            max_duration_sec=seg["dur"],
            vad=vad,
            lid=lid,
            translator=translator,
        )
        elapsed_seg = time.time() - t0_seg

        total_eval_sec += summary["evaluated_duration_sec"]
        total_false_captions += summary["emitted_subtitles"]
        total_triggered_bursts += summary["triggered_bursts"]
        total_discarded_monologues += summary["discarded_english_or_daemon"]

        segment_results.append({
            "segment_name": seg["name"],
            "duration_sec": summary["evaluated_duration_sec"],
            "triggered_bursts": summary["triggered_bursts"],
            "false_captions": summary["emitted_subtitles"],
            "discarded_monologues": summary["discarded_english_or_daemon"],
            "emitted_items": summary["results"],
            "elapsed_sec": round(elapsed_seg, 2),
        })

    total_hours = total_eval_sec / 3600.0
    false_caption_rate_per_hour = total_false_captions / max(total_hours, 1e-6)

    negative_summary = {
        "benchmark": "english_only_negative_control",
        "total_evaluated_duration_sec": round(total_eval_sec, 2),
        "total_evaluated_duration_hours": round(total_hours, 2),
        "total_benchmark_run_time_sec": round(time.time() - t0_suite, 2),
        "total_triggered_bursts": total_triggered_bursts,
        "total_discarded_english_monologues": total_discarded_monologues,
        "total_false_captions": total_false_captions,
        "host_false_caption_rate_per_hour": round(false_caption_rate_per_hour, 2),
        "segments": segment_results,
    }

    out_file = os.path.join(OUTPUT_DIR, "negative_control_summary.json")
    with open(out_file, "w") as f:
        json.dump(negative_summary, f, indent=2)

    print("\n==================================================================")
    print("=== NEGATIVE CONTROL BENCHMARK COMPLETE ===")
    print("==================================================================")
    print(f"Total Evaluated Duration:  {total_hours:.2f} hours ({total_eval_sec:.1f}s)")
    print(f"Total Triggered Bursts:    {total_triggered_bursts}")
    print(f"Discarded Monologues:      {total_discarded_monologues}")
    print(f"Emitted False Captions:    {total_false_captions}")
    print(f"Host False Caption Rate:   {false_caption_rate_per_hour:.2f} captions / hour")
    print(f"Summary Saved To:          {out_file}")
    print("==================================================================\n")


if __name__ == "__main__":
    run_negative_control()
