"""Master benchmark runner for WhatTube across all 20 full-length travel vlogs (11.01 hours).

Evaluates the complete end-to-end pipeline:
- Stage 1 VAD + Whisper Tiny LID continuous streaming (CPU)
- Dynamic Event Aggregator (1.5s window / 0.5s stride, 0.2s/0.3s pre/post roll)
- Breath-pause acoustic splitting & sub-slice fallback
- Stage 2 Whisper Large-v3-Turbo resident daemon on Intel Arc Pro B70 GPU (xpu)
- True Step-0 Language Probability gating & fluent English monologue discard
- Stage 3 CTranslate2 INT8 Marian translation (CPU)
"""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_vlog_suite import run_vlog_benchmark
from whattube.config import default_config
from whattube.lid import WhisperTinyLID
from whattube.translation.marian import MarianTranslator
from whattube.vad import EnergyAndSileroVAD

BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"

CORPUS_20 = [
    {
        "name": "Egypt (Cairo)",
        "file": f"{BENCHMARK_DIR}/egypt_cairo_pIwCc7RuaQU_16k.wav",
        "hints": ["ar"],
        "target": "en",
    },
    {
        "name": "South Korea (Seoul)",
        "file": f"{BENCHMARK_DIR}/korea_seoul_GalLzn7Tyzo_16k.wav",
        "hints": ["ko"],
        "target": "en",
    },
    {
        "name": "Vietnam (Hanoi)",
        "file": f"{BENCHMARK_DIR}/vietnam_hanoi_Feu_DN_UleE_16k.wav",
        "hints": ["vi"],
        "target": "en",
    },
    {
        "name": "Thailand (Bangkok)",
        "file": f"{BENCHMARK_DIR}/thailand_bangkok_dUEMXUMwiAw.wav",
        "hints": ["th"],
        "target": "en",
    },
    {
        "name": "Japan (Tokyo)",
        "file": f"{BENCHMARK_DIR}/japan_tokyo_iszTT9U4OA8.wav",
        "hints": ["ja"],
        "target": "en",
    },
    {
        "name": "Bangladesh (Dhaka)",
        "file": f"{BENCHMARK_DIR}/bangladesh_dhaka_LCgvMgv6cCk.wav",
        "hints": ["bn", "hi"],
        "target": "en",
    },
    {
        "name": "India (Delhi)",
        "file": f"{BENCHMARK_DIR}/india_delhi_vQnydLXjUlE_16k.wav",
        "hints": ["hi", "ur"],
        "target": "en",
    },
    {
        "name": "Turkey (Istanbul)",
        "file": f"{BENCHMARK_DIR}/turkey_istanbul_QoG1_v1xagU_16k.wav",
        "hints": ["tr"],
        "target": "en",
    },
    {
        "name": "Brazil (Rio)",
        "file": f"{BENCHMARK_DIR}/brazil_rio_TlWDF3BfUdo_16k.wav",
        "hints": ["pt"],
        "target": "en",
    },
    {
        "name": "Taiwan (Taipei)",
        "file": f"{BENCHMARK_DIR}/taiwan_taipei_SSQdmrgWnLo_16k.wav",
        "hints": ["zh"],
        "target": "en",
    },
    {
        "name": "Philippines (Manila)",
        "file": f"{BENCHMARK_DIR}/philippines_manila_IiCfEzRvUIQ_16k.wav",
        "hints": ["tl"],
        "target": "en",
    },
    {
        "name": "Germany (Munich)",
        "file": f"{BENCHMARK_DIR}/germany_munich_nrf576J4Lg8_16k.wav",
        "hints": ["de"],
        "target": "en",
    },
    {
        "name": "Greece (Athens)",
        "file": f"{BENCHMARK_DIR}/greece_athens_Kvg9kJ35jEY_16k.wav",
        "hints": ["el"],
        "target": "en",
    },
    {
        "name": "Poland (Krakow)",
        "file": f"{BENCHMARK_DIR}/poland_krakow_IwT4ZXKWNSs_16k.wav",
        "hints": ["pl"],
        "target": "en",
    },
    {
        "name": "Morocco (Marrakech)",
        "file": f"{BENCHMARK_DIR}/morocco_marrakech_mNqNF0iiWR0_16k.wav",
        "hints": ["ar", "fr"],
        "target": "en",
    },
    {
        "name": "Colombia (Medellin)",
        "file": f"{BENCHMARK_DIR}/colombia_medellin_KrSpBWnBelE_16k.wav",
        "hints": ["es"],
        "target": "en",
    },
    {
        "name": "Indonesia (Jakarta)",
        "file": f"{BENCHMARK_DIR}/indonesia_jakarta_P13mMiIL_2I_16k.wav",
        "hints": ["id"],
        "target": "en",
    },
    {
        "name": "France (Paris)",
        "file": f"{BENCHMARK_DIR}/france_paris_h13cCP9wCpw_16k.wav",
        "hints": ["fr"],
        "target": "en",
    },
    {
        "name": "Italy (Naples)",
        "file": f"{BENCHMARK_DIR}/italy_naples_nV8zOhYBHP4_16k.wav",
        "hints": ["it"],
        "target": "en",
    },
    {
        "name": "Mexico (CDMX)",
        "file": f"{BENCHMARK_DIR}/mexico_cdmx_OO9kSxcT9Rg_16k.wav",
        "hints": ["es"],
        "target": "en",
    },
]


def check_daemon():
    print("[*] Checking resident ASR daemon health...")
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=5)
        data = json.loads(req.read().decode())
        print(f"[+] Daemon is healthy: {data}")
        return True
    except Exception as e:
        print(f"[-] Daemon check failed: {e}")
        return False


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==========================================================================")
    print("=== WHATTUBE MASTER 20-VLOG FULL-LENGTH BENCHMARK EVALUATION SUITE ===")
    print("==========================================================================")

    if not check_daemon():
        print("[-] Aborting: ASR daemon is not reachable on port 8766.")
        sys.exit(1)

    print("[*] Pre-loading shared pipeline models (VAD, LID, Marian CT2)...")
    vad = EnergyAndSileroVAD(default_config.silero_vad_onnx)
    lid = WhisperTinyLID(default_config.encoder_onnx, default_config.decoder_onnx)
    translator = MarianTranslator(device="cpu", use_ct2=True)
    print("[+] Shared models ready.")

    total_suite_start = time.time()
    all_summaries = {}
    total_audio_sec = 0.0
    total_triggered_bursts = 0
    total_emitted_subtitles = 0
    total_discarded_english = 0
    total_discarded_low_conf = 0
    total_discarded_dedup = 0

    for idx, item in enumerate(CORPUS_20, 1):
        file_path = item["file"]
        country_name = item["name"]

        if not os.path.exists(file_path):
            print(f"[!] Warning: Audio file not found: {file_path}, skipping.")
            continue

        print(f"\n==========================================================================")
        print(f"[{idx}/{len(CORPUS_20)}] EVALUATING: {country_name}")
        print(f"File: {file_path}")
        print(f"Language hints: {item['hints']}")
        print(f"==========================================================================")

        t0 = time.time()
        out_json = file_path.replace(".wav", "_result_v2.json")

        try:
            summary = run_vlog_benchmark(
                audio_path=file_path,
                title_hints=item["hints"],
                target_lang=item["target"],
                vad=vad,
                lid=lid,
                translator=translator,
            )
            summary["country"] = country_name
            summary["hints"] = item["hints"]
            summary["elapsed_benchmark_sec"] = time.time() - t0
            all_summaries[country_name] = summary

            total_audio_sec += summary["evaluated_duration_sec"]
            total_triggered_bursts += summary["triggered_bursts"]
            total_emitted_subtitles += summary["emitted_subtitles"]
            total_discarded_english += summary["discarded_english_or_daemon"]
            total_discarded_low_conf += summary["discarded_low_confidence"]
            total_discarded_dedup += summary["discarded_dedup"]

            with open(out_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[+] Saved {country_name} result to {out_json}")

        except Exception as e:
            print(f"[-] Error evaluating {country_name}: {e}")
            import traceback
            traceback.print_exc()

    total_suite_elapsed = time.time() - total_suite_start

    aggregate_summary = {
        "suite_version": "v2_hardened",
        "total_countries": len(all_summaries),
        "total_audio_duration_sec": round(total_audio_sec, 2),
        "total_audio_duration_hours": round(total_audio_sec / 3600.0, 2),
        "total_benchmark_elapsed_sec": round(total_suite_elapsed, 2),
        "overall_speedup_vs_realtime": round(total_audio_sec / max(total_suite_elapsed, 1.0), 2),
        "total_triggered_bursts": total_triggered_bursts,
        "total_emitted_subtitles": total_emitted_subtitles,
        "total_discarded_english_or_daemon": total_discarded_english,
        "total_discarded_low_confidence": total_discarded_low_conf,
        "total_discarded_dedup": total_discarded_dedup,
        "per_country_results": {
            name: {
                "file": os.path.basename(s["audio_file"]),
                "duration_min": round(s["evaluated_duration_sec"] / 60.0, 2),
                "elapsed_sec": round(s["elapsed_benchmark_sec"], 2),
                "triggered_bursts": s["triggered_bursts"],
                "emitted_subtitles": s["emitted_subtitles"],
                "discarded_english": s["discarded_english_or_daemon"],
                "discarded_low_conf": s["discarded_low_confidence"],
                "discarded_dedup": s["discarded_dedup"],
            }
            for name, s in all_summaries.items()
        },
    }

    master_summary_file = f"{BENCHMARK_DIR}/master_20country_benchmark_summary_v2.json"
    with open(master_summary_file, "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print("\n==========================================================================")
    print("=== MASTER 20-COUNTRY BENCHMARK EVALUATION COMPLETE ===")
    print("==========================================================================")
    print(f"Total Evaluated Audio: {total_audio_sec/3600:.2f} hours ({total_audio_sec:.1f}s)")
    print(f"Total Suite Run Time:  {total_suite_elapsed/60:.2f} minutes ({total_suite_elapsed:.1f}s)")
    print(f"Overall Speedup:       {total_audio_sec/max(total_suite_elapsed, 1.0):.1f}x real-time")
    print(f"Triggered Bursts:      {total_triggered_bursts}")
    print(f"Emitted Subtitles:     {total_emitted_subtitles}")
    print(f"Discarded English:     {total_discarded_english}")
    print(f"Discarded Low-Conf:    {total_discarded_low_conf}")
    print(f"Discarded Dedup:       {total_discarded_dedup}")
    print(f"Master Summary File:   {master_summary_file}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
