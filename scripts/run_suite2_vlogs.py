"""Orchestrator to run WhatTube benchmarks sequentially across Suite 2 (10 additional diverse travel vlogs)."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_vlog_suite import run_vlog_benchmark

BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"

SUITE2 = [
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
        "name": "Colombia (Medellín)",
        "file": f"{BENCHMARK_DIR}/colombia_medellin_KrSpBWnBelE_16k.wav",
        "hints": ["es"],
        "target": "en",
    },
]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== WHATTUBE SUITE 2 (10-COUNTRY EXPANSION) BENCHMARK ===")
    print("==================================================================")

    all_summaries = {}
    for item in SUITE2:
        file_path = item["file"]
        if not os.path.exists(file_path):
            print(f"[!] Warning: Audio file not found: {file_path}, skipping.")
            continue

        print(f"\n>>> Running Evaluation: {item['name']} ({os.path.basename(file_path)})")
        t0 = time.time()
        out_json = file_path.replace(".wav", "_result.json")
        try:
            summary = run_vlog_benchmark(
                audio_path=file_path,
                title_hints=item["hints"],
                target_lang=item["target"],
            )
            summary["country"] = item["name"]
            summary["elapsed_benchmark_sec"] = time.time() - t0
            all_summaries[item["name"]] = summary

            with open(out_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[+] Saved {item['name']} result to {out_json}")
        except Exception as e:
            print(f"[-] Error evaluating {item['name']}: {e}")

    summary_file = f"{BENCHMARK_DIR}/suite2_aggregate_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[+] Suite 2 complete! Aggregate saved to {summary_file}")

if __name__ == "__main__":
    main()
