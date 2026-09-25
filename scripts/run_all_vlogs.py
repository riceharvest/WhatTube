"""Orchestrator to run WhatTube benchmarks sequentially across all 10 target vlog videos."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

from scripts.benchmark_vlog_suite import run_vlog_benchmark

BENCHMARK_DIR = "/mnt/ssd/scratch_benchmark"

SUITE = [
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
]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== WHATTUBE 10-COUNTRY BENCHMARK EVALUATION SUITE ===")
    print("==================================================================")

    all_summaries = {}
    for item in SUITE:
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
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[-] Error evaluating {item['name']}: {e}")

    summary_file = f"{BENCHMARK_DIR}/suite_aggregate_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[+] Suite complete! Aggregate saved to {summary_file}")

if __name__ == "__main__":
    main()
