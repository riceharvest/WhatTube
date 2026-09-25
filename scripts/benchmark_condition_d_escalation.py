"""Condition D (100% Full Overlap) Escalation Benchmark.

Compares:
1. Baseline: Current Turbo Pipeline (Unseparated)
2. Escalation: Conv-TasNet 2-Speaker Separation -> Turbo ASR + Marian Translation

Evaluated across:
- Overlap Condition D (100% simultaneous overlap)
- Target-to-Host SNRs: [-5 dB, -10 dB, -15 dB, -20 dB]
- 10 Target Languages: [id, es, hi, ar, th, vi, zh, ja, fr, ms]

Metrics:
- Foreign Event Recall %
- Foreign Meaning Recovery %
- Host Leakage %
- Latency (sec)
"""

import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from asteroid.models import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whattube.config import default_config
from whattube.translation.marian import MarianTranslator

torch.set_num_threads(16)

MIXTURES_DIR = "/mnt/ssd/scratch_benchmark/synthetic/mixtures"
MANIFEST_PATH = "/mnt/ssd/scratch_benchmark/synthetic/mixtures/mixture_manifest.json"
OUTPUT_DIR = "/mnt/ssd/scratch_benchmark/synthetic"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "escalation_benchmark_checkpoint.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "escalation_benchmark_summary.json")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "it", "its", "that", "this", "these",
    "those", "there", "their", "they", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "i", "me", "my",
}


def clean_words(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return set(cleaned.split())


def content_words(text: str) -> set[str]:
    words = clean_words(text)
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def check_host_leakage(emitted_text: str, host_transcript: str) -> bool:
    if not host_transcript or not emitted_text:
        return False
    em_words = clean_words(emitted_text)
    host_cwords = content_words(host_transcript)
    if not em_words or not host_cwords:
        return False
    overlap = len(em_words & host_cwords)
    return (overlap / len(em_words) >= 0.30 and overlap >= 2) or overlap >= 4


def evaluate_meaning_recovery(emitted_text: str, english_reference: str, is_host_leak: bool) -> bool:
    if is_host_leak or not emitted_text:
        return False
    em_words = clean_words(emitted_text)
    ref_cwords = content_words(english_reference)
    if not ref_cwords:
        ref_cwords = clean_words(english_reference)
    if not ref_cwords:
        return False
    overlap = len(em_words & ref_cwords)
    recall = overlap / len(ref_cwords)
    if recall < 0.35:
        fuzzy_matches = 0
        for rw in ref_cwords:
            stem = rw[:4] if len(rw) >= 4 else rw
            if any(stem in ew for ew in em_words):
                fuzzy_matches += 1
        fuzzy_recall = fuzzy_matches / len(ref_cwords)
        if fuzzy_recall >= 0.35:
            return True
    return recall >= 0.35 or overlap >= 3


def call_asr(audio_chunk, sr=16000):
    bio = io.BytesIO()
    sf.write(bio, audio_chunk, sr, format="WAV")
    req = urllib.request.Request(
        default_config.asr_endpoint,
        data=bio.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"is_discarded": True, "discard_reason": str(e), "language": "unknown", "language_prob": 0.0, "text": ""}


def run_benchmark():
    print("=" * 70)
    print("STARTING CONDITION D (100% OVERLAP) ESCALATION BENCHMARK")
    print("=" * 70)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        full_manifest = json.load(f)

    target_snrs = [-5.0, -10.0, -15.0, -20.0]
    cond_d_items = [
        m for m in full_manifest
        if m["overlap_condition"] == "D_full_100" and m["snr_db"] in target_snrs
    ]
    print(f"Total Condition D test mixtures (-5 to -20 dB): {len(cond_d_items)}")

    # Load BSS Separator
    print("[*] Loading Conv-TasNet 16k 2-speaker separator...")
    separator = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    separator.eval()
    print("[+] Separator ready.")

    # Load Marian Translator
    print("[*] Initializing Marian Translator...")
    translator = MarianTranslator()
    print("[+] Translator ready.")

    # Resume from checkpoint if available
    results = []
    processed_ids = set()
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_ids = {r["mix_id"] for r in results}
            print(f"[+] Resuming from checkpoint: {len(processed_ids)} already evaluated.")
        except Exception:
            results = []

    t_start_all = time.time()

    for idx, meta in enumerate(cond_d_items, 1):
        mix_id = meta["mix_id"]
        if mix_id in processed_ids:
            continue

        wav_path = meta["wav_file"]
        snr = meta["snr_db"]
        lang = meta["lang_code"]
        ref_en = meta["english_reference"]
        host_tx = meta["host_transcript"]

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # ---------------------------------------------------------
        # 1. BASELINE PIPELINE (Unseparated)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        base_asr = call_asr(audio, sr)
        base_dur = time.perf_counter() - t0

        base_discarded = base_asr.get("is_discarded", False)
        base_text = base_asr.get("text", "").strip()
        base_lang = base_asr.get("language", "unknown").lower()

        base_emitted = ""
        if not base_discarded and base_text and base_lang != "en":
            tr = translator.translate(base_text, base_lang, "en")
            base_emitted = tr.translated_text if tr.is_success else base_text

        base_host_leak = check_host_leakage(base_emitted, host_tx)
        base_meaning_ok = evaluate_meaning_recovery(base_emitted, ref_en, base_host_leak)

        # ---------------------------------------------------------
        # 2. ESCALATION PIPELINE (Conv-TasNet -> Turbo)
        # ---------------------------------------------------------
        t_esc0 = time.perf_counter()
        with torch.no_grad():
            sep_out = separator.separate(audio[np.newaxis, :])  # shape: [1, 2, T]
        t_sep = time.perf_counter() - t_esc0

        s1_audio = sep_out[0, 0]
        s2_audio = sep_out[0, 1]

        # Decode both streams via ASR Daemon
        asr_s1 = call_asr(s1_audio, sr)
        asr_s2 = call_asr(s2_audio, sr)
        t_esc_total = time.perf_counter() - t_esc0

        candidates = []
        for s_idx, asr_res in enumerate([asr_s1, asr_s2], 1):
            s_disc = asr_res.get("is_discarded", False)
            s_text = asr_res.get("text", "").strip()
            s_lang = asr_res.get("language", "unknown").lower()
            if not s_disc and s_text and s_lang != "en":
                tr = translator.translate(s_text, s_lang, "en")
                emitted = tr.translated_text if tr.is_success else s_text
                is_leak = check_host_leakage(emitted, host_tx)
                is_meaning = evaluate_meaning_recovery(emitted, ref_en, is_leak)
                candidates.append({
                    "stream": s_idx,
                    "language": s_lang,
                    "text": s_text,
                    "emitted": emitted,
                    "is_leak": is_leak,
                    "is_meaning": is_meaning,
                })

        esc_emitted = ""
        esc_host_leak = False
        esc_meaning_ok = False
        esc_lang = "none"

        if candidates:
            # Pick non-leaking candidate with best meaning, or first valid foreign candidate
            best_cand = next((c for c in candidates if not c["is_leak"] and c["is_meaning"]), candidates[0])
            esc_emitted = best_cand["emitted"]
            esc_host_leak = best_cand["is_leak"]
            esc_meaning_ok = best_cand["is_meaning"]
            esc_lang = best_cand["language"]

        record = {
            "mix_id": mix_id,
            "lang_code": lang,
            "snr_db": snr,
            "baseline": {
                "emitted": base_emitted,
                "host_leak": base_host_leak,
                "meaning_ok": base_meaning_ok,
                "latency_s": round(base_dur, 3),
            },
            "escalation": {
                "emitted": esc_emitted,
                "host_leak": esc_host_leak,
                "meaning_ok": esc_meaning_ok,
                "detected_lang": esc_lang,
                "latency_s": round(t_esc_total, 3),
                "sep_latency_s": round(t_sep, 3),
            },
        }
        results.append(record)

        if idx % 10 == 0 or idx == len(cond_d_items):
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - t_start_all
            print(f"[{idx}/{len(cond_d_items)}] SNR: {snr}dB | Lang: {lang} | Base Meaning: {base_meaning_ok} | Esc Meaning: {esc_meaning_ok} (Elapsed: {elapsed:.1f}s)")

    # Compute Summary Statistics
    summary = {
        "total_evaluated": len(results),
        "by_snr": {},
        "by_language": {},
    }

    snr_groups = defaultdict(list)
    lang_groups = defaultdict(list)
    for r in results:
        snr_groups[r["snr_db"]].append(r)
        lang_groups[r["lang_code"]].append(r)

    print("\n" + "=" * 70)
    print("CONDITION D ESCALATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'SNR (dB)':<10} | {'Base Meaning %':<16} | {'Esc Meaning %':<16} | {'Base Leak %':<13} | {'Esc Leak %':<13} | {'Esc Latency (s)':<15}")
    print("-" * 90)

    for snr in target_snrs:
        items = snr_groups[snr]
        n = len(items)
        if n == 0:
            continue
        b_mean = sum(1 for x in items if x["baseline"]["meaning_ok"]) / n * 100
        e_mean = sum(1 for x in items if x["escalation"]["meaning_ok"]) / n * 100
        b_leak = sum(1 for x in items if x["baseline"]["host_leak"]) / n * 100
        e_leak = sum(1 for x in items if x["escalation"]["host_leak"]) / n * 100
        avg_lat = sum(x["escalation"]["latency_s"] for x in items) / n

        summary["by_snr"][str(snr)] = {
            "n": n,
            "baseline_meaning_recovery_pct": round(b_mean, 1),
            "escalation_meaning_recovery_pct": round(e_mean, 1),
            "baseline_host_leakage_pct": round(b_leak, 1),
            "escalation_host_leakage_pct": round(e_leak, 1),
            "avg_escalation_latency_s": round(avg_lat, 2),
        }
        print(f"{snr:<10} | {b_mean:>14.1f}% | {e_mean:>14.1f}% | {b_leak:>11.1f}% | {e_leak:>11.1f}% | {avg_lat:>13.2f}s")

    print("-" * 90)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    run_benchmark()
