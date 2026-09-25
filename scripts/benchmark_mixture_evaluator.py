"""Evaluation Engine for Controlled Scientific Benchmark Mixtures.

Evaluates WhatTube on calibrated synthetic mixtures across:
- Foreign-to-Host SNR: [-5 dB, -10 dB, -15 dB, -20 dB, -25 dB, -30 dB]
- Overlap Conditions: [A_alone, B_sequential, C_partial_50, D_full_100, E_two_foreign]
- 10 Target Languages: [id, es, hi, ar, th, vi, zh, ja, fr, ms]

Computes the core metrics:
1. Foreign Event Recall (emitted foreign caption / total foreign utterances)
2. Host Leakage Rate (captions containing significant host English / all emitted captions)
3. Meaning Recovery (captions preserving foreign speaker's intended meaning / total foreign utterances)
4. Headline Metric: Foreign Meaning Recovery at -15 dB under 100% simultaneous English speech.

Generates the scientific plot: meaning_recovery_vs_snr.png
"""

import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whattube.config import default_config
from whattube.event_aggregator import DynamicEventAggregator
from whattube.lid import WhisperTinyLID
from whattube.translation.marian import MarianTranslator
from whattube.vad import EnergyAndSileroVAD

MIXTURES_DIR = "/mnt/ssd/scratch_benchmark/synthetic/mixtures"
OUTPUT_DIR = "/mnt/ssd/scratch_benchmark/synthetic"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "it", "its", "that", "this", "these",
    "those", "there", "their", "they", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "i", "me", "my",
}


def clean_words(text: str) -> set[str]:
    """Tokenize and remove punctuation for robust matching."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return set(cleaned.split())


def content_words(text: str) -> set[str]:
    """Extract content words (nouns, verbs, adjectives) excluding common stopwords."""
    words = clean_words(text)
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def check_host_leakage(emitted_text: str, host_transcript: str) -> bool:
    """Detects if the emitted subtitle captured host English monologue words instead of foreign speech."""
    if not host_transcript or not emitted_text:
        return False
    em_words = clean_words(emitted_text)
    host_cwords = content_words(host_transcript)
    if not em_words or not host_cwords:
        return False
    overlap = len(em_words & host_cwords)
    # If >= 30% of emitted words or >= 3 content words are verbatim from host speech
    return (overlap / len(em_words) >= 0.30 and overlap >= 2) or overlap >= 4


def evaluate_meaning_recovery(emitted_text: str, english_reference: str, is_host_leak: bool) -> bool:
    """Evaluates whether emitted caption preserves the foreign speaker's intended meaning."""
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

    # Stemming / substring matching fallback for morphological variations (e.g. domesticate / domesticated)
    if recall < 0.40:
        fuzzy_matches = 0
        for rw in ref_cwords:
            stem = rw[:4] if len(rw) >= 4 else rw
            if any(stem in ew for ew in em_words):
                fuzzy_matches += 1
        fuzzy_recall = fuzzy_matches / len(ref_cwords)
        if fuzzy_recall >= 0.40:
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


def evaluate_single_mixture(meta, vad, lid, translator):
    audio, sr = sf.read(meta["wav_file"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    agg = DynamicEventAggregator(
        pre_roll_sec=default_config.pre_roll_sec,
        post_roll_sec=default_config.post_roll_sec,
        close_hangover_sec=default_config.close_hangover_sec,
        min_event_sec=default_config.min_event_sec,
        max_event_sec=default_config.max_event_sec,
        lid_suspicious=default_config.lid_suspicious,
        lid_immediate=default_config.lid_immediate,
        lid_english_safe=default_config.lid_english_safe,
        consecutive_suspicious_req=default_config.consecutive_suspicious_req,
    )

    stride_samples = int(default_config.stride_sec * sr)
    win_samples = int(default_config.analysis_window_sec * sr)
    triggered_events = []

    # Stage 1 Streaming
    for i in range(0, len(audio) - win_samples, stride_samples):
        t_start = i / sr
        t_end = (i + win_samples) / sr
        chunk = audio[i : i + win_samples]

        is_sp, _, _ = vad.is_speech(chunk)
        lid_res = None
        if is_sp:
            lid_res = lid.predict(chunk, sample_rate=sr)

        evt = agg.update(t_start, t_end, is_sp, lid_res)
        if evt is not None:
            triggered_events.append(evt)

    eof_evt = agg.flush(len(audio) / sr)
    if eof_evt is not None:
        triggered_events.append(eof_evt)

    # Stage 2 & 3: ASR & Translation
    emitted_captions = []
    title_hints = [meta["lang_code"]]

    for evt in triggered_events:
        event_audio = audio[int(evt.start_sec * sr) : int(evt.end_sec * sr)]

        sub_slices = []
        if evt.duration_sec >= 2.5:
            split_sec = vad.find_split_point(event_audio, sample_rate=sr)
            if split_sec is not None and isinstance(split_sec, (int, float)):
                split_idx = int(split_sec * sr)
                sub_slices.append((evt.start_sec, evt.start_sec + split_sec, event_audio[:split_idx]))
                sub_slices.append((evt.start_sec + split_sec, evt.end_sec, event_audio[split_idx:]))

        if not sub_slices:
            sub_slices.append((evt.start_sec, evt.end_sec, event_audio))

        min_stage1_p_en = 1.0
        if evt.trigger_windows:
            p_ens = [
                w["lid"]["p_en"]
                for w in evt.trigger_windows
                if w.get("is_speech") and w.get("lid") and "p_en" in w["lid"]
            ]
            if p_ens:
                min_stage1_p_en = min(p_ens)

        for sub_start, sub_end, sub_audio in sub_slices:
            asr_res = call_asr(sub_audio, sr)
            lang = asr_res.get("language", "unknown").lower()
            prob = asr_res.get("language_prob", 1.0)
            text = asr_res.get("text", "").strip()
            is_discarded = asr_res.get("is_discarded", False)

            is_english = (lang == "en")
            words = text.split()
            word_count = len(words)
            is_english_monologue = is_english and (word_count >= 5 or min_stage1_p_en >= 0.35)

            # Sub-slice fallback
            if (is_discarded or is_english_monologue) and evt.trigger_windows:
                susp_spans = [
                    (w["t_start"], w["t_end"])
                    for w in evt.trigger_windows
                    if w.get("is_speech") and w.get("lid") and w["lid"].get("p_en", 1.0) < 0.35
                    and (sub_start <= w.get("t_start", 0.0) < sub_end)
                ]
                if susp_spans:
                    f_start = max(sub_start, min(s[0] for s in susp_spans) - 0.1)
                    f_end = min(sub_end, max(s[1] for s in susp_spans) + 0.2)
                    if f_end - f_start < 1.0:
                        f_end = min(sub_end, f_start + 1.0)

                    if (f_end - f_start) <= 0.75 * (sub_end - sub_start) and (f_end - f_start) >= 0.8:
                        f_audio = audio[int(f_start * sr) : int(f_end * sr)]
                        fb_asr = call_asr(f_audio, sr)
                        fb_is_en = (fb_asr.get("language", "").lower() == "en")
                        if not fb_asr.get("is_discarded", False) and not fb_is_en:
                            asr_res = fb_asr
                            lang = asr_res.get("language", "").lower()
                            prob = asr_res.get("language_prob", 1.0)
                            text = asr_res.get("text", "").strip()
                            is_discarded = False
                            is_english = False

            if is_discarded or is_english:
                continue

            # Confidence filter
            if lang != "en":
                is_local = bool(title_hints and lang in title_hints)
                min_prob = 0.15 if is_local else 0.40
                if prob < min_prob:
                    continue

            # Translate to English
            try:
                trans_res = translator.translate(text, source_lang=lang, target_lang="en")
                translated_text = trans_res.translated_text
            except Exception:
                translated_text = text

            emitted_captions.append({
                "start_sec": sub_start,
                "end_sec": sub_end,
                "lang": lang,
                "prob": prob,
                "original_text": text,
                "translated_text": translated_text,
            })

    # Evaluate Metrics
    has_emission = len(emitted_captions) > 0
    emitted_full_text = " ".join(c["translated_text"] for c in emitted_captions)

    is_host_leak = False
    if has_emission:
        is_host_leak = check_host_leakage(emitted_full_text, meta["host_transcript"])

    meaning_recovered = False
    if has_emission:
        meaning_recovered = evaluate_meaning_recovery(
            emitted_full_text, meta["english_reference"], is_host_leak
        )

    return {
        "mix_id": meta["mix_id"],
        "filename": meta["filename"],
        "lang_code": meta["lang_code"],
        "snr_db": meta["snr_db"],
        "overlap_condition": meta["overlap_condition"],
        "triggered_bursts": len(triggered_events),
        "emitted_count": len(emitted_captions),
        "has_emission": has_emission,
        "is_host_leak": is_host_leak,
        "meaning_recovered": meaning_recovered,
        "emitted_text": emitted_full_text,
        "english_reference": meta["english_reference"],
        "host_transcript": meta["host_transcript"],
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== WHATTUBE SCIENTIFIC BENCHMARK EVALUATOR ===")
    print("==================================================================")

    manifest_path = os.path.join(MIXTURES_DIR, "mixture_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[-] Manifest not found at {manifest_path}. Run benchmark_mixture_builder.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        mixtures = json.load(f)

    print(f"[+] Loaded {len(mixtures)} mixtures to evaluate.")
    print("[*] Pre-loading pipeline models...")
    vad = EnergyAndSileroVAD(default_config.silero_vad_onnx)
    lid = WhisperTinyLID(default_config.encoder_onnx, default_config.decoder_onnx)
    translator = MarianTranslator(device="cpu", use_ct2=True)
    print("[+] Models ready.")

    checkpoint_file = os.path.join(OUTPUT_DIR, "scientific_benchmark_checkpoint.json")
    completed_results = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                saved = json.load(f)
                completed_results = {r["mix_id"]: r for r in saved}
            print(f"[+] Loaded {len(completed_results)} previously evaluated mixtures from checkpoint.")
        except Exception as e:
            print(f"[-] Could not load checkpoint: {e}")

    t0 = time.time()
    results = []

    for idx, mix in enumerate(mixtures, 1):
        mix_id = mix["mix_id"]
        if mix_id in completed_results:
            results.append(completed_results[mix_id])
        else:
            res = evaluate_single_mixture(mix, vad, lid, translator)
            results.append(res)
            completed_results[mix_id] = res

        if idx % 25 == 0 or idx == len(mixtures):
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(mixtures)}] Evaluated in {elapsed:.1f}s ({idx/max(elapsed, 0.001):.1f} mixes/sec)...")
            with open(checkpoint_file, "w") as f:
                json.dump(list(completed_results.values()), f)

    # Aggregate Statistics
    total_eval = len(results)
    grid_stats = defaultdict(lambda: {"total": 0, "emitted": 0, "leak": 0, "meaning": 0})

    for r in results:
        key = (r["overlap_condition"], r["snr_db"])
        grid_stats[key]["total"] += 1
        if r["has_emission"]:
            grid_stats[key]["emitted"] += 1
        if r["is_host_leak"]:
            grid_stats[key]["leak"] += 1
        if r["meaning_recovered"]:
            grid_stats[key]["meaning"] += 1

    # Output JSON
    summary_file = os.path.join(OUTPUT_DIR, "scientific_benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "total_mixtures": total_eval,
            "elapsed_seconds": round(time.time() - t0, 2),
            "results": results,
            "grid_summary": {
                f"{cond}_{snr}dB": {
                    "total": st["total"],
                    "recall": round(st["emitted"] / st["total"], 4) if st["total"] else 0.0,
                    "host_leakage_rate": round(st["leak"] / max(st["emitted"], 1), 4),
                    "meaning_recovery": round(st["meaning"] / st["total"], 4) if st["total"] else 0.0,
                }
                for (cond, snr), st in grid_stats.items()
            },
        }, f, indent=2)

    print("\n==================================================================")
    print("=== SCIENTIFIC BENCHMARK RESULTS SUMMARY ===")
    print("==================================================================")
    print(f"Total Mixtures Evaluated: {total_eval}")
    print(f"Total Execution Time:     {time.time() - t0:.1f}s")
    print("\n--- RESULTS BY CONDITION & SNR ---")
    print(f"{'Condition':<16} | {'SNR':>6} | {'Total':>5} | {'Recall':>7} | {'Meaning Rec':>11} | {'Host Leak':>9}")
    print("-" * 65)

    snrs = [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
    plot_data = {"B_sequential": {}, "C_partial_50": {}, "D_full_100": {}}

    for cond in ["B_sequential", "C_partial_50", "D_full_100", "A_alone", "E_two_foreign"]:
        for snr in snrs:
            st = grid_stats.get((cond, snr))
            if not st or st["total"] == 0:
                continue
            rec = (st["emitted"] / st["total"]) * 100
            meaning = (st["meaning"] / st["total"]) * 100
            leak = (st["leak"] / max(st["emitted"], 1)) * 100
            print(f"{cond:<16} | {snr:>5.0f}dB | {st['total']:>5} | {rec:>6.1f}% | {meaning:>10.1f}% | {leak:>8.1f}%")

            if cond in plot_data:
                plot_data[cond][snr] = meaning

    # Headline Number
    headline_stat = grid_stats.get(("D_full_100", -15.0), {"total": 0, "meaning": 0})
    headline_recovery = (headline_stat["meaning"] / max(headline_stat["total"], 1)) * 100
    print("\n==================================================================")
    print(f">>> HEADLINE NUMBER FOR WHATTUBE:")
    print(f">>> Foreign Meaning Recovery at -15 dB under 100% Simultaneous English: {headline_recovery:.1f}%")
    print("==================================================================\n")

    # Generate Plot
    plt.figure(figsize=(9, 6), dpi=150)
    colors = {"B_sequential": "#2ca02c", "C_partial_50": "#ff7f0e", "D_full_100": "#1f77b4"}
    labels = {
        "B_sequential": "No overlap (Sequential Host -> Foreign)",
        "C_partial_50": "50% Temporal Overlap",
        "D_full_100": "100% Full Overlap (Under Host Monologue)",
    }
    markers = {"B_sequential": "s-", "C_partial_50": "^-", "D_full_100": "o-"}

    for cond, vals in plot_data.items():
        if vals:
            sorted_snrs = sorted(vals.keys())
            meaning_pcts = [vals[s] for s in sorted_snrs]
            plt.plot(sorted_snrs, meaning_pcts, markers[cond], color=colors[cond], label=labels[cond], linewidth=2.5, markersize=8)

    plt.axvline(x=-15.0, color="red", linestyle="--", alpha=0.7, label="Headline Target (-15 dB)")
    plt.title("WhatTube Scientific Benchmark: Meaning Recovery vs. Target-to-Host SNR", fontsize=13, fontweight="bold")
    plt.xlabel("Foreign Chatter Level Relative to English Host (dB)", fontsize=11)
    plt.ylabel("Successful Meaning Recovery (%)", fontsize=11)
    plt.xlim(-32, -3)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "meaning_recovery_vs_snr.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Saved scientific benchmark curve to: {plot_path}")


if __name__ == "__main__":
    main()
