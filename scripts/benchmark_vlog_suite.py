"""Comprehensive verification harness for WhatTube on full-length travel vlogs."""

import os
import sys
import time
import json
import io
import argparse
from typing import List, Dict, Any
import soundfile as sf
import urllib.request
import urllib.error

from whattube.config import default_config
from whattube.vad import EnergyAndSileroVAD
from whattube.lid import WhisperTinyLID
from whattube.event_aggregator import DynamicEventAggregator
from whattube.translation.marian import MarianTranslator

def run_vlog_benchmark(
    audio_path: str,
    title_hints: List[str],
    target_lang: str = "en",
    start_sec: float = 0.0,
    max_duration_sec: float = None,
    stride_sec: float = 1.0,
    window_sec: float = 3.0,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"[*] Starting Vlog Benchmark: {os.path.basename(audio_path)}")
    print(f"[*] Title hints: {title_hints}, Target lang: {target_lang}")
    print(f"=======================================================")

    t0_total = time.perf_counter()
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_len_sec = len(audio) / sr
    end_sec = total_len_sec if max_duration_sec is None else min(total_len_sec, start_sec + max_duration_sec)
    print(f"[*] Total duration: {total_len_sec:.1f}s. Evaluating [{start_sec:.1f}s - {end_sec:.1f}s]")

    # Stage 1 components
    vad = EnergyAndSileroVAD(default_config.silero_vad_onnx)
    lid = WhisperTinyLID(default_config.encoder_onnx, default_config.decoder_onnx)
    agg = DynamicEventAggregator(
        lid_suspicious=default_config.lid_suspicious,
        lid_english_safe=default_config.lid_english_safe,
    )
    translator = MarianTranslator(device="cpu", use_ct2=True)

    # 1. Stage 1 Streaming Simulation
    t0_s1 = time.perf_counter()
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    step_samples = int(stride_sec * sr)
    win_samples = int(window_sec * sr)

    triggered_events = []
    total_steps = 0
    speech_steps = 0

    print("[*] Running Stage 1 continuous VAD + LID streaming...")
    for i in range(start_sample, end_sample - win_samples, step_samples):
        total_steps += 1
        t_start = i / sr
        t_end = (i + win_samples) / sr
        chunk = audio[i : i + win_samples]

        is_sp, _, _ = vad.is_speech(chunk)
        lid_res = None
        if is_sp:
            speech_steps += 1
            lid_res = lid.predict(chunk, sample_rate=sr)

        evt = agg.update(t_start, t_end, is_sp, lid_res)
        if evt is not None:
            triggered_events.append(evt)

        if total_steps % 300 == 0:
            print(f"  ... Stage 1 progress: {total_steps}/{int(end_sec - start_sec)}s ({len(triggered_events)} events triggered)", flush=True)

    s1_elapsed = time.perf_counter() - t0_s1
    print(
        f"[+] Stage 1 completed in {s1_elapsed:.2f}s "
        f"({(end_sec - start_sec) / s1_elapsed:.1f}x real-time). "
        f"Triggered {len(triggered_events)} burst events across {total_steps} seconds.",
        flush=True,
    )

    # 2. Stage 2 GPU ASR & Stage 3 Translation
    print(f"[*] Dispatching {len(triggered_events)} events to Resident ASR Daemon & Marian Translator...", flush=True)
    
    results = []
    emitted_count = 0
    discarded_english = 0
    discarded_confidence = 0
    discarded_dedup = 0

    last_emitted_text = ""
    last_emitted_time = 0.0

    south_asian = {"bn", "hi", "ur", "ar"}
    east_asian = {"ja", "zh", "ko"}
    southeast_asian = {"id", "ms", "th", "vi", "tl"}
    european = {"fr", "es", "it", "de", "pt", "nl", "ru", "pl", "uk", "sv", "no", "fi"}

    for idx, evt in enumerate(triggered_events):
        s_audio = audio[int(evt.start_sec * sr) : int(evt.end_sec * sr)]
        bio = io.BytesIO()
        sf.write(bio, s_audio, 16000, format="WAV")

        req = urllib.request.Request(
            default_config.asr_endpoint,
            data=bio.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                asr_res = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[-] ASR request failed for event #{evt.event_id}: {e}")
            continue

        lang = asr_res.get("language", "unknown").lower()
        prob = asr_res.get("language_prob", 1.0)
        text = asr_res.get("text", "").strip()
        is_discarded = asr_res.get("is_discarded", False)
        discard_reason = asr_res.get("discard_reason", "")

        # ASR language check
        if is_discarded or lang == "en":
            discarded_english += 1
            results.append({
                "event_id": evt.event_id,
                "start_sec": evt.start_sec,
                "end_sec": evt.end_sec,
                "status": "discarded",
                "reason": discard_reason or "Classified as English",
                "lang": lang,
                "prob": prob,
                "text": text,
            })
            continue

        # Bayesian confidence filter
        is_local = bool(title_hints and lang in title_hints)
        min_prob = 0.15 if is_local else (0.40 if title_hints else 0.25)
        if prob < min_prob:
            discarded_confidence += 1
            reason = f"Low prob {lang.upper()} ({prob:.2f} < {min_prob:.2f})"
            results.append({
                "event_id": evt.event_id,
                "start_sec": evt.start_sec,
                "end_sec": evt.end_sec,
                "status": "discarded",
                "reason": reason,
                "lang": lang,
                "prob": prob,
                "text": text,
            })
            continue

        # Multi-window deduplication
        clean_curr = text.lower()
        clean_last = last_emitted_text.lower()
        time_diff = abs(evt.start_sec - last_emitted_time)
        curr_words = set(clean_curr.replace(".", "").replace(",", "").split())
        last_words = set(clean_last.replace(".", "").replace(",", "").split())
        overlap = len(curr_words & last_words) / max(len(curr_words), 1) if last_words else 0.0

        if (clean_curr == clean_last or overlap >= 0.8) and time_diff < 3.5 and len(clean_curr) <= len(clean_last):
            discarded_dedup += 1
            continue

        # Translation
        trans_res = translator.translate(text, source_lang=lang, target_lang=target_lang)
        last_emitted_text = text
        last_emitted_time = evt.end_sec
        emitted_count += 1

        print(
            f"  [EMITTED] [{evt.start_sec:.1f}s - {evt.end_sec:.1f}s] ({lang.upper()}, p={prob:.2f}) "
            f"Original: \"{text}\" -> Subtitle: \"{trans_res.translated_text}\"",
            flush=True,
        )
        results.append({
            "event_id": evt.event_id,
            "start_sec": evt.start_sec,
            "end_sec": evt.end_sec,
            "status": "emitted",
            "lang": lang,
            "prob": prob,
            "original_text": text,
            "translated_text": trans_res.translated_text,
            "latency_ms": trans_res.latency_ms,
        })

    summary = {
        "audio_file": os.path.basename(audio_path),
        "evaluated_duration_sec": end_sec - start_sec,
        "total_steps": total_steps,
        "speech_steps": speech_steps,
        "triggered_bursts": len(triggered_events),
        "emitted_subtitles": emitted_count,
        "discarded_english_or_daemon": discarded_english,
        "discarded_low_confidence": discarded_confidence,
        "discarded_dedup": discarded_dedup,
        "results": results,
    }

    print("\n---------------- SUMMARY ----------------")
    print(f"Evaluated Duration: {summary['evaluated_duration_sec']:.1f}s ({summary['evaluated_duration_sec']/60:.1f} min)")
    print(f"Triggered Bursts:   {summary['triggered_bursts']}")
    print(f"Emitted Subtitles:  {summary['emitted_subtitles']}")
    print(f"Discarded English:  {summary['discarded_english_or_daemon']}")
    print(f"Discarded Low-Prob: {summary['discarded_low_confidence']}")
    print(f"-----------------------------------------\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatTube Vlog Benchmark Harness")
    parser.add_argument("audio_path", help="Path to 16kHz WAV audio")
    parser.add_argument("--hints", nargs="+", default=[], help="Title language hints (e.g. fr es it ja)")
    parser.add_argument("--target", default="en", help="Target translation language")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration to evaluate")
    parser.add_argument("--output", default=None, help="Save summary JSON path")
    args = parser.parse_args()

    res = run_vlog_benchmark(
        audio_path=args.audio_path,
        title_hints=args.hints,
        target_lang=args.target,
        start_sec=args.start,
        max_duration_sec=args.duration,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[+] Saved results to {args.output}")
