"""Controlled Mixture Synthesizer for Scientific WhatTube Benchmark.

Generates realistic acoustic mixtures of clear English host speech + foreign chatter stems:
- Exact ground truth transcripts for BOTH speakers.
- Foreign speech relative to host SNR: [-5 dB, -10 dB, -15 dB, -20 dB, -25 dB, -30 dB]
- Overlap conditions:
    A_alone: Foreign speaker alone (no host)
    B_sequential: Host -> Foreign speaker (sequential, no temporal overlap)
    C_partial_50: 50% temporal overlap (foreign begins halfway through host utterance)
    D_full_100: 100% full temporal overlap (foreign completely submerged inside host speech)
    E_two_foreign: Host + TWO foreign background speakers
- Acoustic realism:
    + Distance coloration (low-pass Butterworth filter cutoff 4.2 kHz)
    + Spatial room/street convolution reverb (RT60 ~ 0.20s)
    + Ambient street/market background bed (-28 dB)
    + Randomized window phase offset inside the 1.5s LID analysis window
"""

import json
import os
import random
import sys
import time

import numpy as np
import scipy.signal
import soundfile as sf

STEMS_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems"
HOST_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems/host"
AMBIENT_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems/ambient"
OUTPUT_DIR = "/mnt/ssd/scratch_benchmark/synthetic/mixtures"

SNR_LEVELS = [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
OVERLAP_CONDITIONS = ["A_alone", "B_sequential", "C_partial_50", "D_full_100", "E_two_foreign"]


def calc_active_rms(audio: np.ndarray, frame_len: int = 480, hop_len: int = 160) -> float:
    """Computes RMS over active speech frames to avoid silence dilution."""
    if len(audio) < frame_len:
        return float(np.sqrt(np.mean(audio**2))) + 1e-8
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    max_rms = np.max(frame_rms) if len(frame_rms) > 0 else 1e-6
    active_frames = frame_rms[frame_rms > 0.1 * max_rms]
    if len(active_frames) == 0:
        return float(np.sqrt(np.mean(audio**2))) + 1e-8
    return float(np.mean(active_frames))


def apply_distance_lowpass(audio: np.ndarray, sr: int = 16000, cutoff_hz: float = 4200.0) -> np.ndarray:
    """Simulates high-frequency distance attenuation."""
    b, a = scipy.signal.butter(2, cutoff_hz / (sr / 2.0), btype="low")
    return scipy.signal.filtfilt(b, a, audio).astype(np.float32)


def apply_room_reverb(audio: np.ndarray, sr: int = 16000, rt60: float = 0.20, wet: float = 0.18) -> np.ndarray:
    """Applies realistic room/street impulse response convolution."""
    ir_len = int(rt60 * sr)
    t = np.linspace(0, rt60, ir_len)
    decay = np.exp(-3.0 * t / rt60)
    noise = np.random.randn(ir_len)
    ir = (decay * noise).astype(np.float32)
    ir /= np.sum(np.abs(ir)) + 1e-8

    reverbed = scipy.signal.fftconvolve(audio, ir, mode="full")[:len(audio)]
    return ((1.0 - wet) * audio + wet * reverbed).astype(np.float32)


def generate_mixtures():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== BUILDING CONTROLLED SCIENTIFIC BENCHMARK MIXTURES ===")
    print("==================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # 1. Load Stems Manifests
    with open(os.path.join(STEMS_DIR, "manifest.json")) as f:
        foreign_manifest = json.load(f)
    with open(os.path.join(HOST_DIR, "host_manifest.json")) as f:
        host_manifest = json.load(f)

    ambient_files = []
    amb_manifest_path = os.path.join(AMBIENT_DIR, "ambient_manifest.json")
    if os.path.exists(amb_manifest_path):
        with open(amb_manifest_path) as f:
            ambient_files = [item["wav_file"] for item in json.load(f) if os.path.exists(item["wav_file"])]

    print(f"[+] Loaded {len(foreign_manifest)} foreign stems across 10 languages.")
    print(f"[+] Loaded {len(host_manifest)} English host speech stems.")
    print(f"[+] Loaded {len(ambient_files)} ambient acoustic bed stems.")

    # Group foreign stems by language code
    by_lang = {}
    for item in foreign_manifest:
        by_lang.setdefault(item["lang_code"], []).append(item)

    sr = 16000
    mixtures_manifest = []
    mix_counter = 0

    # Grid Design:
    # 1. Primary Grid across all 10 languages:
    #    For each language (10 stems):
    #    - Evaluate across Conditions B, C, D at each SNR [-5, -10, -15, -20, -25, -30]
    #    - Condition A (Alone) at [-10, -20]
    #    - Condition E (Two foreign) at [-15]
    # This produces ~1,000 highly targeted, balanced mixtures.

    tasks = []
    for lang_code, stems in by_lang.items():
        # Condition D (100% full overlap - the critical headline condition):
        # 6 SNRs x 10 stems = 60 per lang -> 600 total across 10 langs
        for snr in SNR_LEVELS:
            for s_idx, stem in enumerate(stems[:4]):  # 4 stems per SNR for D -> 240
                tasks.append((stem, "D_full_100", snr))

        # Condition B (Sequential, no overlap):
        for snr in [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            for stem in stems[:3]:  # 18 per lang -> 180 total
                tasks.append((stem, "B_sequential", snr))

        # Condition C (50% partial overlap):
        for snr in [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            for stem in stems[:3]:  # 18 per lang -> 180 total
                tasks.append((stem, "C_partial_50", snr))

        # Condition A (Alone):
        for snr in [-10.0, -20.0, -30.0]:
            for stem in stems[:3]:  # 9 per lang -> 90 total
                tasks.append((stem, "A_alone", snr))

        # Condition E (Two foreign):
        for stem in stems[:3]:  # 3 per lang -> 30 total
            tasks.append((stem, "E_two_foreign", -15.0))

    print(f"[*] Total planned synthetic mixtures: {len(tasks)}")

    t0_gen = time.time()
    for task_idx, (foreign_meta, condition, snr_db) in enumerate(tasks, 1):
        mix_counter += 1
        lang_code = foreign_meta["lang_code"]
        lang_name = foreign_meta["lang_name"]

        # Read foreign stem
        f_raw, f_sr = sf.read(foreign_meta["wav_file"], dtype="float32")
        if f_raw.ndim > 1:
            f_raw = f_raw.mean(axis=1)

        # Apply acoustic coloration (low-pass + reverb)
        f_colored = apply_distance_lowpass(f_raw, sr=sr, cutoff_hz=4200.0)
        f_colored = apply_room_reverb(f_colored, sr=sr, rt60=0.20, wet=0.18)
        f_len = len(f_colored)

        # Select host stem
        host_meta = random.choice(host_manifest)
        h_raw, h_sr = sf.read(host_meta["wav_file"], dtype="float32")
        if h_raw.ndim > 1:
            h_raw = h_raw.mean(axis=1)
        h_len = len(h_raw)

        # Scale foreign speech relative to host active speech
        host_rms = calc_active_rms(h_raw)
        foreign_rms = calc_active_rms(f_colored)
        target_foreign_rms = host_rms * (10.0 ** (snr_db / 20.0))
        gain = target_foreign_rms / max(foreign_rms, 1e-6)
        f_scaled = (f_colored * gain).astype(np.float32)

        # Window jitter (random offset within 1.5s LID window stride)
        jitter_samples = int(random.uniform(0.1, 0.6) * sr)

        # Build composite audio based on condition
        if condition == "A_alone":
            total_samples = f_len + int(2.0 * sr)
            mix_audio = np.zeros(total_samples, dtype=np.float32)
            f_start_s = int(1.0 * sr) + jitter_samples
            f_end_s = f_start_s + f_len
            mix_audio[f_start_s:f_end_s] += f_scaled
            host_transcript = ""
            h_start_sec = 0.0
            h_end_sec = 0.0

        elif condition == "B_sequential":
            pause_samples = int(random.uniform(0.3, 0.6) * sr)
            h_start_s = int(0.5 * sr)
            h_end_s = h_start_s + h_len
            f_start_s = h_end_s + pause_samples + jitter_samples
            f_end_s = f_start_s + f_len
            total_samples = f_end_s + int(1.0 * sr)

            mix_audio = np.zeros(total_samples, dtype=np.float32)
            mix_audio[h_start_s:h_end_s] += h_raw
            mix_audio[f_start_s:f_end_s] += f_scaled
            host_transcript = host_meta["transcript"]
            h_start_sec = h_start_s / sr
            h_end_sec = h_end_s / sr

        elif condition == "C_partial_50":
            # Foreign begins at host midpoint (50% overlap)
            h_start_s = int(0.5 * sr)
            h_end_s = h_start_s + h_len
            f_start_s = h_start_s + int(h_len * 0.5) + jitter_samples
            f_end_s = f_start_s + f_len
            total_samples = max(h_end_s, f_end_s) + int(1.0 * sr)

            mix_audio = np.zeros(total_samples, dtype=np.float32)
            mix_audio[h_start_s:h_end_s] += h_raw
            mix_audio[f_start_s:f_end_s] += f_scaled
            host_transcript = host_meta["transcript"]
            h_start_sec = h_start_s / sr
            h_end_sec = h_end_s / sr

        elif condition == "D_full_100":
            # Foreign is completely enveloped by host speech
            # Host must start 1.0s before and end 1.0s after
            f_start_s = int(1.2 * sr) + jitter_samples
            f_end_s = f_start_s + f_len
            needed_host_samples = f_end_s + int(1.2 * sr)

            # Pad or tile host if foreign is longer than host stem
            if h_len < needed_host_samples:
                reps = int(np.ceil(needed_host_samples / h_len))
                h_extended = np.tile(h_raw, reps)[:needed_host_samples]
            else:
                h_extended = h_raw[:needed_host_samples]

            total_samples = len(h_extended)
            mix_audio = np.array(h_extended, dtype=np.float32)
            mix_audio[f_start_s:f_end_s] += f_scaled
            host_transcript = host_meta["transcript"]
            h_start_sec = 0.0
            h_end_sec = total_samples / sr

        elif condition == "E_two_foreign":
            # Host + primary foreign + secondary foreign speaker
            h_start_s = int(0.5 * sr)
            h_end_s = h_start_s + h_len
            f_start_s = int(1.2 * sr) + jitter_samples
            f_end_s = f_start_s + f_len

            # Pick second foreign stem from same or another language
            sec_foreign_meta = random.choice([m for m in foreign_manifest if m["wav_file"] != foreign_meta["wav_file"]])
            s2_raw, _ = sf.read(sec_foreign_meta["wav_file"], dtype="float32")
            if s2_raw.ndim > 1:
                s2_raw = s2_raw.mean(axis=1)
            s2_colored = apply_distance_lowpass(s2_raw, sr=sr, cutoff_hz=3800.0)
            s2_colored = apply_room_reverb(s2_colored, sr=sr, rt60=0.25, wet=0.25)
            s2_rms = calc_active_rms(s2_colored)
            s2_gain = (host_rms * (10.0 ** ((snr_db - 5.0) / 20.0))) / max(s2_rms, 1e-6)
            s2_scaled = (s2_colored * s2_gain).astype(np.float32)

            f2_start_s = f_start_s + int(1.2 * sr)
            f2_end_s = f2_start_s + len(s2_scaled)

            total_samples = max(h_end_s, f_end_s, f2_end_s) + int(1.0 * sr)
            mix_audio = np.zeros(total_samples, dtype=np.float32)
            mix_audio[h_start_s:h_end_s] += h_raw
            mix_audio[f_start_s:f_end_s] += f_scaled
            mix_audio[f2_start_s:f2_end_s] += s2_scaled
            host_transcript = host_meta["transcript"]
            h_start_sec = h_start_s / sr
            h_end_sec = h_end_s / sr

        # Add light background ambience at -28 dB relative to host
        if ambient_files:
            amb_path = random.choice(ambient_files)
            amb_raw, _ = sf.read(amb_path, dtype="float32")
            if amb_raw.ndim > 1:
                amb_raw = amb_raw.mean(axis=1)
            if len(amb_raw) < len(mix_audio):
                amb_raw = np.tile(amb_raw, int(np.ceil(len(mix_audio) / len(amb_raw))))
            amb_chunk = amb_raw[:len(mix_audio)]
            amb_rms = calc_active_rms(amb_chunk)
            amb_gain = (host_rms * (10.0 ** (-28.0 / 20.0))) / max(amb_rms, 1e-6)
            mix_audio += (amb_chunk * amb_gain).astype(np.float32)

        # Normalize to prevent digital clipping
        peak = np.max(np.abs(mix_audio))
        if peak > 0.95:
            mix_audio = (mix_audio / peak) * 0.95

        out_filename = f"mix_{mix_counter:04d}_{lang_code}_{condition}_{int(abs(snr_db))}dB.wav"
        out_wav = os.path.join(OUTPUT_DIR, out_filename)
        sf.write(out_wav, mix_audio, sr)

        meta_record = {
            "mix_id": mix_counter,
            "wav_file": out_wav,
            "filename": out_filename,
            "lang_code": lang_code,
            "lang_name": lang_name,
            "snr_db": snr_db,
            "overlap_condition": condition,
            "total_duration_sec": round(len(mix_audio) / sr, 2),
            "foreign_stem_id": foreign_meta["fleurs_id"],
            "foreign_start_sec": round(f_start_s / sr, 2),
            "foreign_end_sec": round(f_end_s / sr, 2),
            "foreign_duration_sec": round(f_len / sr, 2),
            "foreign_text": foreign_meta["foreign_text"],
            "english_reference": foreign_meta["english_reference"],
            "host_source": host_meta.get("source_vlog", "munich"),
            "host_start_sec": round(h_start_sec, 2),
            "host_end_sec": round(h_end_sec, 2),
            "host_transcript": host_transcript,
        }
        mixtures_manifest.append(meta_record)

        if task_idx % 100 == 0 or task_idx == len(tasks):
            print(f"  [{task_idx}/{len(tasks)}] Generated {condition} {lang_code} ({snr_db}dB) -> {out_filename}")

    manifest_file = os.path.join(OUTPUT_DIR, "mixture_manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(mixtures_manifest, f, indent=2)

    print("\n==================================================================")
    print(f"[+] Generation complete in {time.time() - t0_gen:.1f}s!")
    print(f"[+] Total mixtures created: {len(mixtures_manifest)}")
    print(f"[+] Output directory: {OUTPUT_DIR}")
    print(f"[+] Manifest: {manifest_file}")
    print("==================================================================")


if __name__ == "__main__":
    generate_mixtures()
