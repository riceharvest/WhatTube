"""Download 10 foreign language stems (10 utterances each) from Google FLEURS

Also fetches English ground-truth reference translations via aligned sentence IDs.
Saves audio as 16kHz mono WAV and writes a comprehensive metadata manifest.
"""

import io
import json
import os
import sys
import time

import datasets
import numpy as np
import soundfile as sf
from datasets import load_dataset

STEMS_DIR = "/mnt/ssd/scratch_benchmark/synthetic/stems"

LANGUAGES = [
    {"code": "id", "fleurs_config": "id_id", "name": "Indonesian"},
    {"code": "es", "fleurs_config": "es_419", "name": "Spanish"},
    {"code": "hi", "fleurs_config": "hi_in", "name": "Hindi"},
    {"code": "ar", "fleurs_config": "ar_eg", "name": "Arabic"},
    {"code": "th", "fleurs_config": "th_th", "name": "Thai"},
    {"code": "vi", "fleurs_config": "vi_vn", "name": "Vietnamese"},
    {"code": "zh", "fleurs_config": "cmn_hans_cn", "name": "Mandarin"},
    {"code": "ja", "fleurs_config": "ja_jp", "name": "Japanese"},
    {"code": "fr", "fleurs_config": "fr_fr", "name": "French"},
    {"code": "ms", "fleurs_config": "ms_my", "name": "Malay"},
]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("==================================================================")
    print("=== DOWNLOADING CONTROLLED FLEURS STEMS (10 LANGS x 10 UTTERANCES) ===")
    print("==================================================================")

    os.makedirs(STEMS_DIR, exist_ok=True)
    manifest_path = os.path.join(STEMS_DIR, "manifest.json")

    # Step 1: Fetch English reference translations from en_us
    print("\n[*] Step 1: Fetching English ground-truth sentences from FLEURS en_us...")
    en_refs = {}
    en_ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    en_ds = en_ds.cast_column("audio", datasets.Audio(decode=False))

    count_en = 0
    for sample in en_ds:
        s_id = sample.get("id")
        text = sample.get("transcription") or sample.get("raw_transcription")
        if s_id and text:
            en_refs[s_id] = text.strip()
            count_en += 1
        if count_en >= 100:
            break
    print(f"[+] Loaded {len(en_refs)} English reference sentences.")

    # Step 2: Fetch 10 utterances per foreign language
    manifest = []
    t0_all = time.time()

    for lang_info in LANGUAGES:
        code = lang_info["code"]
        config = lang_info["fleurs_config"]
        name = lang_info["name"]

        lang_dir = os.path.join(STEMS_DIR, code)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"\n[*] Fetching 10 stems for {name} ({config})...")
        t0_lang = time.time()
        try:
            ds = load_dataset("google/fleurs", config, split="test", streaming=True)
            ds = ds.cast_column("audio", datasets.Audio(decode=False))

            collected = 0
            for sample in ds:
                s_id = sample.get("id")
                # Ensure we have the English translation reference
                if s_id not in en_refs:
                    continue

                foreign_text = (sample.get("raw_transcription") or sample.get("transcription", "")).strip()
                if not foreign_text or len(foreign_text) < 5:
                    continue

                audio_bytes = sample["audio"]["bytes"]
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))

                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                # Ensure 16kHz
                if sr != 16000:
                    import scipy.signal
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
                    sr = 16000

                dur_sec = round(len(audio_data) / sr, 2)
                # Keep utterances between 1.5s and 8.0s (realistic chatter lengths)
                if dur_sec < 1.5 or dur_sec > 8.0:
                    continue

                out_wav = os.path.join(lang_dir, f"stem_{code}_{collected+1:02d}_{s_id}.wav")
                sf.write(out_wav, audio_data, sr)

                stem_meta = {
                    "lang_code": code,
                    "lang_name": name,
                    "sample_idx": collected + 1,
                    "fleurs_id": s_id,
                    "wav_file": out_wav,
                    "duration_sec": dur_sec,
                    "foreign_text": foreign_text,
                    "english_reference": en_refs[s_id],
                    "gender": sample.get("gender"),
                }
                manifest.append(stem_meta)
                collected += 1
                print(f"  [{collected}/10] ID={s_id} ({dur_sec}s): \"{foreign_text[:40]}...\" -> \"{en_refs[s_id][:40]}...\"")

                if collected >= 10:
                    break

            print(f"[+] Finished {name} in {time.time() - t0_lang:.1f}s.")

        except Exception as e:
            print(f"[-] Error fetching {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n==================================================================")
    print(f"[+] Download complete in {time.time() - t0_all:.1f}s!")
    print(f"[+] Total foreign stems downloaded: {len(manifest)}")
    print(f"[+] Saved manifest to: {manifest_path}")
    print("==================================================================")

if __name__ == "__main__":
    main()
