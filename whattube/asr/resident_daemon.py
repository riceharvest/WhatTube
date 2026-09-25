"""Resident ASR Microservice for persistent GPU Whisper (large-v3-turbo)."""

import io
import time
import asyncio
import argparse
from math import gcd
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from contextlib import asynccontextmanager

# Global state
processor = None
model = None
device = "cpu"
dtype = torch.float32
lang_tokens_list = []
lang_token_ids = []
inference_lock = asyncio.Lock()
MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB limit (~5 minutes of 16kHz float32)

def get_device() -> str:
    """Auto-detect best available compute device across Linux, macOS, and Windows."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device, dtype, lang_tokens_list, lang_token_ids
    device = get_device()
    print(f"[*] Initializing WhatTube ASR Daemon on target device: {device}")

    model_id = "openai/whisper-large-v3-turbo"
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(model_id)

    # Build language token list for language probability extraction
    vocab = processor.tokenizer.get_vocab()
    lang_tokens_list = [
        k for k in vocab
        if k.startswith("<|") and k.endswith("|>") and len(k) <= 6
        and k not in ["<|startoftranscript|>", "<|endoftranscript|>", "<|notimestamps|>", "<|translate|>", "<|transcribe|>"]
    ]
    lang_token_ids = [vocab[k] for k in lang_tokens_list]

    dtype = torch.float16 if device in ["xpu", "cuda"] else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.max_length = None

    # Warmup dummy pass
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
    with torch.no_grad():
        _ = model.generate(inputs, max_new_tokens=8)

    print(f"[+] Model resident in {device.upper()} memory in {time.time() - t0:.2f} s")
    yield

app = FastAPI(title="WhatTube Resident ASR Daemon", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": device,
        "model": "openai/whisper-large-v3-turbo",
    }

@app.post("/transcribe")
async def transcribe(request: Request):
    t0 = time.perf_counter()

    body = await request.body()
    if not body:
        return JSONResponse({"error": "Empty audio payload"}, status_code=400)
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"error": "Payload exceeds maximum allowed size (10 MB)"}, status_code=413)

    # Decode audio container or raw float32
    audio = None
    try:
        with io.BytesIO(body) as bio:
            audio, sr = sf.read(bio, dtype="float32")
        if sr != 16000:
            from scipy.signal import resample_poly
            g = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            sr = 16000
    except Exception:
        # Fallback to raw float32 LE if valid
        if len(body) % 4 != 0:
            return JSONResponse({"error": "Malformed audio payload: unrecognized container and length not divisible by 4"}, status_code=400)
        audio = np.frombuffer(body, dtype=np.float32)
        sr = 16000
        if not np.all(np.isfinite(audio)) or (len(audio) > 0 and np.max(np.abs(audio)) > 10.0):
            return JSONResponse({"error": "Malformed audio payload: raw PCM contains NaN, Inf, or out-of-range samples"}, status_code=400)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Pad or enforce minimal samples (>100ms)
    if len(audio) < 1600:
        return JSONResponse({
            "language": "unknown",
            "language_prob": 0.0,
            "text": "",
            "latency_ms": 0.0,
            "is_discarded": True,
            "discard_reason": "Audio chunk too short (<100ms)",
        })

    # Inference lock guarantees exactly 1 inference execution at a time per GPU process
    async with inference_lock:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)

        with torch.no_grad():
            gen_out = model.generate(
                inputs,
                max_new_tokens=64,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = gen_out.sequences[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Extract language token and probability
        lang_token_id = int(seq[1].item()) if len(seq) > 1 else None
        raw_lang = processor.tokenizer.decode([lang_token_id]).strip() if lang_token_id else ""
        detected_lang = raw_lang.replace("<|", "").replace("|>", "").strip()

        language_prob = 1.0
        if hasattr(gen_out, "scores") and len(gen_out.scores) > 0 and lang_token_ids:
            try:
                lang_logits = gen_out.scores[0][0, lang_token_ids]
                lang_probs = torch.softmax(lang_logits.float(), dim=-1)
                token_tag = f"<|{detected_lang}|>"
                if token_tag in lang_tokens_list:
                    idx = lang_tokens_list.index(token_tag)
                    language_prob = round(float(lang_probs[idx].item()), 4)
            except Exception:
                language_prob = 1.0

        # Extract text (skip special tokens)
        decoded_text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()

        # Output filtering
        is_discarded = False
        discard_reason = ""

        noise_hallucinations = {
            "", ".", "...", "!", "?", "♪", "[music]",
            "thank you", "thank you.", "thank you for watching", "thank you for watching.",
            "please subscribe", "subscribe", "subtitles by",
            "реклама", "реклама.", "подпишись", "advertisement", "promotion",
        }
        clean_lower = decoded_text.lower().strip()
        stripped_word = clean_lower.strip("[]().,!?:;-\"'/~")
        acoustic_descriptions = {
            "music", "musik", "musica", "musique", "applause", "cheering",
            "laughter", "chatter", "silence", "реклама", "advertising",
        }

        if detected_lang.lower() == "en":
            is_discarded = True
            discard_reason = "Turbo ASR classified language as English"
        elif language_prob < 0.15:
            is_discarded = True
            discard_reason = f"Low language confidence ({language_prob:.2f} < 0.15) in background noise"
        elif (
            clean_lower in noise_hallucinations
            or stripped_word in acoustic_descriptions
            or any(clean_lower.startswith(h) for h in ["subtitles by", "translated by", "thank you for watching"])
        ):
            is_discarded = True
            discard_reason = f"Noise hallucination filter: '{decoded_text}'"
        elif len(clean_lower) <= 2 or clean_lower.isdigit():
            is_discarded = True
            discard_reason = f"Single token or digit noise: '{decoded_text}'"

        return {
            "language": detected_lang,
            "language_prob": language_prob,
            "text": decoded_text,
            "latency_ms": round(latency_ms, 2),
            "is_discarded": is_discarded,
            "discard_reason": discard_reason,
        }

def main():
    parser = argparse.ArgumentParser(description="WhatTube Resident ASR Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Bind port (default: 8766)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
