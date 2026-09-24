"""Resident ASR Microservice for persistent GPU Whisper (large-v3-turbo)."""

import io
import time
import argparse
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI(title="WhatTube Resident ASR Daemon")

# Global state
processor = None
model = None
device = "cpu"

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@app.on_event("startup")
def load_model():
    global processor, model, device
    device = get_device()
    print(f"[*] Initializing WhatTube ASR Daemon on target device: {device}")

    model_id = "openai/whisper-large-v3-turbo"
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(model_id)

    dtype = torch.float16 if device in ["xpu", "cuda"] else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    # Warmup dummy pass
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
    with torch.no_grad():
        _ = model.generate(inputs, max_new_tokens=16)

    print(f"[+] Model resident in {device.upper()} memory in {time.time() - t0:.2f} s")

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

    # Audio input can be raw bytes (PCM float32/int16) or WAV file bytes
    body = await request.body()
    if not body:
        return JSONResponse({"error": "Empty audio payload"}, status_code=400)

    try:
        # Try reading as WAV container
        with io.BytesIO(body) as bio:
            audio, sr = sf.read(bio, dtype="float32")
    except Exception:
        # Fallback to raw float32 LE
        audio = np.frombuffer(body, dtype=np.float32)
        sr = 16000

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Pad or enforce minimal samples
    if len(audio) < 1600:
        return JSONResponse({
            "language": "unknown",
            "text": "",
            "latency_ms": 0.0,
            "is_discarded": True,
            "discard_reason": "Audio chunk too short (<100ms)",
        })

    dtype = torch.float16 if device in ["xpu", "cuda"] else torch.float32
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)

    with torch.no_grad():
        gen_out = model.generate(
            inputs,
            max_new_tokens=64,
            return_dict_in_generate=True,
        )

    seq = gen_out.sequences[0]
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Extract language token (index 1: <|startoftranscript|>, <|lang|>, ...)
    lang_token_id = int(seq[1].item()) if len(seq) > 1 else None
    raw_lang = processor.tokenizer.decode([lang_token_id]).strip() if lang_token_id else ""
    detected_lang = raw_lang.replace("<|", "").replace("|>", "").strip()

    # Extract text (skip special tokens)
    decoded_text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()

    # Output filtering
    is_discarded = False
    discard_reason = ""

    if detected_lang.lower() == "en":
        is_discarded = True
        discard_reason = "Turbo ASR classified language as English"
    elif len(decoded_text) == 0 or decoded_text in [".", "...", "!", "?", "♪", "[music]"]:
        is_discarded = True
        discard_reason = "Empty or hallucinated punctuation"

    return {
        "language": detected_lang,
        "text": decoded_text,
        "latency_ms": round(latency_ms, 2),
        "is_discarded": is_discarded,
        "discard_reason": discard_reason,
    }

def main():
    parser = argparse.ArgumentParser(description="WhatTube Resident ASR Daemon")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8766, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
