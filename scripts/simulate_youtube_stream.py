"""Simulate live YouTube tab audio streaming to WhatTube WebSocket server."""

import asyncio
import json
import time
import argparse
import numpy as np
import soundfile as sf
import websockets

async def simulate_stream(wav_path: str, start_sec: float, duration_sec: float, ws_url: str = "ws://localhost:8765"):
    print(f"\n=== SIMULATING YOUTUBE AUDIO PLAYBACK ===")
    print(f"Audio file: {wav_path}")
    print(f"Slice: {start_sec:.2f}s -> {start_sec + duration_sec:.2f}s ({duration_sec:.2f}s)")
    print(f"Target WebSocket: {ws_url}\n")

    audio_full, sr = sf.read(wav_path, dtype="float32")
    if audio_full.ndim > 1:
        audio_full = audio_full.mean(axis=1)

    start_samp = int(start_sec * sr)
    end_samp = int((start_sec + duration_sec) * sr)
    audio = audio_full[start_samp:end_samp]

    captions_received = []

    async with websockets.connect(ws_url) as ws:
        print("[+] Connected to WhatTube WebSocket server!")

        async def listen_for_captions():
            try:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "caption":
                        arrival_time = time.time()
                        captions_received.append((arrival_time, data))
                        print(f"\n==========================================")
                        print(f"⚡ [SUBTITLE RECEIVED] +{arrival_time - t_start_stream:.2f}s relative to start")
                        print(f"   Time Window:  [{data['start']:.2f}s - {data['end']:.2f}s]")
                        print(f"   Language:     {data['language'].upper()}")
                        print(f"   Original:     \"{data['original']}\"")
                        print(f"   Translation:  \"{data['translation']}\"")
                        print(f"   Pipeline RT:  {data.get('latency_ms', 0):.1f}ms")
                        print(f"==========================================\n")
            except websockets.ConnectionClosed:
                pass

        listen_task = asyncio.create_task(listen_for_captions())

        chunk_size = 4096
        chunk_duration = chunk_size / sr
        total_chunks = len(audio) // chunk_size

        t_start_stream = time.time()
        print(f"[*] Streaming {len(audio)/sr:.2f}s of audio in {total_chunks} chunks ({chunk_size} samples / {chunk_duration*1000:.1f}ms per chunk)...")

        for i in range(total_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            chunk_bytes = chunk.tobytes()
            await ws.send(chunk_bytes)
            await asyncio.sleep(chunk_duration)

        print("[*] Stream complete. Waiting 3.0s for any trailing event aggregations...")
        await asyncio.sleep(3.0)
        listen_task.cancel()

    print(f"\n[+] Total foreign captions captured: {len(captions_received)}")
    return captions_received

def main():
    parser = argparse.ArgumentParser(description="WhatTube Live Stream Simulator")
    parser.add_argument("--wav", default="/mnt/ssd/hermes/cache/scratch/seal-test-P13/P13mMiIL_2I_full_16k.wav")
    parser.add_argument("--start", type=float, default=820.0, help="Start offset in seconds (13:40 = 820s)")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration to stream in seconds")
    parser.add_argument("--ws", default="ws://localhost:8765")
    args = parser.parse_args()

    asyncio.run(simulate_stream(args.wav, args.start, args.duration, args.ws))

if __name__ == "__main__":
    main()
