"""WhatTube Streaming Pipeline WebSocket Server."""

import asyncio
import json
import logging
import time
from typing import Set
import numpy as np
import websockets

from whattube.config import Config, default_config
from whattube.audio_buffer import AudioRingBuffer
from whattube.vad import EnergyAndSileroVAD
from whattube.lid import WhisperTinyLID
from whattube.event_aggregator import DynamicEventAggregator
from whattube.asr.client import ResidentASRClient
from whattube.translation.marian import MarianTranslator
from whattube.logger import EventAuditLogger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("WhatTube")

class WhatTubeServer:
    def __init__(self, config: Config = default_config):
        self.config = config
        self.buffer = AudioRingBuffer(
            sample_rate=config.sample_rate,
            capacity_sec=config.buffer_capacity_sec,
        )

        logger.info("[*] Initializing VAD...")
        self.vad = EnergyAndSileroVAD(
            model_path=config.silero_vad_onnx,
            energy_threshold=config.energy_threshold_rms,
            vad_threshold=config.vad_threshold,
        )

        logger.info("[*] Initializing Whisper Tiny INT8 LID...")
        self.lid = WhisperTinyLID(
            encoder_path=config.encoder_onnx,
            decoder_path=config.decoder_onnx,
            lid_suspicious=config.lid_suspicious,
            lid_immediate=config.lid_immediate,
            num_threads=4,
        )

        logger.info("[*] Initializing Event Aggregator...")
        self.aggregator = DynamicEventAggregator(
            pre_roll_sec=config.pre_roll_sec,
            post_roll_sec=config.post_roll_sec,
            close_hangover_sec=config.close_hangover_sec,
            min_event_sec=config.min_event_sec,
            max_event_sec=config.max_event_sec,
            lid_suspicious=config.lid_suspicious,
            lid_immediate=config.lid_immediate,
        )

        logger.info(f"[*] Initializing ASR Client -> {config.asr_endpoint}...")
        self.asr_client = ResidentASRClient(endpoint_url=config.asr_endpoint)

        logger.info("[*] Initializing CPU Marian Translator...")
        self.translator = MarianTranslator(device="cpu")
        # Pre-warm default target language model
        self.translator._load_model("id", config.target_language)

        self.audit_logger = EventAuditLogger(log_path=config.log_file)
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        self.last_stride_time = 0.0
        self.last_audio_rx_time = 0.0

    async def broadcast(self, message: dict):
        if not self.clients:
            return
        payload = json.dumps(message)
        to_remove = set()
        for ws in self.clients:
            try:
                await ws.send(payload)
            except Exception:
                to_remove.add(ws)
        self.clients.difference_update(to_remove)

    async def process_sliding_window(self):
        """Continuous Stage 1: Run every stride_sec (1.0s) over the last 3.0s."""
        cur_time = self.buffer.current_time_sec
        if cur_time - self.last_stride_time < self.config.stride_sec:
            return

        window_duration = self.config.analysis_window_sec
        if cur_time < window_duration:
            return

        t_start = cur_time - window_duration
        t_end = cur_time
        self.last_stride_time = cur_time

        audio_window = self.buffer.get_slice(t_start, t_end)
        if audio_window is None or len(audio_window) < int(window_duration * self.config.sample_rate):
            return

        # 1. VAD & Energy check (<2ms)
        t_vad_0 = time.perf_counter()
        is_speech, rms, vad_prob = self.vad.is_speech(audio_window)
        t_vad = (time.perf_counter() - t_vad_0) * 1000.0

        lid_res = None
        t_lid = 0.0
        if is_speech:
            # 2. Spoken Language ID (<55ms)
            t_lid_0 = time.perf_counter()
            lid_res = self.lid.predict(audio_window, sample_rate=self.config.sample_rate)
            t_lid = (time.perf_counter() - t_lid_0) * 1000.0

        # 3. Dynamic Aggregator state update
        event = self.aggregator.update(t_start, t_end, is_speech, lid_res)

        if event is not None:
            asyncio.create_task(self.handle_event_burst(event, t_vad + t_lid))

    async def background_tick(self):
        """Periodic background ticker to finalize timed-out events on pauses or stream ends."""
        while self.running:
            await asyncio.sleep(0.3)
            cur_time = self.buffer.current_time_sec
            if self.aggregator.is_active:
                event = self.aggregator.check_hangover_timeout(cur_time)
                if event is not None:
                    asyncio.create_task(self.handle_event_burst(event, 0.0))

    async def handle_event_burst(self, event, stage1_latency_ms: float):
        """Dispatches coalesced burst to GPU Turbo ASR and CPU Translator."""
        logger.info(
            f"[!] Event #{event.event_id} TRIGGERED: [{event.start_sec:.2f}s - {event.end_sec:.2f}s] "
            f"({event.duration_sec:.2f}s duration)"
        )

        event_audio = self.buffer.get_slice(event.start_sec, event.end_sec)
        if event_audio is None:
            logger.warning(f"[-] Could not extract audio slice for Event #{event.event_id}")
            return

        # 4. GPU Stage 2: Whisper Large-v3-Turbo
        loop = asyncio.get_running_loop()
        t_asr_0 = time.perf_counter()
        asr_res = await loop.run_in_executor(
            None, self.asr_client.transcribe, event_audio, self.config.sample_rate
        )
        t_asr = (time.perf_counter() - t_asr_0) * 1000.0

        trigger_reason = f"LID min p_en < {self.config.lid_suspicious}"

        # 5. Turbo Language Gate Filter
        if asr_res.is_discarded or asr_res.language.lower() == "en":
            logger.info(
                f"[x] Event #{event.event_id} DISCARDED: {asr_res.discard_reason or 'Classified as English'} "
                f"(lang={asr_res.language}, text='{asr_res.text}')"
            )
            self.audit_logger.log_event(
                event_id=event.event_id,
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                duration_sec=event.duration_sec,
                trigger_reason=trigger_reason,
                asr_lang=asr_res.language,
                original_text=asr_res.text,
                translated_text="",
                is_emitted=False,
                discard_reason=asr_res.discard_reason or "Classified as English",
                timings={"stage1_ms": stage1_latency_ms, "asr_ms": t_asr, "total_ms": stage1_latency_ms + t_asr},
            )
            return

        # 6. Stage 3: CPU Multilingual Translation
        t_trans_0 = time.perf_counter()
        trans_res = await loop.run_in_executor(
            None,
            self.translator.translate,
            asr_res.text,
            asr_res.language,
            self.config.target_language,
        )
        t_trans = (time.perf_counter() - t_trans_0) * 1000.0

        total_latency_ms = stage1_latency_ms + t_asr + t_trans
        logger.info(
            f"[+] EMITTING CAPTION #{event.event_id} ({asr_res.language.upper()} -> {self.config.target_language.upper()}): "
            f"'{trans_res.translated_text}' [orig: '{asr_res.text}'] ({total_latency_ms:.1f}ms total)"
        )

        caption_payload = {
            "type": "caption",
            "start": event.start_sec,
            "end": event.end_sec,
            "language": asr_res.language,
            "original": asr_res.text,
            "translation": trans_res.translated_text,
            "latency_ms": round(total_latency_ms, 1),
        }

        await self.broadcast(caption_payload)

        self.audit_logger.log_event(
            event_id=event.event_id,
            start_sec=event.start_sec,
            end_sec=event.end_sec,
            duration_sec=event.duration_sec,
            trigger_reason=trigger_reason,
            asr_lang=asr_res.language,
            original_text=asr_res.text,
            translated_text=trans_res.translated_text,
            is_emitted=True,
            discard_reason="",
            timings={
                "stage1_ms": stage1_latency_ms,
                "asr_ms": t_asr,
                "trans_ms": t_trans,
                "total_ms": total_latency_ms,
            },
        )

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        logger.info(f"[+] Client connected: {websocket.remote_address} (Total: {len(self.clients)})")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Raw PCM Float32 audio chunk
                    pcm_chunk = np.frombuffer(message, dtype=np.float32)
                    self.buffer.append(pcm_chunk)
                    self.last_audio_rx_time = time.time()
                    await self.process_sliding_window()
                else:
                    # JSON control message
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        if msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                        elif msg_type == "seek":
                            logger.info(f"[*] Player seek detected to {data.get('video_time', 0):.2f}s")
                            self.buffer.reset()
                        elif msg_type == "reset":
                            self.buffer.reset()
                    except json.JSONDecodeError:
                        pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"[-] Client disconnected (Remaining: {len(self.clients)})")
            # If an event is still pending upon client departure, finalize it
            if self.aggregator.is_active:
                event = self.aggregator.flush(self.buffer.current_time_sec)
                if event is not None:
                    asyncio.create_task(self.handle_event_burst(event, 0.0))

    async def start(self):
        self.running = True
        logger.info(f"[+] Starting WhatTube WebSocket Server on {self.config.ws_host}:{self.config.ws_port}")
        ticker_task = asyncio.create_task(self.background_tick())
        try:
            async with websockets.serve(self.handle_client, self.config.ws_host, self.config.ws_port, max_size=10_000_000):
                while self.running:
                    await asyncio.sleep(0.05)
        finally:
            ticker_task.cancel()

def main():
    server = WhatTubeServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        server.audit_logger.print_summary()
        print("\nWhatTube server stopped cleanly.")

if __name__ == "__main__":
    main()
