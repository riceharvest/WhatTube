"""WhatTube Multi-Session Streaming Pipeline WebSocket Server."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import numpy as np
import websockets
from websockets.server import ServerConnection

from whattube.asr.client import ResidentASRClient
from whattube.audio_buffer import AudioRingBuffer
from whattube.config import Config, default_config
from whattube.event_aggregator import DynamicEventAggregator
from whattube.lid import WhisperTinyLID
from whattube.logger import EventAuditLogger
from whattube.translation.marian import MarianTranslator
from whattube.vad import EnergyAndSileroVAD

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("WhatTube")

TITLE_KEYWORD_LANG_MAP = {
    "japan": "ja", "japanese": "ja", "tokyo": "ja", "osaka": "ja", "kyoto": "ja", "ramen": "ja", "sushi": "ja",
    "indonesia": "id", "indonesian": "id", "jakarta": "id", "bali": "id",
    "bangladesh": "bn", "bengali": "bn", "dhaka": "bn", "bangla": "bn",
    "thailand": "th", "thai": "th", "bangkok": "th", "phuket": "th", "chiang mai": "th",
    "korea": "ko", "korean": "ko", "seoul": "ko", "busan": "ko",
    "china": "zh", "chinese": "zh", "taiwan": "zh", "beijing": "zh", "shanghai": "zh", "hong kong": "zh", "cantonese": "zh", "taipei": "zh",
    "france": "fr", "french": "fr", "paris": "fr", "nice": "fr", "lyon": "fr",
    "spain": "es", "spanish": "es", "mexico": "es", "colombia": "es", "madrid": "es", "barcelona": "es", "oaxaca": "es", "cancun": "es", "argentina": "es", "buenos aires": "es", "peru": "es", "lima": "es", "medellin": "es",
    "italy": "it", "italian": "it", "rome": "it", "milan": "it", "naples": "it", "venice": "it", "florence": "it", "sicily": "it",
    "germany": "de", "german": "de", "berlin": "de", "munich": "de", "vienna": "de", "austria": "de", "zurich": "de", "swiss": "de",
    "vietnam": "vi", "vietnamese": "vi", "hanoi": "vi", "saigon": "vi", "da nang": "vi",
    "india": "hi", "hindi": "hi", "mumbai": "hi", "delhi": "hi", "varanasi": "hi", "kolkata": "hi", "jaipur": "hi", "indian": "hi",
    "arab": "ar", "arabic": "ar", "egypt": "ar", "cairo": "ar", "dubai": "ar", "morocco": "ar", "marrakech": "ar", "casablanca": "ar", "jordan": "ar", "amman": "ar",
    "russia": "ru", "russian": "ru", "moscow": "ru",
    "poland": "pl", "polish": "pl", "warsaw": "pl", "krakow": "pl",
    "turkey": "tr", "turkish": "tr", "istanbul": "tr", "antalya": "tr", "ankara": "tr",
    "brazil": "pt", "brazilian": "pt", "portugal": "pt", "portuguese": "pt", "rio": "pt", "sao paulo": "pt", "lisbon": "pt",
    "philippines": "tl", "filipino": "tl", "tagalog": "tl", "manila": "tl", "cebu": "tl",
    "greece": "el", "greek": "el", "athens": "el", "santorini": "el",
}

def extract_language_hints_from_title(title: str) -> set[str]:
    hints = set()
    if not title:
        return hints
    clean = title.lower()
    for ch in [",", ".", "!", "?", "-", "_", "/", "|", "[", "]", "(", ")", ":", "#"]:
        clean = clean.replace(ch, " ")
    tokens = set(clean.split())
    for kw, lang in TITLE_KEYWORD_LANG_MAP.items():
        if kw in tokens or (len(kw) >= 5 and kw in clean):
            hints.add(lang)
    return hints

class SessionState:
    """Isolated state machine per connected browser tab."""

    def __init__(
        self,
        session_id: str,
        websocket: ServerConnection,
        config: Config,
        target_lang: str = "en",
        initial_video_time: float = 0.0,
        initial_playback_rate: float = 1.0,
        video_title: str = "",
    ):
        self.session_id = session_id
        self.websocket = websocket
        self.config = config
        self.target_lang = target_lang
        self.epoch = 0

        self.ring_buffer = AudioRingBuffer(
            sample_rate=config.sample_rate,
            capacity_sec=config.buffer_capacity_sec,
        )
        self.vad = EnergyAndSileroVAD(
            model_path=config.silero_vad_onnx,
            energy_threshold=config.energy_threshold_rms,
            vad_threshold=config.vad_threshold,
        )
        self.lid = WhisperTinyLID(
            encoder_path=config.encoder_onnx,
            decoder_path=config.decoder_onnx,
            lid_suspicious=config.lid_suspicious,
            lid_immediate=config.lid_immediate,
            num_threads=4,
        )
        self.aggregator = DynamicEventAggregator(
            pre_roll_sec=config.pre_roll_sec,
            post_roll_sec=config.post_roll_sec,
            close_hangover_sec=config.close_hangover_sec,
            min_event_sec=config.min_event_sec,
            max_event_sec=config.max_event_sec,
            lid_suspicious=config.lid_suspicious,
            lid_immediate=config.lid_immediate,
            lid_english_safe=config.lid_english_safe,
            consecutive_suspicious_req=config.consecutive_suspicious_req,
        )

        # Timeline anchor mapping:
        # video_time = anchor_video_time + (audio_time - anchor_audio_time) * playback_rate
        self.anchor_audio_time = 0.0
        self.anchor_video_time = initial_video_time
        self.playback_rate = max(0.1, initial_playback_rate)

        self.video_title = video_title
        self.title_hints = extract_language_hints_from_title(video_title)
        self.last_stride_time = 0.0
        self.last_emitted_text = ""
        self.last_emitted_time = 0.0
        self.last_emitted_lang = ""
        self.is_closed = False
        self.active_tasks: set[asyncio.Task] = set()

    def update_video_title(self, title: str):
        """Update video title and refresh language domain priors."""
        if title and title != self.video_title:
            self.video_title = title
            self.title_hints = extract_language_hints_from_title(title)
            logger.info(f"[*] [Session {self.session_id[:6]}] Video title updated: '{title}' -> Language hints: {self.title_hints}")

    def set_anchor(self, audio_time: float, video_time: float, playback_rate: float | None = None):
        """Re-anchors the audio-to-video timeline mapping."""
        self.anchor_audio_time = audio_time
        self.anchor_video_time = video_time
        if playback_rate is not None and playback_rate > 0.0:
            self.playback_rate = playback_rate

    def audio_to_video_time(self, audio_sec: float) -> float:
        """Converts audio buffer timestamp to exact video player timeline position."""
        return self.anchor_video_time + (audio_sec - self.anchor_audio_time) * self.playback_rate

    def increment_epoch(self, new_video_time: float | None = None, playback_rate: float | None = None):
        """Invalidates all in-flight ASR/translation tasks, resets temporal buffer and re-anchors."""
        self.epoch += 1
        self.ring_buffer.reset()
        self.aggregator.reset()
        self.last_stride_time = 0.0
        self.last_emitted_text = ""

        # Re-anchor to new video position with audio_time = 0.0
        rate = playback_rate if (playback_rate is not None and playback_rate > 0.0) else self.playback_rate
        vtime = new_video_time if new_video_time is not None else self.anchor_video_time
        self.set_anchor(audio_time=0.0, video_time=vtime, playback_rate=rate)

        # Cancel all pending burst tasks belonging to older epoch
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        self.active_tasks.clear()

class WhatTubeServer:
    def __init__(self, config: Config = default_config):
        self.config = config
        self.sessions: dict[str, SessionState] = {}
        self.ws_to_session: dict[Any, str] = {}

        logger.info(f"[*] Initializing shared ASR Client -> {config.asr_endpoint}...")
        self.asr_client = ResidentASRClient(endpoint_url=config.asr_endpoint)
        # Bounded GPU queue: guarantees at most 1 inference execution at a time across all sessions
        self.asr_semaphore = asyncio.Semaphore(1)

        logger.info("[*] Initializing shared CPU Marian Translator (CTranslate2 INT8)...")
        self.translator = MarianTranslator(device="cpu", use_ct2=True)
        # Prefetch default target language
        self.translator.prefetch("id", config.target_language)

        self.audit_logger = EventAuditLogger(
            log_path=config.log_file,
            enabled=config.enable_transcript_log,
        )
        self.running = False

    async def process_sliding_window(self, session: SessionState):
        """Runs continuous Stage 1 VAD + LID over the session's isolated buffer."""
        cur_time = session.ring_buffer.current_time_sec
        if cur_time - session.last_stride_time < self.config.stride_sec:
            return

        window_duration = self.config.analysis_window_sec
        if cur_time < window_duration:
            return

        t_start = cur_time - window_duration
        t_end = cur_time
        session.last_stride_time = cur_time

        audio_window = session.ring_buffer.get_slice(t_start, t_end)
        if audio_window is None or len(audio_window) < int(window_duration * self.config.sample_rate):
            return

        # 1. VAD & Energy check (<2ms)
        t_vad_0 = time.perf_counter()
        is_speech, _rms, _vad_prob = session.vad.is_speech(audio_window)
        t_vad = (time.perf_counter() - t_vad_0) * 1000.0

        lid_res = None
        t_lid = 0.0
        if is_speech:
            # 2. Spoken Language ID (<55ms)
            t_lid_0 = time.perf_counter()
            lid_res = session.lid.predict(audio_window, sample_rate=self.config.sample_rate)
            t_lid = (time.perf_counter() - t_lid_0) * 1000.0

        # 3. Dynamic Aggregator state update
        event = session.aggregator.update(t_start, t_end, is_speech, lid_res)

        if event is not None:
            task = asyncio.create_task(
                self.handle_event_burst(session, event, session.epoch, t_vad + t_lid)
            )
            session.active_tasks.add(task)
            task.add_done_callback(session.active_tasks.discard)

    async def handle_event_burst(
        self,
        session: SessionState,
        event,
        event_epoch: int,
        stage1_latency_ms: float,
    ):
        """Processes an event burst through acoustic breath splitting, GPU ASR, and translation."""
        if session.epoch != event_epoch or session.is_closed:
            return

        logger.info(
            f"[!] [Session {session.session_id[:6]}] Event #{event.event_id} TRIGGERED: "
            f"[{event.start_sec:.2f}s - {event.end_sec:.2f}s] ({event.duration_sec:.2f}s, epoch={event_epoch})"
        )

        event_audio = session.ring_buffer.get_slice(event.start_sec, event.end_sec)
        if event_audio is None:
            return

        # Acoustic breath-pause splitting: decouples preceding English from foreign chatter
        sub_slices = []
        if event.duration_sec >= 5.0:
            split_sec = session.vad.find_split_point(event_audio, sample_rate=self.config.sample_rate)
            if split_sec is not None:
                split_idx = int(split_sec * self.config.sample_rate)
                sub_slices.append((event.start_sec, event.start_sec + split_sec, event_audio[:split_idx]))
                sub_slices.append((event.start_sec + split_sec, event.end_sec, event_audio[split_idx:]))

        if not sub_slices:
            sub_slices.append((event.start_sec, event.end_sec, event_audio))

        loop = asyncio.get_running_loop()
        trigger_reason = f"LID min p_en < {self.config.lid_suspicious}"

        for sub_start, sub_end, sub_audio in sub_slices:
            if session.epoch != event_epoch or session.is_closed:
                logger.info(f"[*] [Session {session.session_id[:6]}] Stale event discarded before ASR (epoch changed).")
                return

            # Shared GPU queue scheduler: exactly 1 inference at a time across all tabs
            async with self.asr_semaphore:
                # Check epoch again after acquiring semaphore
                if session.epoch != event_epoch or session.is_closed:
                    return

                t_asr_0 = time.perf_counter()
                asr_res = await loop.run_in_executor(
                    None, self.asr_client.transcribe, sub_audio, self.config.sample_rate
                )
                t_asr = (time.perf_counter() - t_asr_0) * 1000.0

            if session.epoch != event_epoch or session.is_closed:
                logger.info(f"[*] [Session {session.session_id[:6]}] Stale ASR result discarded (epoch changed).")
                return

            # Turbo Language Gate Filter
            if asr_res.is_discarded or asr_res.language.lower() == "en":
                logger.info(
                    f"[x] [Session {session.session_id[:6]}] Sub-Event [{sub_start:.2f}s - {sub_end:.2f}s] DISCARDED: "
                    f"{asr_res.discard_reason or 'Classified as English'} (lang={asr_res.language}, text='{asr_res.text}')"
                )
                self.audit_logger.log_event(
                    event_id=event.event_id,
                    start_sec=sub_start,
                    end_sec=sub_end,
                    duration_sec=sub_end - sub_start,
                    trigger_reason=trigger_reason,
                    asr_lang=asr_res.language,
                    original_text=asr_res.text,
                    translated_text="",
                    is_emitted=False,
                    discard_reason=asr_res.discard_reason or "Classified as English",
                    timings={"stage1_ms": stage1_latency_ms, "asr_ms": t_asr, "total_ms": stage1_latency_ms + t_asr},
                )
                continue

            # Bayesian Confidence & Out-of-Domain Filter:
            # - In-domain/local speech (matching title hints): threshold relaxed to p >= 0.15 to capture noisy street chatter
            # - Out-of-domain speech with title hints: threshold tightened to p >= 0.55 to reject acoustic hallucinations
            # - No title hints available: default p >= 0.25
            lang_code = asr_res.language.lower()
            lang_prob = getattr(asr_res, "language_prob", 1.0)
            if isinstance(lang_prob, (int, float)):
                # Only languages directly matching title hints get the relaxed local prior (p >= 0.15).
                # All other languages in foreign destination require high confidence (p >= 0.55).
                is_local = bool(session.title_hints and lang_code in session.title_hints)
                min_prob = 0.15 if is_local else (0.55 if session.title_hints else 0.25)
                if lang_prob < min_prob:
                    discard_msg = (
                        f"Confidence gate: low-prob {lang_code.upper()} ({lang_prob:.2f} < {min_prob:.2f}) "
                        f"{'inconsistent with' if not is_local and session.title_hints else 'below confidence floor for'} title hints {session.title_hints}"
                    )
                    logger.info(f"[*] [Session {session.session_id[:6]}] DISCARDED: {discard_msg}")
                    self.audit_logger.log_event(
                        event_id=event.event_id,
                        start_sec=sub_start,
                        end_sec=sub_end,
                        duration_sec=sub_end - sub_start,
                        trigger_reason=trigger_reason,
                        asr_lang=asr_res.language,
                        original_text=asr_res.text,
                        translated_text="",
                        is_emitted=False,
                        discard_reason=discard_msg,
                        timings={"stage1_ms": stage1_latency_ms, "asr_ms": t_asr, "total_ms": stage1_latency_ms + t_asr},
                    )
                    continue

            # Multi-window consensus & deduplication filter
            clean_curr = asr_res.text.strip().lower()
            clean_last = session.last_emitted_text.strip().lower()
            time_diff = abs(sub_start - session.last_emitted_time)

            curr_words = set(clean_curr.replace(".", "").replace(",", "").split())
            last_words = set(clean_last.replace(".", "").replace(",", "").split())
            overlap = len(curr_words & last_words) / max(len(curr_words), 1) if last_words else 0.0

            if (clean_curr == clean_last or overlap >= 0.8) and time_diff < 3.5 and len(clean_curr) <= len(clean_last):
                logger.info(f"[*] Multi-window deduplication: skipping near-duplicate '{asr_res.text}'")
                continue

            if (clean_curr in clean_last or overlap >= 0.6) and len(clean_curr) < len(clean_last) and time_diff < 3.5:
                logger.info(f"[*] Multi-window deduplication: skipping sub-fragment '{asr_res.text}'")
                continue

            is_extension = (
                clean_last != "" and
                (clean_last in clean_curr or overlap >= 0.4) and
                len(clean_curr) > len(clean_last) and
                time_diff < 3.5
            )

            # Stage 3: CPU Multilingual Translation (CTranslate2 INT8, ~50ms)
            t_trans_0 = time.perf_counter()
            trans_res = await loop.run_in_executor(
                None,
                self.translator.translate,
                asr_res.text,
                asr_res.language,
                session.target_lang,
            )
            t_trans = (time.perf_counter() - t_trans_0) * 1000.0
            total_latency_ms = stage1_latency_ms + t_asr + t_trans

            if session.epoch != event_epoch or session.is_closed:
                return

            video_start = round(session.audio_to_video_time(sub_start), 2)
            video_end = round(session.audio_to_video_time(sub_end), 2)

            session.last_emitted_text = asr_res.text
            session.last_emitted_time = sub_start
            session.last_emitted_lang = asr_res.language

            action_type = "update" if is_extension else "new"
            logger.info(
                f"[+] [Session {session.session_id[:6]}] EMITTING CAPTION [{action_type.upper()}] "
                f"({asr_res.language.upper()} -> {session.target_lang.upper()}): "
                f"[{video_start}s - {video_end}s] '{trans_res.translated_text}' [orig: '{asr_res.text}'] ({total_latency_ms:.1f}ms total)"
            )

            caption_payload = {
                "type": "caption",
                "action": action_type,
                "epoch": event_epoch,
                "start": video_start,
                "end": video_end,
                "language": asr_res.language,
                "original": asr_res.text,
                "translation": trans_res.translated_text,
                "latency_ms": round(total_latency_ms, 1),
            }

            # ROUTE PRIVATELY TO THIS SESSION'S TAB ONLY (NO GLOBAL BROADCAST)
            try:
                await session.websocket.send(json.dumps(caption_payload))
            except (websockets.ConnectionClosed, OSError, RuntimeError) as e:
                logger.warning(f"[-] Failed to deliver caption to session {session.session_id[:6]}: {e}")

            self.audit_logger.log_event(
                event_id=event.event_id,
                start_sec=sub_start,
                end_sec=sub_end,
                duration_sec=sub_end - sub_start,
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

    async def background_tick(self):
        """Periodic background ticker to finalize hanging events across all active sessions."""
        while self.running:
            await asyncio.sleep(0.3)
            for session in list(self.sessions.values()):
                if session.aggregator.is_active:
                    cur_time = session.ring_buffer.current_time_sec
                    event = session.aggregator.check_hangover_timeout(cur_time)
                    if event is not None:
                        task = asyncio.create_task(
                            self.handle_event_burst(session, event, session.epoch, 0.0)
                        )
                        session.active_tasks.add(task)
                        task.add_done_callback(session.active_tasks.discard)

    async def handle_client(self, websocket):
        # 1. Origin header security check
        origin = None
        if hasattr(websocket, "request") and websocket.request:
            origin = websocket.request.headers.get("origin")
        elif hasattr(websocket, "request_headers"):
            origin = websocket.request_headers.get("origin")

        if origin:
            origin_lower = origin.lower()
            allowed_prefixes = (
                "chrome-extension://",
                "moz-extension://",
                "http://localhost",
                "http://127.0.0.1",
                "https://localhost",
                "https://127.0.0.1",
            )
            if origin_lower != "null" and not any(origin_lower.startswith(p) for p in allowed_prefixes):
                logger.warning(f"[-] Rejected connection from untrusted origin: {origin}")
                await websocket.close(code=4403, reason="Forbidden origin")
                return

        session: SessionState | None = None
        session_id: str | None = None

        try:
            # 2. STRICT AUTH & INIT PHASE: Only an 'init' message with valid auth token is accepted
            try:
                first_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("[-] Client timed out waiting for init handshake")
                await websocket.close(code=4408, reason="Handshake timeout")
                return

            if isinstance(first_msg, bytes):
                logger.warning("[-] Client sent binary PCM before init/auth handshake")
                await websocket.close(code=4403, reason="Binary data before init")
                return

            try:
                init_data = json.loads(first_msg)
            except (json.JSONDecodeError, UnicodeDecodeError):
                await websocket.close(code=4400, reason="Invalid JSON handshake")
                return

            if init_data.get("type") != "init":
                logger.warning(f"[-] First message was '{init_data.get('type')}', expected 'init'")
                await websocket.close(code=4400, reason="First message must be init")
                return

            # Validate auth token
            token = init_data.get("token")
            if self.config.auth_token and (not token or token != self.config.auth_token):
                logger.warning("[-] Client failed auth token verification")
                await websocket.send(json.dumps({"type": "error", "error": "Unauthorized"}))
                await websocket.close(code=4401, reason="Unauthorized")
                return

            # 3. Authenticated: Construct SessionState now (expensive VAD/LID only created once auth passes)
            session_id = str(uuid.uuid4())
            client_target = init_data.get("target_lang", self.config.target_language)
            vtime = float(init_data.get("video_time", 0.0))
            prate = float(init_data.get("playback_rate", 1.0))

            session = SessionState(
                session_id=session_id,
                websocket=websocket,
                config=self.config,
                target_lang=client_target.lower() if client_target else "en",
                initial_video_time=vtime,
                initial_playback_rate=prate,
                video_title=str(init_data.get("video_title", "")),
            )
            self.sessions[session_id] = session
            self.ws_to_session[websocket] = session_id
            logger.info(
                f"[+] Client authenticated & connected: session={session_id[:8]} "
                f"(target_lang={session.target_lang}, video_time={vtime:.2f}s, rate={session.playback_rate}x, "
                f"title='{session.video_title}', hints={session.title_hints}, total active: {len(self.sessions)})"
            )
            await websocket.send(json.dumps({"type": "ready", "session_id": session_id}))

            # 4. Main message loop
            async for message in websocket:
                if isinstance(message, bytes):
                    if len(message) > 1024 * 1024:  # 1MB max chunk limit
                        logger.warning(f"[-] Dropping oversized PCM chunk ({len(message)} bytes)")
                        continue
                    pcm_chunk = np.frombuffer(message, dtype=np.float32)
                    session.ring_buffer.append(pcm_chunk)
                    session.last_audio_rx_time = time.time()
                    await self.process_sliding_window(session)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "sync":
                            vtime = float(data.get("video_time", 0.0))
                            prate = float(data.get("playback_rate", session.playback_rate))
                            vtitle = data.get("video_title")
                            if vtitle:
                                session.update_video_title(vtitle)
                            cur_audio = session.ring_buffer.current_time_sec
                            session.set_anchor(audio_time=cur_audio, video_time=vtime, playback_rate=prate)

                        elif msg_type == "seek":
                            vtime = float(data.get("video_time", 0.0))
                            prate = float(data.get("playback_rate", session.playback_rate))
                            vtitle = data.get("video_title")
                            if vtitle:
                                session.update_video_title(vtitle)
                            logger.info(f"[*] [Session {session_id[:8]}] Seek/navigate to {vtime:.2f}s (new epoch {session.epoch + 1}, title='{vtitle}')")
                            session.increment_epoch(new_video_time=vtime, playback_rate=prate)

                        elif msg_type == "ratechange":
                            vtime = float(data.get("video_time", session.anchor_video_time))
                            prate = float(data.get("playback_rate", 1.0))
                            cur_audio = session.ring_buffer.current_time_sec
                            session.set_anchor(audio_time=cur_audio, video_time=vtime, playback_rate=prate)
                            logger.info(f"[*] [Session {session_id[:8]}] Playback rate changed to {prate}x at video_time={vtime:.2f}s")

                        elif msg_type == "set_target_lang":
                            new_lang = data.get("target_lang", "en").lower()
                            session.target_lang = new_lang
                            logger.info(f"[*] [Session {session_id[:8]}] Target language changed to: {new_lang}")

                        elif msg_type == "reset":
                            session.increment_epoch()

                        elif msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))

                    except json.JSONDecodeError:
                        pass
        except websockets.ConnectionClosed:
            pass
        finally:
            if session:
                session.is_closed = True
                session.increment_epoch()
                self.sessions.pop(session_id, None)
                self.ws_to_session.pop(websocket, None)
                logger.info(f"[-] Session {session_id[:8]} disconnected (Remaining sessions: {len(self.sessions)})")

    async def start(self):
        self.running = True
        logger.info(f"[+] Starting WhatTube WebSocket Server on {self.config.ws_host}:{self.config.ws_port}")
        logger.info(f"[*] Auth Token required: {self.config.auth_token} (saved to ~/.whattube/token)")
        ticker_task = asyncio.create_task(self.background_tick())
        try:
            async with websockets.serve(
                self.handle_client,
                self.config.ws_host,
                self.config.ws_port,
                max_size=10_000_000,
            ):
                while self.running:
                    await asyncio.sleep(0.5)
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
