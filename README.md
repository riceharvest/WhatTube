# WhatTube 🗣️

> **"What did they say?"** — Sparse, real-time foreign chatter detection, transcription, and translation overlay for YouTube.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![VRAM Footprint](https://img.shields.io/badge/VRAM-1.54%20GB-green.svg)]()
[![GPU Duty Cycle](https://img.shields.io/badge/GPU%20Duty%20Cycle-0.31%25-brightgreen.svg)]()

WhatTube is a specialized, lightweight audio pipeline designed for travel vlogs and multilingual videos where a **foreground host speaks English** while **distant background locals converse in foreign languages** (e.g. street vendors, bystanders, youth talking slang).

Rather than wasting compute running continuous heavy models or fragile acoustic waveform separation, WhatTube uses a **Sparse Spoken-Language-ID (LID) + Event-Aggregated GPU ASR Pipeline**:
1. **Continuous CPU Stage 1 (<55 ms, <5% CPU):** Silero VAD + quantized Whisper Tiny INT8 Spoken Language ID acting as a "clean English filter".
2. **Dynamic Event Aggregator:** State machine coalescing suspicious non-English speech into 3–10s bursts with pre-roll and hangover (~1.45 events/minute).
3. **Sparse GPU Stage 2 (127 ms latency, 1.54 GB VRAM):** Whisper `large-v3-turbo` FP16 resident ASR triggered on bursts. If Turbo detects English or noise, it silently discards.
4. **CPU Multilingual Translation (<80 ms, 0 MB VRAM):** Dynamic MarianMT routing (`detected_lang -> target_lang`).
5. **Browser Extension:** Manifest V3 tab audio capture streaming PCM over WebSocket to inject responsive subtitle cards directly into the YouTube video DOM (`.html5-video-player`).

---

## Architecture

```
                  YouTube Video Stream (Browser Tab Audio)
                                     │ (AudioWorklet 16kHz PCM)
                                     ▼
                ┌───────────────────────────────────────────┐
                │ SessionState (Isolated Per-Tab Worker)    │
                │ Dedicated Ring Buffer, VAD & LID State    │
                │                                           │
                │ 1. Energy & Silero VAD Filter             │ ~2 ms
                │ 2. sherpa-onnx Whisper Tiny INT8 LID     │ ~53 ms
                │ 3. Temporal State Machine:                │
                │    - Confident English (p >= 0.75)        │ -> Ignore
                │    - 2-in-3 Suspicious (p < 0.55)         │ -> Queue Event
                │    - Extremely Non-English (p < 0.20)     │ -> Immediate Trigger
                └────────────────────┬──────────────────────┘
                                     │ (Average: ~1.45 events / minute)
                                     ▼
                ┌───────────────────────────────────────────┐
                │ Dynamic Event Aggregator & Coalescer      │
                │ - Dynamic 3-10s adaptive burst duration   │
                │ - 1.0s pre-roll + 0.5s post-roll padding  │
                │ - Epoch checking (discards on seek/reset) │
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼ (Serialized GPU Semaphore Queue)
                ┌───────────────────────────────────────────┐
                │ GPU Stage 2: Hardware-Agnostic ASR        │
                │ (CUDA / Apple MPS / Intel XPU / CPU)     │
                │ Target VRAM: <= 1.54 GB                   │
                │                                           │
                │ Resident Whisper large-v3-turbo FP16      │ ~127 ms
                │ Output Gate: If ASR lang == English -> Drop│
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼
                ┌───────────────────────────────────────────┐
                │ CPU Stage 3: Low-Footprint Translation    │
                │ (CTranslate2 INT8 on CPU with LRU cache)  │ ~50-80 ms
                │ - 0 MB VRAM, leaves GPU 100% free         │
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼
                    YouTube DOM Subtitle Overlay Card
             "Don't forget to tap the screen until it breaks!"
                    [ID] Indonesian · original available
```

---

## Empirical Benchmark Highlights (Reference Baseline)

Evaluated on a continuous **41-minute, 15-second** travel vlog recorded on location in Jakarta, Indonesia (`P13mMiIL_2I`):

| Metric | Result | Impact |
|---|---|---|
| **CPU LID Speed** | **53.8 ms** per 3s window | $18.5\times$ faster than real-time on 4 CPU threads |
| **GPU Event Reduction** | **2,472 windows $\to$ 60 bursts** | **97.6% reduction** in GPU activations |
| **Ground-Truth Chatter Recall** | **100%** (3/3 segments) | 0 false-negative misses on known local chatter |
| **GPU VRAM Footprint** | **1,543.7 MB (1.54 GB)** | Runs comfortably alongside LLMs or budget GPUs |
| **GPU Inference Latency** | **127.44 ms** per 5s burst | Near-instantaneous caption turnaround |
| **GPU Duty Cycle** | **0.308%** | **99.69% GPU idle** across the 41-minute playback |
| **CPU Translation Latency** | **<80 ms** | 0 MB VRAM consumed for multilingual translation |

> [!NOTE]
> This 41-minute Jakarta vlog serves as our empirical baseline smoke test. Broad-spectrum evaluations across varied recording environments, accents, and background music are ongoing.

---

## Installation & Setup

### 1. Requirements
* Linux / macOS / Windows
* Python 3.10+
* GPU Acceleration (auto-detected):
  * **NVIDIA** (CUDA)
  * **Apple Silicon** (Metal / MPS)
  * **Intel Arc / Data Center** (XPU / OpenVINO)
  * **CPU Fallback**
* Chrome, Brave, or Chromium-based browser

### 2. Clone and Setup Environment
```bash
git clone https://github.com/riceharvest/WhatTube.git
cd WhatTube

# Create virtual environment and install dependencies
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download quantized Whisper Tiny INT8 LID models (~75 MB)
./scripts/download_models.sh
```

### 3. Run the Test Suite
Verify that ring buffers, ASR contracts, session isolation, and translation caching pass:
```bash
pytest tests/ -v
```

### 4. Start the Pipeline
Start both the resident GPU ASR daemon and the WebSocket server:
```bash
# Launch server (auto-spawns ASR daemon if not already running)
./scripts/start_server.sh
```

Or run the ASR daemon independently:
```bash
# Native Python daemon
./scripts/start_asr_daemon.sh --native

# Or in a container
./scripts/start_asr_daemon.sh --docker
```

### 5. Load the Chrome Extension
1. Open Chrome or Brave and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in top-right).
3. Click **Load unpacked** and select the `WhatTube/extension` directory.
4. Pin the **WhatTube** icon to your toolbar.
5. Open any YouTube travel vlog, select your preferred target language in the popup, click **Start Listening**, and enjoy!

---

## Key Architectural Highlights

1. **Per-Tab Session Isolation (`SessionState`):**
   - Each connected browser tab maintains completely independent `AudioRingBuffer`, `EnergyAndSileroVAD`, `WhisperTinyLID`, and `DynamicEventAggregator` instances.
   - Audio PCM and subtitles never cross-contaminate between tabs.

2. **Stream Epoch Invalidation:**
   - Whenever a user seeks, scrubs, pauses, or resets playback, the tab's stream epoch is incremented.
   - In-flight background ASR bursts and translation tasks verify the epoch before submission and emission; stale predictions from previous video timestamps are automatically dropped.

3. **Acoustic Breath-Pause VAD Splitting:**
   - In travel footage, foreground hosts often trail off in English (e.g. *"Goodbye!"*) right before background locals speak. Passing the full slice into Whisper causes the English tokens to dominate the cross-attention layers.
   - WhatTube uses Silero VAD energy valleys ($p < 0.30$) to detect natural acoustic breath pauses between speakers and decouple them into sub-bursts. The English segment is recognized by Turbo ASR and silently dropped; the foreign segment is cleanly transcribed and translated.

4. **CTranslate2 INT8 Multilingual Translation Engine:**
   - Drops translation latency from **~600 ms (PyTorch MarianMT CPU) to ~50 ms (CTranslate2 INT8)**.
   - Thread-safe bounded LRU model cache (`max_cached_models=3`), consuming only ~72 MB RAM per active language pair with zero GPU VRAM.

5. **AudioWorklet & Manifest V3 Lifecycle Resilience:**
   - Capture runs through a dedicated `AudioWorkletProcessor` delivering clean 16 kHz single-channel PCM without main-thread audio glitching.
   - Extension state is stored in `chrome.storage.session`, surviving Chrome Manifest V3 service worker suspensions.
   - Subtitle lifecycles are directly bound to YouTube video time, auto-dismissing cleanly and pausing whenever the video pauses.

---

## Subtitle Display Features
* **Native Integration:** Injects floating caption cards directly into YouTube's `.html5-video-player`, repositioning smoothly above playback controls.
* **Dual Representation:**
  * Primary line: Translated fluent English (or chosen target language).
  * Metadata badge: `[ID] Indonesian · original available`
  * Click to expand: Reveals exact transcribed foreign slang / colloquial text.
* **Responsive Timeline Sync:** Automatically synchronizes and clears subtitles upon seek or pause.

---

## Repository Structure
```
WhatTube/
├── whattube/
│   ├── audio_buffer.py        # Rolling ring buffer with pre/post-roll extraction (RLock protected)
│   ├── vad.py                 # Fast energy gate + Silero VAD
│   ├── lid.py                 # Sub-55ms Whisper Tiny INT8 language identification
│   ├── event_aggregator.py    # Dynamic 3-10s adaptive burst state machine
│   ├── logger.py              # Event audit precision & diagnostic logger
│   ├── asr/
│   │   ├── resident_daemon.py # Persistent GPU Whisper microservice (CUDA/MPS/XPU/CPU)
│   │   └── client.py          # Fast HTTP client to resident worker
│   ├── translation/
│   │   └── marian.py          # Dynamic multilingual CTranslate2 INT8 CPU translator
│   └── server.py              # WebSocket server & SessionState coordinator
├── extension/                 # Manifest V3 Chrome/Brave Extension
│   ├── manifest.json
│   ├── background.js          # Service worker with chrome.storage.session persistence
│   ├── offscreen.js           # Real-time tab audio capture & AudioWorklet pipeline
│   ├── pcm-worklet-processor.js # Low-overhead 16kHz PCM audio worklet
│   ├── content.js             # YouTube player DOM subtitle injection & timeline sync
│   └── popup/                 # User settings, language selector, and toggle UI
├── scripts/
│   ├── download_models.sh     # Fetch quantized Whisper Tiny INT8 ONNX models
│   ├── start_server.sh        # Server launch script
│   ├── start_asr_daemon.sh    # Resident GPU worker launcher (native/docker)
│   ├── stop_asr_daemon.sh     # Clean shutdown for ASR worker
│   └── simulate_youtube_stream.py # Real-time playback test harness
└── tests/
    ├── test_audio_buffer.py       # Ring buffer append, wrap, slice, deadlock prevention
    ├── test_event_aggregator.py   # Burst detection and window coalescing tests
    ├── test_session_isolation.py  # Multi-client isolation & epoch cancellation tests
    ├── test_asr_contract.py       # ASR daemon endpoint contract & resampling tests
    └── test_translation.py       # CTranslate2 translation & LRU cache tests
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.

