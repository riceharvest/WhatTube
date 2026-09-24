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
                                     │
                                     ▼
                ┌───────────────────────────────────────────┐
                │ CPU Stage 1: Continuous Ring Buffer       │
                │ (5950X / Host CPU, <5% single-core load)  │
                │                                           │
                │ 1. Energy & Silero VAD Filter             │ ~2 ms
                │ 2. sherpa-onnx Whisper Tiny INT8 LID     │ ~53 ms
                │ 3. Temporal State Machine:                │
                │    - Confident English (p >= 0.75)        │ -> Ignore
                │    - 2-in-3 Suspicious (p < 0.55)         │ -> Queue Event
                │    - Extremely Non-English (p < 0.20)     │ -> Immediate Trigger
                └────────────────────┬──────────────────────┘
                                     │ (Average: 1.45 events / minute)
                                     ▼
                ┌───────────────────────────────────────────┐
                │ Dynamic Event Aggregator & Coalescer      │
                │ - Dynamic 3-10s adaptive burst duration   │
                │ - 1.0s pre-roll + 0.5s post-roll padding  │
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼ (0.31% GPU duty cycle)
                ┌───────────────────────────────────────────┐
                │ GPU Stage 2: Hardware-Agnostic ASR        │
                │ Target VRAM: <= 1.54 GB                   │
                │                                           │
                │ Resident Whisper large-v3-turbo FP16      │ ~127 ms
                │ Output Gate: If ASR lang == English -> Drop│
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼
                ┌───────────────────────────────────────────┐
                │ CPU Stage 3: Low-Footprint Translation    │
                │ (MarianMT on CPU: src_lang -> target_lang)│ ~50-80 ms
                │ - 0 MB VRAM, leaves GPU 100% free         │
                └────────────────────┬──────────────────────┘
                                     │
                                     ▼
                    YouTube DOM Subtitle Overlay Card
             "Don't forget to tap the screen until it breaks!"
                    [ID] Indonesian · original available
```

---

## Empirical Benchmark Highlights

Evaluated on a continuous **41-minute, 15-second** travel vlog recorded on location in Jakarta, Indonesia (`P13mMiIL_2I`):

| Metric | Result | Impact |
|---|---|---|
| **CPU LID Speed** | **53.8 ms** per 3s window | $18.5\times$ faster than real-time on 4 threads |
| **GPU Event Reduction** | **2,472 windows $\to$ 60 bursts** | **97.6% reduction** in GPU activations |
| **Ground-Truth Chatter Recall** | **100%** (3/3 segments) | 0 false-negative misses on known local chatter |
| **GPU VRAM Footprint** | **1,543.7 MB (1.54 GB)** | Runs on standard laptops and budget GPUs |
| **GPU Inference Latency** | **127.44 ms** per 5s burst | Near-instantaneous caption turnaround |
| **B70 GPU Duty Cycle** | **0.308%** | **99.69% GPU idle** across the 41-minute playback |
| **CPU Translation Latency** | **<80 ms** | 0 MB VRAM consumed for LLM translation |

---

## Installation & Setup

### 1. Requirements
* Linux / macOS / Windows
* Python 3.10+
* GPU: Intel Arc (via XPU / OpenVINO), NVIDIA (CUDA), Apple Silicon (MPS/CoreML), or CPU fallback
* Chrome or Brave browser

### 2. Clone and Setup Environment
```bash
git clone https://github.com/myappleiddarimaan/WhatTube.git
cd WhatTube

# Create virtual environment and install dependencies
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Start the Pipeline
Start both the resident GPU ASR daemon and the WebSocket server:
```bash
# Launch server (auto-spawns ASR daemon)
./scripts/start_server.sh
```

### 4. Load the Chrome Extension
1. Open Chrome or Brave and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in top-right).
3. Click **Load unpacked** and select the `WhatTube/extension` directory.
4. Pin the **WhatTube** icon to your toolbar.
5. Open any YouTube travel vlog, click the WhatTube icon, and press **Start Listening**!

---

## Subtitle Display Features
* **Native Integration:** Injects floating caption cards directly into YouTube's `.html5-video-player`, repositioning smoothly above playback controls.
* **Dual Representation:**
  * Primary line: Translated fluent English (or chosen target language).
  * Metadata badge: `[ID] Indonesian · original available`
  * Click to expand: Reveals exact transcribed foreign slang / colloquial text.
* **Responsive Timeline Sync:** Automatically clears subtitles upon seek or pause.

---

## Repository Structure
```
WhatTube/
├── whattube/
│   ├── audio_buffer.py        # Rolling ring buffer with pre/post-roll extraction
│   ├── vad.py                 # Fast energy gate + Silero VAD
│   ├── lid.py                 # Sub-55ms Whisper Tiny INT8 language identification
│   ├── event_aggregator.py    # Dynamic 3-10s adaptive burst state machine
│   ├── logger.py              # Event audit precision & diagnostic logger
│   ├── asr/
│   │   ├── resident_daemon.py # Persistent GPU Whisper microservice
│   │   └── client.py          # Fast HTTP client to resident worker
│   ├── translation/
│   │   └── marian.py          # Dynamic multilingual CPU translator
│   └── server.py              # WebSocket server & pipeline coordinator
├── extension/                 # Manifest V3 Chrome/Brave Extension
│   ├── manifest.json
│   ├── background.js          # Service worker
│   ├── offscreen.js           # Real-time tab audio capture & resampling
│   ├── content.js             # YouTube player DOM subtitle injection
│   └── popup/                 # User settings and toggle UI
├── scripts/
│   ├── start_server.sh        # Server launch script
│   ├── start_asr_daemon.sh    # Resident GPU worker launcher
│   └── simulate_youtube_stream.py # Real-time playback test harness
└── tests/
    └── test_event_aggregator.py
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
