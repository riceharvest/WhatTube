#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR"

echo "[*] Checking WhatTube model checkpoints in $MODELS_DIR..."

# Check Silero VAD
if [ ! -f "$MODELS_DIR/silero_vad_v6.onnx" ]; then
  echo "[*] Downloading Silero VAD ONNX..."
  curl -L -o "$MODELS_DIR/silero_vad_v6.onnx" \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx || true
fi

# Check Whisper Tiny INT8
if [ ! -f "$MODELS_DIR/tiny-encoder.int8.onnx" ]; then
  echo "[*] Downloading quantized Whisper Tiny ONNX..."
  curl -L -o "$MODELS_DIR/tiny-encoder.int8.onnx" \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2 || true
fi

echo "[+] Model checks complete."
