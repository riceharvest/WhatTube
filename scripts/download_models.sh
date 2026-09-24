#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR"

echo "[*] Checking WhatTube model checkpoints in $MODELS_DIR..."

# 1. Silero VAD v6
if [ ! -f "$MODELS_DIR/silero_vad_v6.onnx" ] || [ ! -s "$MODELS_DIR/silero_vad_v6.onnx" ]; then
  echo "[*] Downloading Silero VAD ONNX..."
  curl -L --fail --retry 3 -o "$MODELS_DIR/silero_vad_v6.onnx" \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
fi

# 2. sherpa-onnx quantized Whisper Tiny (encoder + decoder + tokens)
if [ ! -f "$MODELS_DIR/tiny-encoder.int8.onnx" ] || [ ! -f "$MODELS_DIR/tiny-decoder.int8.onnx" ]; then
  echo "[*] Downloading and extracting quantized Whisper Tiny ONNX..."
  TMP_ARCHIVE="$MODELS_DIR/whisper-tiny-tmp.tar.bz2"
  curl -L --fail --retry 3 -o "$TMP_ARCHIVE" \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  
  tar -xjf "$TMP_ARCHIVE" --strip-components=1 -C "$MODELS_DIR" \
    sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
    sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx \
    sherpa-onnx-whisper-tiny/tiny-tokens.txt
  
  rm -f "$TMP_ARCHIVE"
fi

# 3. Verify files
echo "[*] Verifying model integrity..."
for f in "silero_vad_v6.onnx" "tiny-encoder.int8.onnx" "tiny-decoder.int8.onnx" "tiny-tokens.txt"; do
  if [ ! -f "$MODELS_DIR/$f" ] || [ ! -s "$MODELS_DIR/$f" ]; then
    echo "[-] Error: Expected model file $MODELS_DIR/$f is missing or empty!"
    exit 1
  fi
  echo "    ✓ $f ($(du -h "$MODELS_DIR/$f" | cut -f1))"
done

echo "[+] All required WhatTube model files are verified and ready."
