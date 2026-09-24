#!/usr/bin/env bash
set -e

CONTAINER_NAME="whattube-asr-daemon"
IMAGE_ID="f01e24f6c7ff"
PORT=8766

echo "[*] Checking for existing ASR container..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "[*] Launching resident Whisper Large-v3-Turbo daemon on Intel Arc B70 (Port $PORT)..."
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128 2>/dev/null || echo 0)"

docker run -d \
  --name "$CONTAINER_NAME" \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v /home/dario/.cache/huggingface:/root/.cache/huggingface \
  -v /home/dario/WhatTube:/app \
  -p 127.0.0.1:${PORT}:${PORT} \
  --entrypoint python3 \
  "$IMAGE_ID" /app/whattube/asr/resident_daemon.py --host 0.0.0.0 --port "$PORT"

echo "[*] Waiting for ASR daemon to warm up..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:${PORT}/health | grep -q "ready"; then
    echo "[+] Resident ASR daemon is READY on http://127.0.0.1:${PORT}!"
    exit 0
  fi
  sleep 1
done

echo "[-] Daemon did not respond in 30s. Check logs with: docker logs $CONTAINER_NAME"
exit 1
