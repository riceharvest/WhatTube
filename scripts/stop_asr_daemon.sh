#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="whattube-asr-daemon"

echo "[*] Stopping WhatTube ASR daemon..."

# Stop native process if running
if [ -f "$DIR/.daemon.pid" ]; then
  PID="$(cat "$DIR/.daemon.pid" 2>/dev/null || true)"
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    echo "[+] Stopped native daemon PID $PID"
  fi
  rm -f "$DIR/.daemon.pid"
fi

# Stop container if running
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo "[+] WhatTube ASR daemon stopped."
