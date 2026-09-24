#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Check if resident ASR daemon is running
if ! curl -s http://127.0.0.1:8766/health | grep -q "ready"; then
  echo "[*] ASR daemon not running. Launching resident daemon..."
  ./scripts/start_asr_daemon.sh
fi

echo "[*] Starting WhatTube WebSocket server on port 8765..."
export PYTHONPATH="$DIR:$PYTHONPATH"
exec .venv/bin/python -m whattube.server
