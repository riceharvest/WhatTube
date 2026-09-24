#!/usr/bin/env bash
CONTAINER_NAME="whattube-asr-daemon"
echo "[*] Stopping WhatTube ASR daemon container..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo "[+] WhatTube ASR daemon stopped."
