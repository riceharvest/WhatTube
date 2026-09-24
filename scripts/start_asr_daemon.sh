#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${WHATTUBE_ASR_PORT:-8766}"
HOST="${WHATTUBE_ASR_HOST:-127.0.0.1}"

# Check if daemon is already running
if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q "ready"; then
  echo "[+] Resident ASR daemon is already running on http://${HOST}:${PORT}."
  exit 0
fi

# Mode 1: Native Python execution (Preferred for macOS MPS, local CUDA, local XPU, CPU)
if [ "${WHATTUBE_USE_DOCKER:-0}" != "1" ] && [ -f "$DIR/.venv/bin/python" ]; then
  echo "[*] Launching resident Whisper daemon natively via Python (.venv)..."
  PYTHONPATH="$DIR" "$DIR/.venv/bin/python" -m whattube.asr.resident_daemon --host "$HOST" --port "$PORT" &
  DAEMON_PID=$!
  echo "$DAEMON_PID" > "$DIR/.daemon.pid"

  echo "[*] Waiting for ASR daemon to warm up..."
  for i in {1..30}; do
    if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q "ready"; then
      echo "[+] Resident ASR daemon is READY on http://${HOST}:${PORT}! (PID: $DAEMON_PID)"
      exit 0
    fi
    sleep 1
  done
  echo "[-] Native daemon did not start in 30s."
  exit 1
fi

# Mode 2: Docker Container fallback
CONTAINER_NAME="whattube-asr-daemon"
IMAGE="${WHATTUBE_DOCKER_IMAGE:-f01e24f6c7ff}"

echo "[*] Launching resident Whisper daemon in Docker container '$CONTAINER_NAME'..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

DOCKER_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
  -v "$DIR:/app"
  -p "${HOST}:${PORT}:${PORT}"
)

# Optional GPU device mounts
if [ -d "/dev/dri" ]; then
  RENDER_GID="$(stat -c '%g' /dev/dri/renderD128 2>/dev/null || echo 0)"
  DOCKER_ARGS+=(--device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro)
elif command -v nvidia-smi &>/dev/null; then
  DOCKER_ARGS+=(--gpus all)
fi

docker run "${DOCKER_ARGS[@]}" \
  --entrypoint python3 \
  "$IMAGE" /app/whattube/asr/resident_daemon.py --host 0.0.0.0 --port "$PORT"

echo "[*] Waiting for containerized ASR daemon to warm up..."
for i in {1..30}; do
  if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q "ready"; then
    echo "[+] Resident ASR daemon is READY on http://${HOST}:${PORT}!"
    exit 0
  fi
  sleep 1
done

echo "[-] Daemon did not respond in 30s. Check logs with: docker logs $CONTAINER_NAME"
exit 1
