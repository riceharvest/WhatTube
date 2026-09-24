// WhatTube Offscreen Audio Processor & WebSocket Client

let mediaStream = null;
let audioContext = null;
let workletNode = null;
let scriptNode = null;
let websocket = null;
let isStreaming = false;
let currentTargetLang = "en";
let currentVideoTime = 0.0;
let currentPlaybackRate = 1.0;
let currentAuthToken = "";

let reconnectAttempts = 0;
let reconnectTimer = null;
const MAX_RECONNECT_DELAY_MS = 8000;
const INITIAL_RECONNECT_DELAY_MS = 1000;

const WS_URL = "ws://127.0.0.1:8765";
const MAX_BUFFERED_AMOUNT = 64 * 1024; // 64 KB backpressure threshold

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "START_CAPTURE") {
    currentTargetLang = message.targetLang || "en";
    currentAuthToken = message.authToken || "";
    startAudioProcessing(message.streamId);
  } else if (message.type === "STOP_CAPTURE") {
    stopAudioProcessing();
  } else if (message.type === "SEND_CONTROL") {
    if (message.data) {
      if (message.data.type === "set_target_lang") {
        currentTargetLang = message.data.target_lang || "en";
      }
      if (message.data.video_time !== undefined) {
        currentVideoTime = message.data.video_time;
      }
      if (message.data.playback_rate !== undefined) {
        currentPlaybackRate = message.data.playback_rate;
      }
      sendControlMessage(message.data);
    }
  }
});

function sendControlMessage(data) {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    try {
      websocket.send(JSON.stringify(data));
    } catch (e) {
      console.warn("[WhatTube Offscreen] Failed to send control message:", e);
    }
  }
}

function connectWebSocket() {
  if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    websocket = new WebSocket(WS_URL);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
      console.log("[WhatTube Offscreen] Connected to WhatTube server.");
      reconnectAttempts = 0;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }

      // Handshake with server session, passing auth token and playback rate
      sendControlMessage({
        type: "init",
        token: currentAuthToken,
        target_lang: currentTargetLang,
        video_time: currentVideoTime,
        playback_rate: currentPlaybackRate,
      });
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "caption") {
          chrome.runtime.sendMessage({
            type: "RELAY_CAPTION",
            payload: data,
          });
        } else if (data.type === "error") {
          console.error("[WhatTube Offscreen] Server error:", data.error);
          chrome.runtime.sendMessage({
            type: "CAPTURE_ERROR",
            error: data.error,
          });
        }
      } catch (e) {
        console.warn("[WhatTube Offscreen] Non-JSON message from server:", e);
      }
    };

    websocket.onclose = () => {
      console.log("[WhatTube Offscreen] WebSocket closed.");
      scheduleReconnect();
    };

    websocket.onerror = (err) => {
      console.error("[WhatTube Offscreen] WebSocket error:", err);
      // onclose will trigger scheduleReconnect
    };
  } catch (err) {
    console.error("[WhatTube Offscreen] Error establishing WebSocket:", err);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!isStreaming || reconnectTimer) {
    return;
  }

  const delay = Math.min(
    INITIAL_RECONNECT_DELAY_MS * Math.pow(1.5, reconnectAttempts),
    MAX_RECONNECT_DELAY_MS
  );
  reconnectAttempts++;
  console.log(`[WhatTube Offscreen] Scheduling reconnect in ${delay}ms (attempt #${reconnectAttempts})...`);

  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (isStreaming) {
      connectWebSocket();
    }
  }, delay);
}

function sendAudioChunk(float32Array) {
  if (!isStreaming || !websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Backpressure protection: drop chunks if socket is backlogged
  if (websocket.bufferedAmount > MAX_BUFFERED_AMOUNT) {
    return;
  }

  try {
    websocket.send(float32Array.buffer);
  } catch (err) {
    console.warn("[WhatTube Offscreen] Error sending audio frame:", err);
  }
}

async function startAudioProcessing(streamId) {
  stopAudioProcessing();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    });

    // Native 16 kHz AudioContext - browser automatically resamples input cleanly
    audioContext = new AudioContext({ sampleRate: 16000 });
    const sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // Keep tab audio audible to the user
    sourceNode.connect(audioContext.destination);

    // Connect to WebSocket server
    connectWebSocket();

    // Use AudioWorklet if supported
    try {
      await audioContext.audioWorklet.addModule("pcm-worklet-processor.js");
      workletNode = new AudioWorkletNode(audioContext, "pcm-worklet-processor");
      workletNode.port.onmessage = (event) => {
        sendAudioChunk(event.data);
      };
      sourceNode.connect(workletNode);
      workletNode.connect(audioContext.destination);
      console.log("[WhatTube Offscreen] Using modern AudioWorklet @ 16 kHz.");
    } catch (workletErr) {
      console.warn("[WhatTube Offscreen] AudioWorklet failed, using ScriptProcessor fallback:", workletErr);
      const bufferSize = 4096;
      scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
      scriptNode.onaudioprocess = (event) => {
        const inputChannel = event.inputBuffer.getChannelData(0);
        sendAudioChunk(new Float32Array(inputChannel));
      };
      sourceNode.connect(scriptNode);
      scriptNode.connect(audioContext.destination);
    }

    isStreaming = true;
    console.log("[WhatTube Offscreen] Audio capture active, streaming to server.");
  } catch (err) {
    console.error("[WhatTube Offscreen] Error initializing capture:", err);
  }
}

function stopAudioProcessing() {
  isStreaming = false;

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }

  if (scriptNode) {
    scriptNode.disconnect();
    scriptNode = null;
  }

  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (websocket) {
    websocket.close();
    websocket = null;
  }

  console.log("[WhatTube Offscreen] Audio capture stopped.");
}
