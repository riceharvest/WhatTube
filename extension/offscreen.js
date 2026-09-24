// WhatTube Offscreen Audio Processor & WebSocket Client

let mediaStream = null;
let audioContext = null;
let scriptNode = null;
let websocket = null;
let isStreaming = false;

const TARGET_SAMPLE_RATE = 16000;
const WS_URL = 'ws://localhost:8765';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'START_CAPTURE') {
    startAudioProcessing(message.streamId);
  } else if (message.type === 'STOP_CAPTURE') {
    stopAudioProcessing();
  }
});

// Linear resampling helper
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

async function startAudioProcessing(streamId) {
  stopAudioProcessing();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    });

    audioContext = new AudioContext();
    const sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // Keep tab audio audible to user
    sourceNode.connect(audioContext.destination);

    // Setup WebSocket connection to WhatTube server
    connectWebSocket();

    // Use ScriptProcessor for real-time PCM capture
    const bufferSize = 4096;
    scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptNode.onaudioprocess = (event) => {
      if (!isStreaming || !websocket || websocket.readyState !== WebSocket.OPEN) {
        return;
      }

      const inputChannel = event.inputBuffer.getChannelData(0);
      const resampled = downsampleBuffer(inputChannel, audioContext.sampleRate, TARGET_SAMPLE_RATE);

      // Send Float32Array as binary WebSocket frame
      websocket.send(resampled.buffer);
    };

    sourceNode.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    isStreaming = true;
    console.log('[WhatTube Offscreen] Audio capture active, streaming to server.');
  } catch (err) {
    console.error('[WhatTube Offscreen] Error initializing capture:', err);
  }
}

function connectWebSocket() {
  websocket = new WebSocket(WS_URL);
  websocket.binaryType = 'arraybuffer';

  websocket.onopen = () => {
    console.log('[WhatTube Offscreen] Connected to WhatTube server.');
  };

  websocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'caption') {
        // Relay caption to background service worker
        chrome.runtime.sendMessage({
          type: 'RELAY_CAPTION',
          payload: data,
        });
      }
    } catch (e) {
      console.warn('[WhatTube Offscreen] Non-JSON message from server:', e);
    }
  };

  websocket.onclose = () => {
    console.log('[WhatTube Offscreen] WebSocket closed.');
  };

  websocket.onerror = (err) => {
    console.error('[WhatTube Offscreen] WebSocket error:', err);
  };
}

function stopAudioProcessing() {
  isStreaming = false;

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

  console.log('[WhatTube Offscreen] Audio capture stopped.');
}
