// WhatTube Service Worker (Manifest V3)

let activeTabId = null;
let isRecording = false;

// Ensure offscreen document is created
async function ensureOffscreenDocument() {
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });

  if (existingContexts.length > 0) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA'],
    justification: 'Capture tab audio to transcribe background chatter via WebSocket',
  });
}

// Start tab capture
async function startCapture(tabId) {
  try {
    await ensureOffscreenDocument();

    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tabId,
    });

    activeTabId = tabId;
    isRecording = true;

    chrome.runtime.sendMessage({
      type: 'START_CAPTURE',
      streamId: streamId,
      tabId: tabId,
    });

    console.log(`[WhatTube] Started capture on tab ${tabId}`);
  } catch (err) {
    console.error('[WhatTube] Failed to start capture:', err);
    isRecording = false;
  }
}

// Stop capture
async function stopCapture() {
  chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' });
  isRecording = false;
  activeTabId = null;
  console.log('[WhatTube] Stopped capture');
}

// Handle messages from Popup or Offscreen
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_STATUS') {
    sendResponse({ isRecording, activeTabId });
    return true;
  }

  if (message.type === 'TOGGLE_CAPTURE') {
    if (isRecording) {
      stopCapture().then(() => sendResponse({ isRecording: false }));
    } else {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0] && tabs[0].id) {
          startCapture(tabs[0].id).then(() => sendResponse({ isRecording: true }));
        } else {
          sendResponse({ isRecording: false, error: 'No active tab' });
        }
      });
    }
    return true;
  }

  // Forward captions from offscreen document to the YouTube tab
  if (message.type === 'RELAY_CAPTION') {
    if (activeTabId) {
      chrome.tabs.sendMessage(activeTabId, {
        type: 'DISPLAY_CAPTION',
        payload: message.payload,
      }).catch((err) => {
        // Tab might have navigated or closed
      });
    }
  }

  // Forward video sync and seek events from content script to offscreen document
  if (message.type === 'VIDEO_SYNC' || message.type === 'VIDEO_SEEK') {
    chrome.runtime.sendMessage({
      type: 'SEND_CONTROL',
      data: {
        type: message.type === 'VIDEO_SEEK' ? 'seek' : 'sync',
        video_time: message.video_time,
      },
    }).catch(() => {});
  }
});
