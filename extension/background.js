// WhatTube Service Worker (Manifest V3)

// Retrieve session state safely across SW terminations
async function getSessionState() {
  if (chrome.storage && chrome.storage.session) {
    const data = await chrome.storage.session.get(['isRecording', 'activeTabId', 'targetLang']);
    return {
      isRecording: !!data.isRecording,
      activeTabId: data.activeTabId || null,
      targetLang: data.targetLang || 'en',
    };
  }
  return { isRecording: false, activeTabId: null, targetLang: 'en' };
}

async function setSessionState(patch) {
  if (chrome.storage && chrome.storage.session) {
    await chrome.storage.session.set(patch);
  }
}

// Ensure offscreen document is active
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
async function startCapture(tabId, targetLang = 'en') {
  try {
    await ensureOffscreenDocument();

    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tabId,
    });

    await setSessionState({ isRecording: true, activeTabId: tabId, targetLang });

    chrome.runtime.sendMessage({
      type: 'START_CAPTURE',
      streamId: streamId,
      tabId: tabId,
      targetLang: targetLang,
    });

    // Notify content script that capture is active so it sends immediate video sync
    setTimeout(() => {
      chrome.tabs.sendMessage(tabId, { type: 'CAPTURE_STARTED' }).catch(() => {});
    }, 300);

    console.log(`[WhatTube] Started capture on tab ${tabId} (targetLang: ${targetLang})`);
  } catch (err) {
    console.error('[WhatTube] Failed to start capture:', err);
    await setSessionState({ isRecording: false, activeTabId: null });
  }
}

// Stop capture
async function stopCapture() {
  chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' });
  const { activeTabId } = await getSessionState();
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type: 'CAPTURE_STOPPED' }).catch(() => {});
  }
  await setSessionState({ isRecording: false, activeTabId: null });
  console.log('[WhatTube] Stopped capture');
}

// Handle tab close
chrome.tabs.onRemoved.addListener(async (tabId) => {
  const state = await getSessionState();
  if (state.activeTabId === tabId) {
    await stopCapture();
  }
});

// Handle messages from Popup, Offscreen, or Content Script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_STATUS') {
    getSessionState().then(sendResponse);
    return true;
  }

  if (message.type === 'TOGGLE_CAPTURE') {
    getSessionState().then((state) => {
      if (state.isRecording) {
        stopCapture().then(() => sendResponse({ isRecording: false }));
      } else {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs[0] && tabs[0].id) {
            startCapture(tabs[0].id, message.targetLang || state.targetLang).then(() =>
              sendResponse({ isRecording: true })
            );
          } else {
            sendResponse({ isRecording: false, error: 'No active tab' });
          }
        });
      }
    });
    return true;
  }

  if (message.type === 'SET_TARGET_LANG') {
    setSessionState({ targetLang: message.targetLang }).then(() => {
      chrome.runtime.sendMessage({
        type: 'SEND_CONTROL',
        data: {
          type: 'set_target_lang',
          target_lang: message.targetLang,
        },
      }).catch(() => {});
      sendResponse({ success: true });
    });
    return true;
  }

  // Forward captions from offscreen document to the active YouTube tab
  if (message.type === 'RELAY_CAPTION') {
    getSessionState().then((state) => {
      if (state.activeTabId) {
        chrome.tabs.sendMessage(state.activeTabId, {
          type: 'DISPLAY_CAPTION',
          payload: message.payload,
        }).catch((err) => {
          // Tab may have navigated or closed
        });
      }
    });
    return true;
  }

  // Forward video sync and seek events from content script to offscreen document
  if (message.type === 'VIDEO_SYNC' || message.type === 'VIDEO_SEEK') {
    chrome.runtime.sendMessage({
      type: 'SEND_CONTROL',
      data: {
        type: message.type === 'VIDEO_SEEK' ? 'seek' : 'sync',
        video_time: message.video_time,
        playback_rate: message.playback_rate || 1.0,
      },
    }).catch(() => {});
    return true;
  }
});
