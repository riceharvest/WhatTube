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
async function startCapture(tabId, targetLang = 'en', videoTitle = '') {
  try {
    await ensureOffscreenDocument();

    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tabId,
    });

    if (!streamId) {
      throw new Error('Failed to acquire tab audio stream ID');
    }

    // Retrieve auth token from storage if configured
    let authToken = '';
    if (chrome.storage && chrome.storage.sync) {
      const stored = await chrome.storage.sync.get(['authToken']);
      authToken = stored.authToken || '';
    }

    await setSessionState({ isRecording: true, activeTabId: tabId, targetLang, lastError: null });

    chrome.runtime.sendMessage({
      type: 'START_CAPTURE',
      streamId: streamId,
      tabId: tabId,
      targetLang: targetLang,
      authToken: authToken,
      videoTitle: videoTitle,
    });

    // Notify content script that capture is active so it sends immediate video sync
    setTimeout(() => {
      chrome.tabs.sendMessage(tabId, { type: 'CAPTURE_STARTED' }).catch(() => {});
    }, 300);

    console.log(`[WhatTube] Started capture on tab ${tabId} (targetLang: ${targetLang}, title: "${videoTitle}")`);
    return { ok: true };
  } catch (err) {
    console.error('[WhatTube] Failed to start capture:', err);
    await setSessionState({ isRecording: false, activeTabId: null, lastError: err.message });
    return { ok: false, error: err.message || 'Capture initialization failed' };
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
    getSessionState().then(async (state) => {
      if (state.isRecording) {
        await stopCapture();
        sendResponse({ isRecording: false, status: 'idle' });
      } else {
        chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
          if (!tabs[0] || !tabs[0].id) {
            sendResponse({ isRecording: false, status: 'error', error: 'No active tab found' });
            return;
          }
          const tab = tabs[0];
          if (tab.url && (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('edge://') || tab.url.startsWith('about:'))) {
            sendResponse({ isRecording: false, status: 'error', error: 'Cannot capture audio on system or extension tabs' });
            return;
          }
          const result = await startCapture(tab.id, message.targetLang || state.targetLang, tab.title || '');
          if (result.ok) {
            sendResponse({ isRecording: true, status: 'listening' });
          } else {
            sendResponse({ isRecording: false, status: 'error', error: result.error });
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
        video_title: message.video_title || '',
      },
    }).catch(() => {});
    return true;
  }
});
