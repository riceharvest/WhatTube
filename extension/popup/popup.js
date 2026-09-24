// WhatTube Popup Script

const toggleBtn = document.getElementById("toggleBtn");
const statusBadge = document.getElementById("statusBadge");
const targetLangSelect = document.getElementById("targetLang");

function updateUI(isRecording) {
  if (isRecording) {
    statusBadge.textContent = "Listening";
    statusBadge.className = "status-badge active";
    toggleBtn.textContent = "Stop Listening";
    toggleBtn.className = "btn active-btn";
  } else {
    statusBadge.textContent = "Idle";
    statusBadge.className = "status-badge idle";
    toggleBtn.textContent = "Start Listening";
    toggleBtn.className = "btn primary";
  }
}

// Load persisted target language preference
chrome.storage.sync.get(["targetLang"], (result) => {
  if (result.targetLang) {
    targetLangSelect.value = result.targetLang;
  }
});

// Update target language when changed
targetLangSelect.addEventListener("change", () => {
  const chosenLang = targetLangSelect.value;
  chrome.storage.sync.set({ targetLang: chosenLang });
  chrome.runtime.sendMessage({
    type: "SET_TARGET_LANG",
    targetLang: chosenLang,
  });
});

// Request initial status
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
  if (response) {
    updateUI(response.isRecording);
    if (response.targetLang) {
      targetLangSelect.value = response.targetLang;
    }
  }
});

toggleBtn.addEventListener("click", () => {
  const currentLang = targetLangSelect.value;
  chrome.runtime.sendMessage({ type: "TOGGLE_CAPTURE", targetLang: currentLang }, (response) => {
    if (response) {
      updateUI(response.isRecording);
    }
  });
});
