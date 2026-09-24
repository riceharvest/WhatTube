// WhatTube Popup Script

const toggleBtn = document.getElementById("toggleBtn");
const statusBadge = document.getElementById("statusBadge");
const targetLangSelect = document.getElementById("targetLang");
const authTokenInput = document.getElementById("authToken");
const errorContainer = document.getElementById("errorContainer");
const serverStatusSpan = document.getElementById("serverStatus");

function updateUI(status, errorMsg = null) {
  if (status === "listening" || status === true) {
    statusBadge.textContent = "Listening";
    statusBadge.className = "status-badge active";
    toggleBtn.textContent = "Stop Listening";
    toggleBtn.className = "btn active-btn";
    toggleBtn.disabled = false;
    errorContainer.style.display = "none";
  } else if (status === "starting") {
    statusBadge.textContent = "Starting...";
    statusBadge.className = "status-badge starting";
    toggleBtn.textContent = "Connecting...";
    toggleBtn.disabled = true;
    errorContainer.style.display = "none";
  } else if (status === "error") {
    statusBadge.textContent = "Error";
    statusBadge.className = "status-badge error";
    toggleBtn.textContent = "Start Listening";
    toggleBtn.className = "btn primary";
    toggleBtn.disabled = false;
    if (errorMsg) {
      errorContainer.textContent = errorMsg;
      errorContainer.style.display = "block";
    }
  } else {
    statusBadge.textContent = "Idle";
    statusBadge.className = "status-badge idle";
    toggleBtn.textContent = "Start Listening";
    toggleBtn.className = "btn primary";
    toggleBtn.disabled = false;
    errorContainer.style.display = "none";
  }
}

// Load persisted target language and auth token preferences
chrome.storage.sync.get(["targetLang", "authToken"], (result) => {
  if (result.targetLang) {
    targetLangSelect.value = result.targetLang;
  }
  if (result.authToken && authTokenInput) {
    authTokenInput.value = result.authToken;
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

// Update auth token when changed
if (authTokenInput) {
  authTokenInput.addEventListener("input", () => {
    chrome.storage.sync.set({ authToken: authTokenInput.value.trim() });
  });
}

// Request initial status
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
  if (response) {
    if (response.isRecording) {
      updateUI("listening");
    } else if (response.lastError) {
      updateUI("error", response.lastError);
    } else {
      updateUI("idle");
    }
    if (response.targetLang) {
      targetLangSelect.value = response.targetLang;
    }
  }
});

toggleBtn.addEventListener("click", () => {
  const currentLang = targetLangSelect.value;
  updateUI("starting");
  chrome.runtime.sendMessage({ type: "TOGGLE_CAPTURE", targetLang: currentLang }, (response) => {
    if (response) {
      if (response.status === "listening") {
        updateUI("listening");
      } else if (response.status === "error") {
        updateUI("error", response.error || "Failed to start capture");
      } else {
        updateUI("idle");
      }
    } else {
      updateUI("error", "No response from background worker");
    }
  });
});
