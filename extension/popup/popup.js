// WhatTube Popup Script

const toggleBtn = document.getElementById("toggleBtn");
const statusBadge = document.getElementById("statusBadge");

function updateUI(isRecording) {
  if (isRecording) {
    statusBadge.textContent = "Active";
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

// Request initial status
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
  if (response) {
    updateUI(response.isRecording);
  }
});

toggleBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "TOGGLE_CAPTURE" }, (response) => {
    if (response) {
      updateUI(response.isRecording);
    }
  });
});
