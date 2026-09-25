// WhatTube Content Script for YouTube Player Overlay

let overlayContainer = null;
let activeCard = null;
let activeCardDismissVideoTime = null;
let wallClockFallbackTimer = null;
let lastSyncWallTime = 0.0;

const LANGUAGE_NAMES = {
  id: "Indonesian",
  es: "Spanish",
  fr: "French",
  de: "German",
  ja: "Japanese",
  zh: "Chinese",
  it: "Italian",
  pt: "Portuguese",
  ru: "Russian",
  ko: "Korean",
  ar: "Arabic",
  hi: "Hindi",
  th: "Thai",
  vi: "Vietnamese",
  nl: "Dutch",
  ms: "Malay",
  tl: "Tagalog",
};

function ensureOverlayContainer() {
  const player = document.querySelector("#movie_player") || document.querySelector(".html5-video-player");
  if (!player) return null;

  if (!overlayContainer || !player.contains(overlayContainer)) {
    overlayContainer = document.createElement("div");
    overlayContainer.className = "whattube-container";
    player.appendChild(overlayContainer);
  }
  return overlayContainer;
}

function clearActiveCard() {
  if (wallClockFallbackTimer) {
    clearTimeout(wallClockFallbackTimer);
    wallClockFallbackTimer = null;
  }
  activeCardDismissVideoTime = null;

  if (activeCard) {
    const cardToRemove = activeCard;
    activeCard = null;
    cardToRemove.classList.add("whattube-fade-out");
    setTimeout(() => {
      if (cardToRemove.parentNode) {
        cardToRemove.parentNode.removeChild(cardToRemove);
      }
    }, 350);
  }
}

function displayCaption(caption) {
  const container = ensureOverlayContainer();
  if (!container) return;

  const langCode = (caption.language || "unknown").toLowerCase();
  const langName = LANGUAGE_NAMES[langCode] || langCode.toUpperCase();
  const isUpdate = caption.action === "update";

  // Timeline-bound dismiss time (in video seconds)
  const video = document.querySelector("video");
  const targetEndSec = typeof caption.end === "number" ? caption.end : (video ? video.currentTime + 3.0 : 0.0);
  activeCardDismissVideoTime = targetEndSec + 2.5;

  // Reset wall-clock safety fallback (12s maximum in case playback stops entirely)
  if (wallClockFallbackTimer) {
    clearTimeout(wallClockFallbackTimer);
  }
  wallClockFallbackTimer = setTimeout(() => {
    clearActiveCard();
  }, 12000);

  // If protocol specifies UPDATE and active card exists, morph in-place
  if (isUpdate && activeCard && activeCard.parentNode) {
    const transEl = activeCard.querySelector(".whattube-translation");
    const origEl = activeCard.querySelector(".whattube-original");
    const badgeEl = activeCard.querySelector(".whattube-badge");

    if (transEl && origEl) {
      transEl.textContent = caption.translation;
      origEl.textContent = `“${caption.original}”`;
      if (badgeEl) badgeEl.textContent = langCode;
      activeCard.classList.remove("whattube-fade-out");
      return;
    }
  }

  // Otherwise, clear previous card and create new card
  if (activeCard && activeCard.parentNode) {
    activeCard.parentNode.removeChild(activeCard);
    activeCard = null;
  }

  const card = document.createElement("div");
  card.className = "whattube-card";

  card.innerHTML = `
    <div class="whattube-translation">${escapeHtml(caption.translation)}</div>
    <div class="whattube-meta" title="Click to view original foreign speech">
      <span class="whattube-badge">${escapeHtml(langCode)}</span>
      <span>${escapeHtml(langName)} · original available</span>
    </div>
    <div class="whattube-original">“${escapeHtml(caption.original)}”</div>
  `;

  // Toggle original transcript on click
  const meta = card.querySelector(".whattube-meta");
  meta.addEventListener("click", () => {
    card.classList.toggle("show-original");
  });

  container.appendChild(card);
  activeCard = card;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function sendVideoSync(isSeek = false) {
  const video = document.querySelector("video");
  if (!video) return;

  lastSyncWallTime = performance.now();
  chrome.runtime.sendMessage({
    type: isSeek ? "VIDEO_SEEK" : "VIDEO_SYNC",
    video_time: video.currentTime,
    playback_rate: video.playbackRate || 1.0,
    video_title: document.title || "",
  }).catch(() => {});
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CAPTURE_STARTED") {
    sendVideoSync(false);
    sendResponse({ synced: true });
  } else if (message.type === "CAPTURE_STOPPED") {
    clearActiveCard();
    sendResponse({ stopped: true });
  } else if (message.type === "DISPLAY_CAPTION") {
    displayCaption(message.payload);
    sendResponse({ received: true });
  }
});

let currentAttachedVideo = null;

// Watch video events to synchronize timestamps and lifecycle
function initVideoListeners() {
  const video = document.querySelector("video");
  if (!video) return;

  // Send immediate sync on start or re-bind
  sendVideoSync(false);

  if (video === currentAttachedVideo) return;
  currentAttachedVideo = video;

  video.addEventListener("play", () => {
    sendVideoSync(false);
  });

  video.addEventListener("ratechange", () => {
    sendVideoSync(false);
  });

  video.addEventListener("seeking", () => {
    clearActiveCard();
    sendVideoSync(true);
  });

  video.addEventListener("timeupdate", () => {
    const now = performance.now();
    // Resync timeline every 2.5s during continuous playback
    if (now - lastSyncWallTime > 2500) {
      sendVideoSync(false);
    }

    // Dismiss active card only when video playback actually crosses dismissVideoTime
    if (activeCard && activeCardDismissVideoTime !== null) {
      if (video.currentTime >= activeCardDismissVideoTime) {
        clearActiveCard();
      }
    }
  });
}

// YouTube SPA Navigation Listeners (playlist transition, related video click, channel navigation)
window.addEventListener("yt-navigate-finish", () => {
  clearActiveCard();
  currentAttachedVideo = null;
  setTimeout(() => {
    initVideoListeners();
    sendVideoSync(true);
  }, 500);
});

window.addEventListener("yt-page-data-updated", () => {
  setTimeout(() => {
    sendVideoSync(true);
  }, 500);
});

// Initialize on DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initVideoListeners);
} else {
  initVideoListeners();
}
