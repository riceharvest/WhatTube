// WhatTube Content Script for YouTube Player Overlay

let overlayContainer = null;
let activeCard = null;
let dismissTimeout = null;

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

function displayCaption(caption) {
  const container = ensureOverlayContainer();
  if (!container) return;

  if (dismissTimeout) {
    clearTimeout(dismissTimeout);
    dismissTimeout = null;
  }

  // If active card exists, smoothly morph text in-place
  if (activeCard && activeCard.parentNode) {
    const transEl = activeCard.querySelector(".whattube-translation");
    const origEl = activeCard.querySelector(".whattube-original");
    const badgeEl = activeCard.querySelector(".whattube-badge");

    if (transEl && origEl) {
      transEl.textContent = caption.translation;
      origEl.textContent = `“${caption.original}”`;
      if (badgeEl) badgeEl.textContent = langCode;
      activeCard.classList.remove("whattube-fade-out");

      // Reset auto-dismiss timer
      const displayDurationMs = Math.max(4500, (caption.end - caption.start + 2.5) * 1000);
      dismissTimeout = setTimeout(() => {
        activeCard.classList.add("whattube-fade-out");
        setTimeout(() => {
          if (activeCard && activeCard.parentNode) {
            activeCard.parentNode.removeChild(activeCard);
          }
          activeCard = null;
        }, 400);
      }, displayDurationMs);
      return;
    }
  }

  const card = document.createElement("div");
  card.className = "whattube-card";

  const langCode = (caption.language || "unknown").toLowerCase();
  const langName = LANGUAGE_NAMES[langCode] || langCode.toUpperCase();

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

  // Auto-dismiss after display duration (minimum 4.5s)
  const displayDurationMs = Math.max(4500, (caption.end - caption.start + 2.0) * 1000);
  dismissTimeout = setTimeout(() => {
    card.classList.add("whattube-fade-out");
    setTimeout(() => {
      if (card.parentNode) {
        card.parentNode.removeChild(card);
      }
      if (activeCard === card) {
        activeCard = null;
      }
    }, 400);
  }, displayDurationMs);
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

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "DISPLAY_CAPTION") {
    displayCaption(message.payload);
    sendResponse({ received: true });
  }
});

// Watch for video seeks to clear stale subtitles
function initVideoListeners() {
  const video = document.querySelector("video");
  if (video) {
    video.addEventListener("seeking", () => {
      if (activeCard && activeCard.parentNode) {
        activeCard.parentNode.removeChild(activeCard);
        activeCard = null;
      }
    });
  }
}

// Initialize on DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initVideoListeners);
} else {
  initVideoListeners();
}
