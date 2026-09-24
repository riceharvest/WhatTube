"""Dynamic Event Aggregator for foreign chatter bursts."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AudioEvent:
    event_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    trigger_windows: List[Dict[str, Any]] = field(default_factory=list)
    opened_at: float = field(default_factory=time.time)
    closed_at: float = 0.0

class DynamicEventAggregator:
    """
    State machine that manages dynamic 3-7s foreign chatter bursts:
    - Opens upon immediate trigger (p_en < 0.20) or 2-in-3 suspicious (p_en < 0.55).
    - Anchors pre-roll to the first SUSPICIOUS window (not clean English history).
    - Extends dynamically while suspicious chatter persists.
    - Closes after ~0.8s hangover without non-English activity.
    - Emits chunks of 3-7s for optimal Whisper Turbo context.
    """

    def __init__(
        self,
        pre_roll_sec: float = 0.5,
        post_roll_sec: float = 0.5,
        close_hangover_sec: float = 0.8,
        min_event_sec: float = 3.0,
        max_event_sec: float = 8.0,
        lid_suspicious: float = 0.55,
        lid_immediate: float = 0.20,
        lid_english_safe: float = 0.75,
        consecutive_suspicious_req: int = 2,
    ):
        self.pre_roll_sec = pre_roll_sec
        self.post_roll_sec = post_roll_sec
        self.close_hangover_sec = close_hangover_sec
        self.min_event_sec = min_event_sec
        self.max_event_sec = max_event_sec
        self.lid_suspicious = lid_suspicious
        self.lid_immediate = lid_immediate
        self.lid_english_safe = lid_english_safe
        self.consecutive_suspicious_req = consecutive_suspicious_req

        self.history: deque = deque(maxlen=max(3, consecutive_suspicious_req))
        self.is_active = False
        self.event_id_counter = 0

        self.cur_event_id: Optional[int] = None
        self.event_start_time: float = 0.0
        self.event_last_active_time: float = 0.0
        self.cur_trigger_windows: List[Dict[str, Any]] = []

    def update(
        self,
        t_start: float,
        t_end: float,
        is_speech: bool,
        lid_result: Optional[Dict[str, Any]],
    ) -> Optional[AudioEvent]:
        """
        Process a new sliding window (typically 3s length, 1s stride).
        Returns an AudioEvent when an event closes or reaches max burst duration.
        """
        window_info = {
            "t_start": t_start,
            "t_end": t_end,
            "is_speech": is_speech,
            "lid": lid_result,
        }
        self.history.append(window_info)

        p_en = lid_result["p_en"] if (is_speech and lid_result) else 1.0
        is_immediate = is_speech and (p_en < self.lid_immediate)
        is_suspicious = is_speech and (p_en < self.lid_suspicious)

        # Check trigger criteria when IDLE
        if not self.is_active:
            suspicious_windows = [
                w for w in self.history
                if w["is_speech"] and w["lid"] and (w["lid"]["p_en"] < self.lid_suspicious)
            ]

            if is_immediate or len(suspicious_windows) >= self.consecutive_suspicious_req:
                # Open Event
                self.is_active = True
                self.event_id_counter += 1
                self.cur_event_id = self.event_id_counter

                # Anchor start time to the earliest SUSPICIOUS window (not clean English)
                first_susp_t = min(w["t_start"] for w in suspicious_windows) if suspicious_windows else t_start
                self.event_start_time = max(0.0, first_susp_t - self.pre_roll_sec)
                self.event_last_active_time = t_end
                self.cur_trigger_windows = [window_info]
                return None
            return None

        # When ACTIVE:
        # If confident English (p_en >= lid_english_safe), treat as non-suspicious to prevent event extension
        is_clean_english = is_speech and (p_en >= self.lid_english_safe)
        if (is_suspicious or is_immediate) and not is_clean_english:
            self.event_last_active_time = t_end
            self.cur_trigger_windows.append(window_info)

            # Check if burst reached max allowed duration (e.g. 6.5s)
            current_duration = self.event_last_active_time - self.event_start_time
            if current_duration >= self.max_event_sec:
                event = self._finalize_event(t_end)
                # Chain continuation if foreign activity continues
                self.is_active = True
                self.event_id_counter += 1
                self.cur_event_id = self.event_id_counter
                self.event_start_time = max(0.0, t_start - 0.2)
                self.event_last_active_time = t_end
                self.cur_trigger_windows = [window_info]
                return event
            return None

        # Active, but current window is clean English or silence
        if (t_end - self.event_last_active_time) >= self.close_hangover_sec:
            event = self._finalize_event(self.event_last_active_time + self.post_roll_sec)
            return event

        return None

    def check_hangover_timeout(self, cur_time: float) -> Optional[AudioEvent]:
        """Check if active event has timed out due to stream pause or inactivity."""
        if self.is_active and (cur_time - self.event_last_active_time) >= self.close_hangover_sec:
            return self._finalize_event(self.event_last_active_time + self.post_roll_sec)
        return None

    def _finalize_event(self, end_time: float) -> Optional[AudioEvent]:
        start = self.event_start_time
        end = max(end_time, start + self.min_event_sec)
        duration = end - start

        event = AudioEvent(
            event_id=self.cur_event_id or self.event_id_counter,
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            duration_sec=round(duration, 2),
            trigger_windows=list(self.cur_trigger_windows),
            closed_at=time.time(),
        )

        self.is_active = False
        self.cur_event_id = None
        self.cur_trigger_windows = []
        return event

    def flush(self, current_time: float) -> Optional[AudioEvent]:
        """Force finalize any pending active event."""
        if self.is_active:
            return self._finalize_event(current_time)
        return None

    def reset(self):
        """Reset aggregator state upon seek or session reset."""
        self.history.clear()
        self.is_active = False
        self.cur_event_id = None
        self.event_start_time = 0.0
        self.event_last_active_time = 0.0
        self.cur_trigger_windows = []

# Alias for backward compatibility
SpeechEvent = AudioEvent

