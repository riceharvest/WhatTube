"""Audit Logger for event precision, classification, and diagnostics."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Union

class EventAuditLogger:
    def __init__(self, log_path: Union[str, Path] = "events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_events = 0
        self.emitted_events = 0
        self.discarded_english = 0
        self.discarded_hallucination = 0

    def classify_event(self, a_lang: str, text: str, is_discarded: bool, discard_reason: str) -> str:
        if is_discarded:
            if "English" in discard_reason or a_lang.lower() == "en":
                return "ENGLISH_HOST"
            if "Empty" in discard_reason or "hallucinated" in discard_reason:
                return "HALLUCINATION_OR_NOISE"
            return "DISCARDED_OTHER"

        if a_lang.lower() not in ["en", "unknown", "error"]:
            return "FOREIGN_SPEECH"

        return "UNCERTAIN"

    def log_event(
        self,
        event_id: int,
        start_sec: float,
        end_sec: float,
        duration_sec: float,
        trigger_reason: str,
        asr_lang: str,
        original_text: str,
        translated_text: str,
        is_emitted: bool,
        discard_reason: str,
        timings: Dict[str, float],
    ):
        classification = self.classify_event(asr_lang, original_text, not is_emitted, discard_reason)

        self.total_events += 1
        if is_emitted:
            self.emitted_events += 1
        elif classification == "ENGLISH_HOST":
            self.discarded_english += 1
        elif classification == "HALLUCINATION_OR_NOISE":
            self.discarded_hallucination += 1

        entry = {
            "timestamp": time.time(),
            "event_id": event_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "classification": classification,
            "trigger_reason": trigger_reason,
            "asr_lang": asr_lang,
            "original_text": original_text,
            "translated_text": translated_text,
            "emitted_to_overlay": is_emitted,
            "discard_reason": discard_reason,
            "timings_ms": timings,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def print_summary(self):
        print(f"\n=== WHATTUBE EVENT AUDIT SUMMARY ===")
        print(f"Total GPU Bursts Triggered: {self.total_events}")
        print(f"Emitted Foreign Captions:   {self.emitted_events}")
        print(f"Filtered Clean English:     {self.discarded_english}")
        print(f"Filtered Noise/Hallucinate: {self.discarded_hallucination}")
        if self.total_events > 0:
            precision = (self.emitted_events / self.total_events) * 100.0
            print(f"Emitted Precision:          {precision:.1f}%")
        print("=====================================\n")
