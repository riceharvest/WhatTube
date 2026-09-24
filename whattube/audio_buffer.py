"""Rolling ring buffer with timestamp-indexed audio slicing."""

import numpy as np
import threading
from typing import Optional, Tuple

class AudioRingBuffer:
    def __init__(self, sample_rate: int = 16000, capacity_sec: float = 30.0):
        self.sample_rate = sample_rate
        self.capacity_samples = int(capacity_sec * sample_rate)
        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_samples = 0
        self.lock = threading.Lock()

    def append(self, pcm_data: np.ndarray) -> Tuple[float, float]:
        """
        Append float32 PCM samples (-1.0 to 1.0).
        Returns (chunk_start_sec, chunk_end_sec) relative to playback stream.
        """
        if pcm_data.ndim > 1:
            pcm_data = pcm_data.mean(axis=1)
        pcm_data = pcm_data.astype(np.float32)

        with self.lock:
            n_samples = len(pcm_data)
            start_sec = self.total_samples / self.sample_rate
            end_sec = (self.total_samples + n_samples) / self.sample_rate

            if n_samples >= self.capacity_samples:
                # Overwrite entire buffer with tail
                self.buffer[:] = pcm_data[-self.capacity_samples:]
                self.write_pos = 0
            else:
                end_pos = (self.write_pos + n_samples) % self.capacity_samples
                if self.write_pos + n_samples <= self.capacity_samples:
                    self.buffer[self.write_pos : self.write_pos + n_samples] = pcm_data
                else:
                    first_part = self.capacity_samples - self.write_pos
                    self.buffer[self.write_pos :] = pcm_data[:first_part]
                    self.buffer[: n_samples - first_part] = pcm_data[first_part:]
                self.write_pos = end_pos

            self.total_samples += n_samples
            return start_sec, end_sec

    @property
    def current_time_sec(self) -> float:
        with self.lock:
            return self.total_samples / self.sample_rate

    def get_slice(self, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
        """
        Extract a contiguous float32 slice between [start_sec, end_sec].
        Returns None if requested range is unavailable (e.g. purged from ring).
        """
        with self.lock:
            cur_time = self.total_samples / self.sample_rate
            oldest_time = max(0.0, cur_time - (self.capacity_samples / self.sample_rate))

            # Clamp or validate range
            if end_sec <= oldest_time or start_sec >= cur_time:
                return None

            req_start = max(start_sec, oldest_time)
            req_end = min(end_sec, cur_time)

            n_samples = int(round((req_end - req_start) * self.sample_rate))
            if n_samples <= 0:
                return None

            # Calculate index relative to current write_pos
            samples_ago = int(round((cur_time - req_start) * self.sample_rate))
            idx_start = (self.write_pos - samples_ago) % self.capacity_samples

            if idx_start + n_samples <= self.capacity_samples:
                out = self.buffer[idx_start : idx_start + n_samples].copy()
            else:
                first_part = self.capacity_samples - idx_start
                out = np.empty(n_samples, dtype=np.float32)
                out[:first_part] = self.buffer[idx_start:]
                out[first_part:] = self.buffer[: n_samples - first_part]

            return out

    def get_latest(self, duration_sec: float) -> Optional[np.ndarray]:
        """Get the most recent N seconds of audio."""
        with self.lock:
            cur_time = self.total_samples / self.sample_rate
            return self.get_slice(cur_time - duration_sec, cur_time)

    def reset(self):
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.total_samples = 0
