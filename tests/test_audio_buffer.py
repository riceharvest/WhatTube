"""Unit tests for AudioRingBuffer."""

import numpy as np
import pytest
from whattube.audio_buffer import AudioRingBuffer

def test_audio_ring_buffer_basic_append_and_slice():
    buf = AudioRingBuffer(sample_rate=16000, capacity_sec=10.0)
    assert buf.current_time_sec == 0.0

    # Append 2 seconds of 440Hz sine wave
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    start_sec, end_sec = buf.append(sine)

    assert start_sec == 0.0
    assert end_sec == 2.0
    assert np.isclose(buf.current_time_sec, 2.0)

    # Slice 0.5s to 1.5s (16000 samples)
    sl = buf.get_slice(0.5, 1.5)
    assert sl is not None
    assert len(sl) == 16000
    assert np.allclose(sl, sine[8000:24000], atol=1e-5)

def test_audio_ring_buffer_get_latest_no_deadlock():
    """Verify that get_latest() successfully retrieves samples without deadlocking on re-entrancy."""
    buf = AudioRingBuffer(sample_rate=16000, capacity_sec=5.0)
    data = np.ones(16000 * 3, dtype=np.float32) * 0.42
    buf.append(data)

    latest = buf.get_latest(1.0)
    assert latest is not None
    assert len(latest) == 16000
    assert np.allclose(latest, 0.42)

def test_audio_ring_buffer_wrap_around():
    """Verify circular buffer wrap-around when capacity is exceeded."""
    buf = AudioRingBuffer(sample_rate=16000, capacity_sec=2.0) # Capacity: 32000 samples

    chunk1 = np.ones(16000, dtype=np.float32) * 1.0  # 0s - 1s
    chunk2 = np.ones(16000, dtype=np.float32) * 2.0  # 1s - 2s
    chunk3 = np.ones(16000, dtype=np.float32) * 3.0  # 2s - 3s (purges chunk 1)

    buf.append(chunk1)
    buf.append(chunk2)
    buf.append(chunk3)

    assert buf.current_time_sec == 3.0

    # Chunk 1 (0s to 1s) should be purged
    assert buf.get_slice(0.0, 0.8) is None

    # Slice across boundary (1.5s to 2.5s)
    sl = buf.get_slice(1.5, 2.5)
    assert sl is not None
    assert len(sl) == 16000
    # First half should be 2.0, second half 3.0
    assert np.allclose(sl[:8000], 2.0)
    assert np.allclose(sl[8000:], 3.0)

def test_audio_ring_buffer_reset():
    buf = AudioRingBuffer(sample_rate=16000, capacity_sec=5.0)
    buf.append(np.ones(16000, dtype=np.float32))
    assert buf.current_time_sec == 1.0

    buf.reset()
    assert buf.current_time_sec == 0.0
    assert buf.get_slice(0.0, 1.0) is None
