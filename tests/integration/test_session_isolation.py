"""Tests for multi-client SessionState isolation and epoch cancellation."""

import asyncio
import json
import numpy as np
import pytest
from whattube.config import default_config
from whattube.server import SessionState, WhatTubeServer

class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send(self, data):
        self.sent_messages.append(data)

    async def close(self, code=1000, reason=""):
        self.closed = True

def test_session_state_isolation():
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    s1 = SessionState("session-1", ws1, default_config, target_lang="en")
    s2 = SessionState("session-2", ws2, default_config, target_lang="es")

    # Tab 1 appends audio
    pcm = np.ones(16000 * 2, dtype=np.float32)
    s1.ring_buffer.append(pcm)

    # Tab 1 should have 2s; Tab 2 must have 0s
    assert s1.ring_buffer.current_time_sec == 2.0
    assert s2.ring_buffer.current_time_sec == 0.0

    # Tab 1 changes epoch via seek
    s1.increment_epoch(new_video_time=120.0)
    assert s1.epoch == 1
    assert s1.anchor_video_time == 120.0
    assert s1.ring_buffer.current_time_sec == 0.0

    # Tab 2's epoch and state must remain untouched
    assert s2.epoch == 0
    assert s2.anchor_video_time == 0.0
    assert s2.target_lang == "es"

@pytest.mark.asyncio
async def test_epoch_cancellation_discards_stale_burst():
    """Verify that an in-flight burst from an older epoch is discarded upon seek."""
    server = WhatTubeServer(default_config)
    ws = MockWebSocket()
    session = SessionState("session-test", ws, default_config)

    # Append speech-like samples
    pcm = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 4.0, 64000))).astype(np.float32)
    session.ring_buffer.append(pcm)

    # Simulate an active event from epoch 0
    from whattube.event_aggregator import SpeechEvent
    event = SpeechEvent(event_id=1, start_sec=0.5, end_sec=3.5, duration_sec=3.0)

    # User seeks halfway through event execution -> epoch increments to 1
    session.increment_epoch(new_video_time=45.0)

    # Run handle_event_burst with OLD epoch 0
    await server.handle_event_burst(session, event, event_epoch=0, stage1_latency_ms=10.0)

    # Result must NOT be sent to client
    assert len(ws.sent_messages) == 0
