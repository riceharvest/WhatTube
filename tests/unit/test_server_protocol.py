"""Unit tests for WebSocket protocol, authentication, epoch invalidation, and session isolation."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import websockets

from whattube.config import Config
from whattube.server import WhatTubeServer, SessionState
from whattube.event_aggregator import AudioEvent

@pytest.fixture
def mock_server():
    config = Config(
        ws_host="127.0.0.1",
        ws_port=8799,
        auth_token="secret-token-1234",
    )
    with patch("whattube.server.ResidentASRClient"), \
         patch("whattube.server.MarianTranslator"), \
         patch("whattube.server.EnergyAndSileroVAD"), \
         patch("whattube.server.WhisperTinyLID"):
        server = WhatTubeServer(config)
        yield server

@pytest.mark.asyncio
async def test_reject_binary_pcm_before_init(mock_server):
    port = 8791
    mock_server.config.ws_port = port
    
    async with websockets.serve(mock_server.handle_client, "127.0.0.1", port):
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            # Send raw binary PCM before init handshake
            dummy_pcm = np.zeros(1024, dtype=np.float32).tobytes()
            await ws.send(dummy_pcm)
            
            with pytest.raises(websockets.ConnectionClosed) as exc_info:
                await ws.recv()
            assert exc_info.value.code == 4403

@pytest.mark.asyncio
async def test_reject_invalid_auth_token(mock_server):
    port = 8792
    mock_server.config.ws_port = port
    
    async with websockets.serve(mock_server.handle_client, "127.0.0.1", port):
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            # Send init with wrong token
            await ws.send(json.dumps({
                "type": "init",
                "token": "wrong-token",
                "video_time": 0.0,
            }))
            
            # Should receive error message then close
            resp = await ws.recv()
            data = json.loads(resp)
            assert data.get("type") == "error"
            
            with pytest.raises(websockets.ConnectionClosed) as exc_info:
                await ws.recv()
            assert exc_info.value.code == 4401

@pytest.mark.asyncio
async def test_accept_valid_auth_token(mock_server):
    port = 8793
    mock_server.config.ws_port = port
    
    async with websockets.serve(mock_server.handle_client, "127.0.0.1", port):
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            # Send init with correct token
            await ws.send(json.dumps({
                "type": "init",
                "token": "secret-token-1234",
                "video_time": 42.0,
                "playback_rate": 1.5,
                "target_lang": "es",
            }))
            
            resp = await ws.recv()
            data = json.loads(resp)
            assert data.get("type") == "ready"
            assert "session_id" in data
            
            # Verify session was registered with anchor
            assert len(mock_server.sessions) == 1
            session = list(mock_server.sessions.values())[0]
            assert session.target_lang == "es"
            assert session.playback_rate == 1.5
            assert session.anchor_video_time == 42.0

@pytest.mark.asyncio
async def test_stale_burst_discarded_after_seek_during_asr_queue(mock_server):
    """If epoch increments while an ASR burst is queued or running, it must be discarded."""
    ws = AsyncMock()
    session = SessionState(
        session_id="test-session",
        websocket=ws,
        config=mock_server.config,
    )
    mock_server.sessions[session.session_id] = session
    
    event = AudioEvent(
        event_id=1,
        start_sec=1.0,
        end_sec=4.0,
        duration_sec=3.0,
    )
    
    # Populate session audio buffer so get_slice succeeds
    dummy_audio = np.zeros(16000 * 6, dtype=np.float32)
    session.ring_buffer.append(dummy_audio)

    # Simulate slow ASR
    def delayed_asr(*args, **kwargs):
        import time
        time.sleep(0.2)
        res = MagicMock()
        res.language = "id"
        res.text = "Selamat pagi"
        res.is_discarded = False
        return res
    
    mock_server.asr_client.transcribe = MagicMock(side_effect=delayed_asr)
    
    # Launch burst task in epoch 0
    burst_task = asyncio.create_task(
        mock_server.handle_event_burst(session, event, event_epoch=0, stage1_latency_ms=10.0)
    )
    
    # User seeks immediately, incrementing epoch to 1
    await asyncio.sleep(0.05)
    session.increment_epoch(new_video_time=150.0)
    
    await burst_task
    
    # Caption must NOT be emitted to websocket because epoch no longer matches
    ws.send.assert_not_called()

@pytest.mark.asyncio
async def test_multi_client_session_private_routing(mock_server):
    """Verify captions sent to client A are never broadcast to client B."""
    ws_a = AsyncMock()
    ws_b = AsyncMock()
    
    session_a = SessionState("sess-a", ws_a, mock_server.config)
    session_b = SessionState("sess-b", ws_b, mock_server.config)
    mock_server.sessions["sess-a"] = session_a
    mock_server.sessions["sess-b"] = session_b
    
    event = AudioEvent(
        event_id=99,
        start_sec=2.0,
        end_sec=5.0,
        duration_sec=3.0,
    )
    
    mock_asr = MagicMock()
    mock_asr.language = "es"
    mock_asr.text = "Hola amigo"
    mock_asr.is_discarded = False
    mock_server.asr_client.transcribe = MagicMock(return_value=mock_asr)
    
    mock_trans = MagicMock()
    mock_trans.translated_text = "Hello friend"
    mock_server.translator.translate = MagicMock(return_value=mock_trans)
    
    # Populate session_a's audio buffer so get_slice(2.0, 5.0) succeeds
    dummy_audio = np.zeros(16000 * 6, dtype=np.float32)
    session_a.ring_buffer.append(dummy_audio)
    
    # Run event on session A
    await mock_server.handle_event_burst(session_a, event, event_epoch=session_a.epoch, stage1_latency_ms=5.0)
    
    # Client A must receive caption
    assert ws_a.send.call_count == 1
    payload = json.loads(ws_a.send.call_args[0][0])
    assert payload["type"] == "caption"
    assert payload["translation"] == "Hello friend"
    
    # Client B must receive NOTHING
    ws_b.send.assert_not_called()

def test_title_hints_extraction():
    from whattube.server import extract_language_hints_from_title
    assert "ja" in extract_language_hints_from_title("Japanese Food Tour - HIDDEN-GEMS in Tokyo, Japan")
    assert "bn" in extract_language_hints_from_title("I Traveled to Dhaka, Bangladesh at 3AM")
    assert "id" in extract_language_hints_from_title("41 Hours in Jakarta, Indonesia")
    assert "th" in extract_language_hints_from_title("Thailand Street Food - 5 MUST-EAT Thai Noodle Soups in Bangkok!!")
    assert extract_language_hints_from_title("Random video without country") == set()

@pytest.mark.asyncio
async def test_out_of_domain_hallucination_filtered(mock_server):
    """Verify that a low-probability out-of-domain language (e.g. Russian in Dhaka) is discarded."""
    ws = AsyncMock()
    session = SessionState("sess-dhaka", ws, mock_server.config, video_title="I Traveled to Dhaka, Bangladesh")
    mock_server.sessions["sess-dhaka"] = session

    event = AudioEvent(event_id=1, start_sec=1.0, end_sec=4.0, duration_sec=3.0)

    # Low-prob Russian hallucination (prob=0.33 < 0.40) in Bangladesh video
    mock_asr = MagicMock()
    mock_asr.language = "ru"
    mock_asr.language_prob = 0.33
    mock_asr.text = "Пойди сюда"
    mock_asr.is_discarded = False
    mock_asr.discard_reason = ""
    mock_server.asr_client.transcribe = MagicMock(return_value=mock_asr)

    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    session.ring_buffer.append(dummy_audio)

    await mock_server.handle_event_burst(session, event, event_epoch=session.epoch, stage1_latency_ms=5.0)

    # Discarded: no caption sent to client
    ws.send.assert_not_called()
