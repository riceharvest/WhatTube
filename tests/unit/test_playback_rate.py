"""Unit tests for playback rate timeline anchor mapping."""

import pytest
from unittest.mock import MagicMock
from whattube.config import Config
from whattube.server import SessionState

def make_session(initial_vtime=0.0, initial_rate=1.0):
    ws = MagicMock()
    config = Config(auth_token="test-token")
    session = SessionState(
        session_id="test-session",
        websocket=ws,
        config=config,
        initial_video_time=initial_vtime,
        initial_playback_rate=initial_rate,
    )
    return session

@pytest.mark.parametrize("rate,audio_elapsed,expected_video_elapsed", [
    (0.5, 2.0, 1.0),
    (1.0, 2.0, 2.0),
    (1.5, 2.0, 3.0),
    (2.0, 2.0, 4.0),
])
def test_playback_rate_linear_mapping(rate, audio_elapsed, expected_video_elapsed):
    session = make_session(initial_vtime=100.0, initial_rate=rate)
    
    # After audio_elapsed seconds in the buffer:
    calc_vtime = session.audio_to_video_time(audio_elapsed)
    assert pytest.approx(calc_vtime, 0.001) == 100.0 + expected_video_elapsed

def test_playback_ratechange_reanchoring():
    session = make_session(initial_vtime=50.0, initial_rate=1.0)
    
    # 10s of audio play at 1.0x -> video is at 60.0s
    calc_v1 = session.audio_to_video_time(10.0)
    assert pytest.approx(calc_v1, 0.001) == 60.0
    
    # User switches to 2.0x speed at audio_time=10.0s, video_time=60.0s
    session.set_anchor(audio_time=10.0, video_time=60.0, playback_rate=2.0)
    assert session.playback_rate == 2.0
    
    # 2 seconds later in audio (audio_time=12.0s):
    # At 2x, 2s audio = 4s video -> video is at 64.0s
    calc_v2 = session.audio_to_video_time(12.0)
    assert pytest.approx(calc_v2, 0.001) == 64.0

    # User switches to 0.5x speed at audio_time=12.0s, video_time=64.0s
    session.set_anchor(audio_time=12.0, video_time=64.0, playback_rate=0.5)
    # 4 seconds later in audio (audio_time=16.0s):
    # At 0.5x, 4s audio = 2s video -> video is at 66.0s
    calc_v3 = session.audio_to_video_time(16.0)
    assert pytest.approx(calc_v3, 0.001) == 66.0

def test_seek_reanchors_and_increments_epoch():
    session = make_session(initial_vtime=10.0, initial_rate=1.5)
    old_epoch = session.epoch
    
    # User seeks to video time 300.0s
    session.increment_epoch(new_video_time=300.0, playback_rate=1.5)
    
    assert session.epoch == old_epoch + 1
    assert session.anchor_audio_time == 0.0
    assert session.anchor_video_time == 300.0
    assert session.playback_rate == 1.5
    
    # Fresh audio at audio_time=1.0s should be at 300.0 + 1.5 = 301.5s
    assert pytest.approx(session.audio_to_video_time(1.0), 0.001) == 301.5
