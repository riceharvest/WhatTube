"""Unit tests for DynamicEventAggregator."""

from whattube.event_aggregator import DynamicEventAggregator

def test_single_immediate_trigger():
    agg = DynamicEventAggregator(pre_roll_sec=1.0, post_roll_sec=0.5, close_hangover_sec=0.8)

    # Window 0: Clean English
    ev = agg.update(0.0, 3.0, True, {"p_en": 0.85})
    assert ev is None
    assert not agg.is_active

    # Window 1: Indonesian chatter (p_en = 0.05 < 0.20)
    ev = agg.update(1.0, 4.0, True, {"p_en": 0.05})
    assert ev is None
    assert agg.is_active

    # Window 2: Still chatter (p_en = 0.10)
    ev = agg.update(2.0, 5.0, True, {"p_en": 0.10})
    assert ev is None
    assert agg.is_active

    # Window 3: Host speaks English (p_en = 0.90) -> hangover starts
    ev = agg.update(3.0, 6.0, True, {"p_en": 0.90})
    # Window 3 ends at 6.0, last active was 5.0. 6.0 - 5.0 = 1.0 > 0.8 hangover!
    assert ev is not None
    assert not agg.is_active
    # Start time should include pre-roll (1.0s before earliest window)
    assert ev.start_sec <= 1.0
    assert ev.duration_sec >= 2.5
    print("test_single_immediate_trigger passed:", ev)

def test_two_suspicious_windows():
    agg = DynamicEventAggregator(pre_roll_sec=1.0, post_roll_sec=0.5, close_hangover_sec=0.8)

    # Window 0: Clean English
    ev = agg.update(0.0, 3.0, True, {"p_en": 0.80})
    assert not agg.is_active

    # Window 1: Suspicious (0.45)
    ev = agg.update(1.0, 4.0, True, {"p_en": 0.45})
    assert not agg.is_active  # 1 suspicious is not enough

    # Window 2: Suspicious (0.40) -> 2 out of 3 suspicious -> trigger!
    ev = agg.update(2.0, 5.0, True, {"p_en": 0.40})
    assert agg.is_active

    # Window 3: Clean English (0.85) -> closes after hangover
    ev = agg.update(3.0, 6.0, True, {"p_en": 0.85})
    assert ev is not None
    print("test_two_suspicious_windows passed:", ev)

def test_english_onset_truncation():
    """Verify that confident English resumption closes the event immediately without padding."""
    agg = DynamicEventAggregator(pre_roll_sec=0.5, post_roll_sec=0.5, close_hangover_sec=0.8)

    # Window 0: Foreign chatter trigger (Hindi / Indonesian)
    ev = agg.update(10.0, 13.0, True, {"p_en": 0.05})
    assert agg.is_active

    # Window 1: Continued foreign chatter
    ev = agg.update(11.0, 14.0, True, {"p_en": 0.10})
    assert agg.is_active

    # Window 2: Host resumes clean English (p_en = 0.92 >= lid_english_safe)
    ev = agg.update(12.0, 15.0, True, {"p_en": 0.92})
    assert ev is not None
    assert not agg.is_active
    # End time should be pinned to the last foreign window end (14.0s), NOT padded with post_roll (14.5s)
    assert ev.end_sec == 14.0
    assert ev.start_sec == 9.5  # 10.0 - 0.5 pre_roll

if __name__ == "__main__":
    test_single_immediate_trigger()
    test_two_suspicious_windows()
    test_english_onset_truncation()
    print("All aggregator tests passed!")
