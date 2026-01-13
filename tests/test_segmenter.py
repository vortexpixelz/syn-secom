from src.data.segmenter import segment_dialogue_fixed


def test_segment_dialogue_fixed_creates_segments():
    turns = ["hi", "hello", "how are you", "fine", "bye"]
    segments = segment_dialogue_fixed(turns, window_size=2, dialogue_id=1)

    assert len(segments) == 3
    assert segments[0]["segment_id"] == "1-0"
    assert segments[0]["text"] == "hi hello"
