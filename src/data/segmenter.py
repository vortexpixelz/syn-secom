from __future__ import annotations

from typing import Dict, List


def segment_dialogue_fixed(
    turns: List[str], window_size: int, dialogue_id: int
) -> List[Dict[str, object]]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    segments: List[Dict[str, object]] = []
    for idx in range(0, len(turns), window_size):
        window = turns[idx : idx + window_size]
        if not window:
            continue
        segments.append(
            {
                "dialogue_id": dialogue_id,
                "segment_id": f"{dialogue_id}-{idx}",
                "turn_start": idx,
                "turn_end": idx + len(window) - 1,
                "text": " ".join(window),
            }
        )
    return segments
