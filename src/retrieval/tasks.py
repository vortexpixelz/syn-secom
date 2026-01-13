from __future__ import annotations

import random
from typing import Dict, List, Tuple


def make_queries_from_segments(
    segments: List[Dict[str, object]], num_queries: int
) -> Tuple[List[str], List[str]]:
    if not segments:
        return [], []
    indices = list(range(len(segments)))
    sample_indices = random.sample(indices, k=min(num_queries, len(indices)))
    queries = []
    gold_segment_ids = []
    for idx in sample_indices:
        segment = segments[idx]
        text = str(segment["text"])
        turns = text.split(".")
        query = turns[-1].strip() if turns else text
        queries.append(query or text)
        gold_segment_ids.append(str(idx))
    return queries, gold_segment_ids
