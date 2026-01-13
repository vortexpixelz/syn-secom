from __future__ import annotations

from typing import Dict, List


def _precision_at_k(gold: str, preds: List[str], k: int) -> float:
    return 1.0 if gold in preds[:k] else 0.0


def _recall_at_k(gold: str, preds: List[str], k: int) -> float:
    return 1.0 if gold in preds[:k] else 0.0


def _mrr(gold: str, preds: List[str]) -> float:
    for idx, pred in enumerate(preds, start=1):
        if gold == pred:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval(
    gold_segment_ids: List[str],
    base_nn: List[List[int]],
    syn_nn: List[List[int]],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    metrics = {"baseline": {"p_at_1": 0.0, "recall_at_k": 0.0, "mrr": 0.0}}
    metrics["synesthetic"] = {"p_at_1": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    for gold, base_preds, syn_preds in zip(gold_segment_ids, base_nn, syn_nn):
        base_preds = [str(p) for p in base_preds]
        syn_preds = [str(p) for p in syn_preds]
        metrics["baseline"]["p_at_1"] += _precision_at_k(gold, base_preds, 1)
        metrics["baseline"]["recall_at_k"] += _recall_at_k(gold, base_preds, top_k)
        metrics["baseline"]["mrr"] += _mrr(gold, base_preds)
        metrics["synesthetic"]["p_at_1"] += _precision_at_k(gold, syn_preds, 1)
        metrics["synesthetic"]["recall_at_k"] += _recall_at_k(gold, syn_preds, top_k)
        metrics["synesthetic"]["mrr"] += _mrr(gold, syn_preds)

    total = max(len(gold_segment_ids), 1)
    for block in metrics.values():
        for key in block:
            block[key] = block[key] / total
    metrics["synesthetic"]["compression_gain"] = (
        metrics["synesthetic"]["recall_at_k"] - metrics["baseline"]["recall_at_k"]
    )
    return metrics


def print_report(report: Dict[str, Dict[str, float]]) -> None:
    for label, block in report.items():
        print(f"[{label}]")
        for key, value in block.items():
            print(f"  {key}: {value:.3f}")
