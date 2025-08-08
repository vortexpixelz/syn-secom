#!/usr/bin/env python3
import argparse, os, json, math, random, time
from pathlib import Path
from typing import List, Dict, Any

from src.data.loaders import load_personachat, load_topical_chat
from src.data.segmenter import segment_dialogue_fixed
from src.tagging.tags import Tagger
from src.embed.embedder import Embedder
from src.embed.index import build_faiss, search_faiss
from src.retrieval.tasks import make_queries_from_segments
from src.retrieval.eval import evaluate_retrieval, print_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["personachat", "topical_chat"], default="personachat")
    parser.add_argument("--subset_size", type=int, default=1000, help="max number of turns to load (approx)")
    parser.add_argument("--window_size", type=int, default=4, help="turns per segment (2–8 recommended)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # 1) Load dataset -> list[dialogue], each dialogue is list[turn_text]
    if args.dataset == "personachat":
        dialogues = load_personachat(subset_size=args.subset_size)
    else:
        dialogues = load_topical_chat(subset_size=args.subset_size)

    # 2) Segment into SeCOM-style segments
    segments = []
    for d_id, turns in enumerate(dialogues):
        segs = segment_dialogue_fixed(turns, window_size=args.window_size, dialogue_id=d_id)
        segments.extend(segs)

    print(f"[INFO] Segments: {len(segments)}")

    # 3) Tag segments with synesthetic metadata
    tagger = Tagger()
    for s in segments:
        s["tags"] = tagger.tag_segment(s["text"])

    # 4) Build two text views
    baseline_texts = [s["text"] for s in segments]
    syn_texts = [s["text"] + " || " + tagger.as_string(s["tags"]) for s in segments]

    # 5) Embeddings + FAISS
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_vecs = embedder.encode(baseline_texts, normalize=True)
    syn_vecs  = embedder.encode(syn_texts, normalize=True)

    base_index = build_faiss(base_vecs)
    syn_index  = build_faiss(syn_vecs)

    # 6) Make queries from held-out turns (simple simulation: ask about content in segment)
    queries, gold_segment_ids = make_queries_from_segments(segments, num_queries=min(200, len(segments)))

    # 7) Search
    base_nn = search_faiss(base_index, base_vecs, embedder, queries, top_k=args.top_k)
    syn_nn  = search_faiss(syn_index,  syn_vecs,  embedder, queries, top_k=args.top_k)

    # 8) Evaluate
    report = evaluate_retrieval(gold_segment_ids, base_nn, syn_nn, top_k=args.top_k)
    print_report(report)

    # 9) Save artifacts
    out = Path("outputs"); out.mkdir(exist_ok=True)
    with open(out / "segments.jsonl", "w") as f:
        for s in segments:
            f.write(json.dumps(s) + "\n")
    with open(out / "queries.jsonl", "w") as f:
        for q, gold in zip(queries, gold_segment_ids):
            f.write(json.dumps({"query": q, "gold": gold}) + "\n")
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
