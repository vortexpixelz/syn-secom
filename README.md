# Syn-SECOM: Synesthetic Tagging for Segmented Memory

Syn-SECOM is a lightweight research prototype that tests whether adding synesthetic metadata
(Emotion → Color → Tone) to SeCOM-style memory segments improves retrieval quality for dialogue
systems. The pipeline loads open datasets, segments conversations into fixed windows, tags each
segment, embeds the baseline and tagged variants, then evaluates retrieval performance.

## Project goals
- Demonstrate a clear, reproducible experiment for synesthetic tagging.
- Provide a minimal baseline you can extend with improved taggers, embeddings, or evaluation.

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --dataset personachat --subset_size 1000 --window_size 4 --top_k 5
```

## Quickstart (Google Colab)
```bash
!unzip syn-secom.zip -d syn-secom
%cd syn-secom
!pip -q install -r requirements.txt
!python run.py --dataset personachat --subset_size 1000 --window_size 4 --top_k 5
```

## Example output
The run writes artifacts to `outputs/`:
- `segments.jsonl` — each segment with tags
- `queries.jsonl` — sampled queries and gold segment IDs
- `report.json` — baseline vs synesthetic retrieval metrics

## Repository structure
- `src/data/loaders.py` — loads PersonaChat (Hugging Face) or Topical-Chat (GitHub JSON).
- `src/data/segmenter.py` — fixed-window segmentation.
- `src/tagging/` — rule-based emotion tagging and synesthetic mapping.
- `src/embed/` — embedding + FAISS index helpers.
- `src/retrieval/` — query generation and evaluation (P@1, Recall@K, MRR, gain).
- `run.py` — end-to-end pipeline entry point.

## Notes
- The default tagger is intentionally simple; swap in a transformer-based classifier for higher fidelity.
- For paper-grade runs, increase `subset_size` and run multiple seeds or datasets.

## Tests
```bash
pytest
```
