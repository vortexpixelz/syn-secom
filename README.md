# Syn-SECOM: Synesthetic Tagging for Segmented Memory

This repo is a **minimal, runnable skeleton** to test whether adding synesthetic metadata
(Emotion → Color → Tone) to SeCOM-style segments improves retrieval and response quality.

## Quickstart (Google Colab)
1) **Upload** this zip to Colab (left sidebar → Files → Upload).
2) Unzip and install deps:
```bash
!unzip syn-secom.zip -d syn-secom
%cd syn-secom
!pip -q install -r requirements.txt
```
3) Run a pilot on **PersonaChat** (fast, no auth):
```bash
!python run.py --dataset personachat --subset_size 1000 --window_size 4 --top_k 5
```
4) Optional: Try **Topical-Chat** (pulls JSON from GitHub):
```bash
!python run.py --dataset topical_chat --subset_size 1000 --window_size 4 --top_k 5
```

## What’s inside
- `src/data/loaders.py` — loads PersonaChat (HF) or pulls Topical-Chat JSON from GitHub.
- `src/data/segmenter.py` — SeCOM-style fixed-window segmentation (2–8 turns).
- `src/tagging/` — emotion classifier + color map + optional tone heuristics.
- `src/embed/` — baseline vs syn-SECOM embeddings, FAISS index builders.
- `src/retrieval/` — query generator + eval (P@1, Recall@K, MRR, Synesthetic Compression Gain).
- `run.py` — end-to-end pipeline entry point.
- `configs/` — example config YAMLs you can tweak.

## Notes
- This skeleton favors **clarity > cleverness**. It’s easy to swap models/components.
- For paper-grade runs, increase `subset_size`, run multiple datasets, and add stats (bootstrap CIs).
