from __future__ import annotations

import json
from typing import List

import requests
from datasets import load_dataset


def load_personachat(subset_size: int = 1000) -> List[List[str]]:
    dataset = load_dataset("personachat", "self_original", split="train")
    dialogues: List[List[str]] = []
    total_turns = 0
    for row in dataset:
        turns = [turn["text"] for turn in row["utterances"][-1]["history"]]
        if not turns:
            continue
        dialogues.append(turns)
        total_turns += len(turns)
        if total_turns >= subset_size:
            break
    return dialogues


def load_topical_chat(subset_size: int = 1000) -> List[List[str]]:
    url = (
        "https://raw.githubusercontent.com/alexa/Topical-Chat/master/"
        "data/TopicalChat_test.json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    dialogues: List[List[str]] = []
    total_turns = 0
    for conv in payload.values():
        turns = [turn["message"] for turn in conv["content"]]
        if not turns:
            continue
        dialogues.append(turns)
        total_turns += len(turns)
        if total_turns >= subset_size:
            break
    return dialogues
