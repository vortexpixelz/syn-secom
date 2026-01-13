from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


EMOTION_LEXICON = {
    "happy": "joy",
    "excited": "joy",
    "love": "joy",
    "sad": "sadness",
    "lonely": "sadness",
    "angry": "anger",
    "frustrated": "anger",
    "afraid": "fear",
    "anxious": "fear",
    "surprised": "surprise",
    "shocked": "surprise",
}

EMOTION_TO_COLOR = {
    "joy": "gold",
    "sadness": "indigo",
    "anger": "crimson",
    "fear": "violet",
    "surprise": "teal",
    "neutral": "silver",
}


@dataclass
class Tagger:
    def tag_segment(self, text: str) -> Dict[str, str]:
        lower = text.lower()
        matched = [emotion for token, emotion in EMOTION_LEXICON.items() if token in lower]
        emotion = matched[0] if matched else "neutral"
        tone = self._tone_from_emotion(emotion)
        color = EMOTION_TO_COLOR[emotion]
        return {"emotion": emotion, "color": color, "tone": tone}

    def _tone_from_emotion(self, emotion: str) -> str:
        if emotion in {"joy", "surprise"}:
            return "bright"
        if emotion in {"anger", "fear"}:
            return "tense"
        if emotion == "sadness":
            return "soft"
        return "even"

    def as_string(self, tags: Dict[str, str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in tags.items())
