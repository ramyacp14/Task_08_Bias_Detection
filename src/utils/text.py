from __future__ import annotations
import re
from typing import Dict, List, Tuple
from nltk.sentiment import SentimentIntensityAnalyzer

_VADER = None

def vader() -> SentimentIntensityAnalyzer:
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER

def sentiment_scores(text: str) -> Dict[str, float]:
    return vader().polarity_scores(text)

def extract_player_mentions(text: str, ids: List[str]) -> List[str]:
    pat = r"\b(" + "|".join(re.escape(i) for i in ids) + r")\b"
    return list(dict.fromkeys(re.findall(pat, text)))

def extract_numbers(text: str) -> List[Tuple[str, float]]:
    pairs = []
    for m in re.finditer(r"([A-Za-z_ ]{1,20})\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text):
        pairs.append((m.group(1).strip().lower(), float(m.group(2))))
    return pairs

RECOMMENDATION_KEYS = {
    "defensive": ["defense", "steal", "block", "rebound", "turnover"],
    "offensive": ["offense", "goal", "assist", "shot"],
    "individual": ["individual", "1-on-1", "drill", "personal"],
    "team": ["team", "system", "scheme", "unit"],
}

def recommendation_types(text: str) -> Dict[str, int]:
    t = text.lower()
    out = {k: 0 for k in RECOMMENDATION_KEYS}
    for k, kws in RECOMMENDATION_KEYS.items():
        out[k] = int(any(kw in t for kw in kws))
    return out
