"""Financial sentiment analysis.

Three tiers, selected automatically at runtime:

1. **FinBERT / transformer** - if ``transformers`` + weights are available
2. **VADER-style lexicon** tuned with a finance-specific word list
3. **Heuristic fallback** - always available, deterministic

The public API is stable regardless of the active backend, so the rest of the
platform never needs to care which one answered.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

LABELS = ("positive", "negative", "neutral")

# --------------------------------------------------------------- lexicons
POSITIVE_TERMS = {
    "beat": 2.0, "beats": 2.0, "surge": 2.5, "surges": 2.5, "soar": 2.5, "soars": 2.5,
    "rally": 2.0, "rallies": 2.0, "gain": 1.5, "gains": 1.5, "profit": 1.8, "profits": 1.8,
    "growth": 1.5, "record": 1.8, "upgrade": 2.2, "upgraded": 2.2, "outperform": 2.2,
    "bullish": 2.5, "strong": 1.5, "stronger": 1.7, "boost": 1.8, "boosted": 1.8,
    "exceed": 2.0, "exceeds": 2.0, "exceeded": 2.0, "optimistic": 1.8, "rebound": 2.0,
    "expansion": 1.4, "breakthrough": 2.2, "dividend": 1.2, "buyback": 1.6, "jump": 1.8,
    "jumps": 1.8, "climb": 1.6, "climbs": 1.6, "rise": 1.4, "rises": 1.4, "higher": 1.2,
    "success": 1.6, "successful": 1.6, "innovative": 1.3, "partnership": 1.2, "expand": 1.3,
    "raised": 1.5, "raises": 1.5, "positive": 1.5, "recovery": 1.7, "momentum": 1.4,
    "top": 1.2, "tops": 1.6, "win": 1.5, "wins": 1.5, "approval": 1.6, "approved": 1.6,
}

NEGATIVE_TERMS = {
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "plunge": -2.8, "plunges": -2.8,
    "crash": -3.0, "crashes": -3.0, "slump": -2.3, "slumps": -2.3, "loss": -1.8,
    "losses": -1.8, "decline": -1.6, "declines": -1.6, "downgrade": -2.4, "downgraded": -2.4,
    "underperform": -2.2, "bearish": -2.5, "weak": -1.6, "weaker": -1.8, "warning": -2.0,
    "warns": -2.0, "cut": -1.5, "cuts": -1.5, "layoff": -2.2, "layoffs": -2.2,
    "bankruptcy": -3.0, "fraud": -3.0, "investigation": -2.0, "lawsuit": -1.8,
    "recession": -2.5, "inflation": -1.2, "selloff": -2.4, "sell-off": -2.4, "tumble": -2.4,
    "tumbles": -2.4, "drop": -1.6, "drops": -1.6, "fall": -1.5, "falls": -1.5,
    "lower": -1.2, "concern": -1.4, "concerns": -1.4, "risk": -1.1, "risks": -1.1,
    "volatile": -1.3, "volatility": -1.1, "uncertainty": -1.5, "slowdown": -1.9,
    "deficit": -1.5, "debt": -1.0, "default": -2.6, "halt": -1.8, "halted": -1.8,
    "probe": -1.9, "fine": -1.4, "penalty": -1.7, "resign": -1.6, "resigns": -1.6,
}

NEGATIONS = {"not", "no", "never", "none", "nothing", "neither", "nor", "without", "hardly", "barely"}
INTENSIFIERS = {"very": 1.4, "extremely": 1.7, "highly": 1.4, "significantly": 1.5,
                "sharply": 1.6, "slightly": 0.6, "somewhat": 0.7, "marginally": 0.6}

TOKEN_RE = re.compile(r"[a-z][a-z\-']+")


@dataclass
class SentimentResult:
    label: str
    score: float             # signed polarity in [-1, 1]
    confidence: float        # [0, 1]
    scores: dict             # per-label probabilities
    backend: str
    keywords: list[str]

    def to_dict(self) -> dict:
        return {
            "label": self.label, "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "backend": self.backend, "keywords": self.keywords,
        }


# --------------------------------------------------------------- FinBERT
@lru_cache(maxsize=1)
def _load_finbert():
    """Load FinBERT lazily; returns ``None`` when unavailable (offline / not installed)."""
    if settings.offline:
        return None
    try:
        from transformers import pipeline  # type: ignore

        pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
            device=-1,
        )
        logger.info("FinBERT loaded")
        return pipe
    except Exception as exc:
        logger.info("FinBERT unavailable (%s) - using lexicon backend", type(exc).__name__)
        return None


class SentimentAnalyzer:
    """Finance-tuned sentiment engine with graceful degradation."""

    def __init__(self, prefer_transformer: bool = True) -> None:
        self.prefer_transformer = prefer_transformer

    # ------------------------------------------------------------ lexicon
    def _lexicon_score(self, text: str) -> tuple[float, list[str]]:
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return 0.0, []
        total, hits = 0.0, []
        for i, tok in enumerate(tokens):
            weight = POSITIVE_TERMS.get(tok, 0.0) or NEGATIVE_TERMS.get(tok, 0.0)
            if weight == 0.0:
                continue
            multiplier = 1.0
            window = tokens[max(0, i - 3): i]
            if any(w in NEGATIONS for w in window):
                multiplier *= -0.85
            for w in window:
                if w in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[w]
            total += weight * multiplier
            hits.append(tok)
        # squash: length-normalised then tanh
        norm = total / math.sqrt(max(len(tokens), 1)) * 1.15
        return math.tanh(norm), hits[:8]

    # -------------------------------------------------------------- public
    def analyze(self, text: str) -> SentimentResult:
        text = (text or "").strip()
        if not text:
            return SentimentResult("neutral", 0.0, 0.0, dict.fromkeys(LABELS, 1 / 3), "empty", [])

        if self.prefer_transformer:
            pipe = _load_finbert()
            if pipe is not None:
                try:
                    raw = pipe(text)[0]
                    label = raw["label"].lower()
                    conf = float(raw["score"])
                    signed = conf if label == "positive" else -conf if label == "negative" else 0.0
                    scores = {name: (conf if name == label else (1 - conf) / 2) for name in LABELS}
                    _, hits = self._lexicon_score(text)
                    return SentimentResult(label, signed, conf, scores, "finbert", hits)
                except Exception as exc:  # pragma: no cover
                    logger.debug("FinBERT inference failed: %s", exc)

        score, keywords = self._lexicon_score(text)
        if score > 0.12:
            label = "positive"
        elif score < -0.12:
            label = "negative"
        else:
            label = "neutral"
        confidence = min(abs(score) * 1.6 + 0.25, 0.97)
        pos = max(score, 0.0)
        neg = max(-score, 0.0)
        neutral = max(1.0 - pos - neg, 0.02)
        total = pos + neg + neutral
        scores = {"positive": pos / total, "negative": neg / total, "neutral": neutral / total}
        return SentimentResult(label, score, confidence, scores, "lexicon", keywords)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]

    def aggregate(self, results: list[SentimentResult], weights: list[float] | None = None) -> dict:
        """Weighted aggregation -> a single market-sentiment reading."""
        if not results:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0,
                    "distribution": dict.fromkeys(LABELS, 0), "n": 0}
        w = weights or [1.0] * len(results)
        w = [max(x, 0.0) for x in w]
        tw = sum(w) or 1.0
        score = sum(r.score * wi for r, wi in zip(results, w, strict=False)) / tw
        conf = sum(r.confidence * wi for r, wi in zip(results, w, strict=False)) / tw
        dist = {name: sum(1 for r in results if r.label == name) for name in LABELS}
        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        return {
            "label": label, "score": round(score, 4), "confidence": round(conf, 4),
            "distribution": dist, "n": len(results),
            "bullish_ratio": round(dist["positive"] / len(results), 3),
            "bearish_ratio": round(dist["negative"] / len(results), 3),
        }


sentiment_analyzer = SentimentAnalyzer()
