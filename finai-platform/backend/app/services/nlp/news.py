"""Financial news collection, classification and market-impact scoring.

Sources (tried in order, each optional):
* Yahoo Finance ticker news (via yfinance)
* Finnhub company news
* NewsAPI.org
* deterministic synthetic newsroom (offline fallback)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

import requests

from app.core.config import settings
from app.core.logging import get_logger
from app.services.data import cache
from app.services.data.universe import infer_instrument
from app.services.nlp.sentiment import sentiment_analyzer

logger = get_logger(__name__)

CATEGORIES = [
    "earnings", "mergers_acquisitions", "monetary_policy", "regulation",
    "product_launch", "analyst_rating", "macroeconomic", "geopolitics",
    "crypto", "management", "market_move", "general",
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "revenue", "quarterly", "eps", "profit", "guidance", "results", "q1", "q2", "q3", "q4"],
    "mergers_acquisitions": ["merger", "acquisition", "acquire", "takeover", "buyout", "deal", "stake"],
    "monetary_policy": ["fed", "federal reserve", "ecb", "interest rate", "rate cut", "rate hike", "central bank", "monetary"],
    "regulation": ["regulator", "regulation", "antitrust", "lawsuit", "sec ", "investigation", "fine", "compliance", "probe"],
    "product_launch": ["launch", "unveil", "release", "new product", "announce", "introduces", "debut"],
    "analyst_rating": ["upgrade", "downgrade", "price target", "analyst", "rating", "outperform", "buy rating", "overweight"],
    "macroeconomic": ["inflation", "gdp", "unemployment", "cpi", "ppi", "recession", "economy", "jobs report"],
    "geopolitics": ["war", "sanction", "tariff", "trade war", "election", "conflict", "treaty"],
    "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "token", "defi", "halving", "etf approval"],
    "management": ["ceo", "cfo", "resign", "appoint", "board", "executive", "steps down"],
    "market_move": ["surge", "plunge", "rally", "selloff", "record high", "correction", "tumble", "soar"],
}

IMPACT_WEIGHTS = {
    "earnings": 0.9, "monetary_policy": 0.95, "mergers_acquisitions": 0.85,
    "regulation": 0.7, "analyst_rating": 0.6, "macroeconomic": 0.8,
    "geopolitics": 0.7, "product_launch": 0.5, "crypto": 0.6,
    "management": 0.55, "market_move": 0.75, "general": 0.3,
}


@dataclass
class NewsItem:
    id: str
    symbol: str
    title: str
    summary: str
    source: str
    url: str
    published_at: str
    category: str = "general"
    sentiment: dict = field(default_factory=dict)
    impact_score: float = 0.0
    relevance: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def classify(text: str) -> str:
    lowered = text.lower()
    best, best_hits = "general", 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best, best_hits = category, hits
    return best


# ------------------------------------------------------------ synthetic feed
_HEADLINE_TEMPLATES = [
    ("{name} beats quarterly earnings estimates as revenue climbs {pct}%",
     "{name} reported stronger-than-expected results, with management raising full-year guidance on resilient demand.", "earnings"),
    ("{name} shares slump {pct}% after disappointing guidance",
     "Investors reacted negatively as {name} warned of slowing growth and margin pressure in the coming quarters.", "earnings"),
    ("Analysts upgrade {name} to Outperform, raise price target",
     "A major brokerage lifted its rating on {name}, citing an improving competitive position and attractive valuation.", "analyst_rating"),
    ("{name} downgraded on valuation concerns",
     "Analysts cut their rating on {name}, warning that the recent rally has priced in much of the expected upside.", "analyst_rating"),
    ("Federal Reserve signals caution on rate cuts, weighing on risk assets",
     "Policymakers stressed that inflation remains above target, tempering expectations for near-term easing.", "monetary_policy"),
    ("{name} announces strategic partnership to expand market reach",
     "The agreement is expected to strengthen {name}'s distribution and unlock new revenue streams.", "product_launch"),
    ("Regulators open investigation into {name} business practices",
     "The probe adds uncertainty for {name} and could result in penalties, according to people familiar with the matter.", "regulation"),
    ("{name} unveils new product line, shares rise {pct}%",
     "The launch was received positively by the market, with analysts highlighting a strong innovation pipeline.", "product_launch"),
    ("Inflation data comes in cooler than expected, lifting equities",
     "Softer price pressures revived hopes for policy easing and supported a broad market rally.", "macroeconomic"),
    ("{name} volatility spikes amid broad market selloff",
     "Heightened uncertainty triggered sharp swings, with trading volumes well above the recent average.", "market_move"),
    ("{name} announces share buyback programme",
     "The company will return capital to shareholders, signalling confidence in its cash-flow outlook.", "earnings"),
    ("Crypto markets rally as institutional inflows accelerate",
     "Digital assets extended gains as large investors increased allocations through regulated products.", "crypto"),
]


def _synthetic_news(symbol: str, limit: int = 10) -> list[NewsItem]:
    inst = infer_instrument(symbol)
    seed = int(hashlib.sha256(f"{symbol}{settings.SYNTHETIC_SEED}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    now = datetime.now(UTC)
    items: list[NewsItem] = []
    templates = _HEADLINE_TEMPLATES.copy()
    rng.shuffle(templates)
    outlets = ["Market Wire", "Global Financial Times", "Reuters Digest", "Bloomberg Brief",
               "Investor Daily", "The Capital Report"]
    for i in range(min(limit, len(templates) * 2)):
        title_tpl, summary_tpl, category = templates[i % len(templates)]
        pct = round(rng.uniform(1.2, 9.5), 1)
        title = title_tpl.format(name=inst.name, pct=pct)
        summary = summary_tpl.format(name=inst.name, pct=pct)
        published = now - timedelta(hours=rng.randint(1, 24 * 10), minutes=rng.randint(0, 59))
        items.append(NewsItem(
            id=_hash_id(symbol, title, str(i)),
            symbol=symbol.upper(), title=title, summary=summary,
            source=rng.choice(outlets), url="https://example.com/news/" + _hash_id(title),
            published_at=published.isoformat(), category=category,
            relevance=round(rng.uniform(0.6, 1.0), 2),
        ))
    return sorted(items, key=lambda x: x.published_at, reverse=True)[:limit]


# -------------------------------------------------------------- live feeds
def _yahoo_news(symbol: str, limit: int) -> list[NewsItem]:
    try:
        import yfinance as yf

        raw = yf.Ticker(symbol).news or []
        items = []
        for entry in raw[:limit]:
            content = entry.get("content", entry)
            title = content.get("title") or entry.get("title", "")
            if not title:
                continue
            summary = content.get("summary") or content.get("description") or ""
            provider = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else entry.get("publisher", "Yahoo Finance")
            pub = content.get("pubDate") or entry.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                published = datetime.fromtimestamp(pub, tz=UTC).isoformat()
            else:
                published = str(pub or datetime.now(UTC).isoformat())
            url = (content.get("canonicalUrl") or {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else entry.get("link", "")
            items.append(NewsItem(
                id=_hash_id(symbol, title), symbol=symbol.upper(), title=title,
                summary=summary[:600], source=provider or "Yahoo Finance", url=url,
                published_at=published,
            ))
        return items
    except Exception as exc:
        logger.debug("yahoo news failed for %s: %s", symbol, exc)
        return []


def _finnhub_news(symbol: str, limit: int) -> list[NewsItem]:
    if not settings.FINNHUB_API_KEY:
        return []
    try:
        to = datetime.now(UTC).date()
        frm = to - timedelta(days=14)
        data = requests.get("https://finnhub.io/api/v1/company-news",
                            params={"symbol": symbol, "from": str(frm), "to": str(to),
                                    "token": settings.FINNHUB_API_KEY},
                            timeout=settings.NETWORK_TIMEOUT).json()
        items = []
        for entry in (data or [])[:limit]:
            title = entry.get("headline", "")
            if not title:
                continue
            items.append(NewsItem(
                id=_hash_id(symbol, title), symbol=symbol.upper(), title=title,
                summary=(entry.get("summary") or "")[:600], source=entry.get("source", "Finnhub"),
                url=entry.get("url", ""),
                published_at=datetime.fromtimestamp(entry.get("datetime", 0), tz=UTC).isoformat(),
            ))
        return items
    except Exception as exc:
        logger.debug("finnhub news failed: %s", exc)
        return []


def _newsapi_news(symbol: str, limit: int) -> list[NewsItem]:
    if not settings.NEWSAPI_KEY:
        return []
    try:
        inst = infer_instrument(symbol)
        data = requests.get("https://newsapi.org/v2/everything",
                            params={"q": inst.name, "language": "en", "sortBy": "publishedAt",
                                    "pageSize": limit, "apiKey": settings.NEWSAPI_KEY},
                            timeout=settings.NETWORK_TIMEOUT).json()
        items = []
        for entry in data.get("articles", [])[:limit]:
            title = entry.get("title", "")
            if not title:
                continue
            items.append(NewsItem(
                id=_hash_id(symbol, title), symbol=symbol.upper(), title=title,
                summary=(entry.get("description") or "")[:600],
                source=(entry.get("source") or {}).get("name", "NewsAPI"),
                url=entry.get("url", ""), published_at=entry.get("publishedAt", ""),
            ))
        return items
    except Exception as exc:
        logger.debug("newsapi failed: %s", exc)
        return []


class NewsService:
    """Collect -> classify -> score sentiment -> estimate market impact."""

    def get_news(self, symbol: str, limit: int = 12, analyze: bool = True) -> list[dict]:
        symbol = symbol.upper().strip()
        key = f"news::{symbol}::{limit}::{int(analyze)}"
        cached = cache.cache_get(key)
        if cached:
            return cached

        items: list[NewsItem] = []
        if settings.allow_network:
            for fetcher in (_yahoo_news, _finnhub_news, _newsapi_news):
                items = fetcher(symbol, limit)
                if items:
                    logger.info("news %s via %s (%d items)", symbol, fetcher.__name__, len(items))
                    break
        if not items:
            items = _synthetic_news(symbol, limit)
            logger.info("news %s served by synthetic newsroom", symbol)

        for item in items:
            blob = f"{item.title}. {item.summary}"
            item.category = classify(blob)
            if analyze:
                result = sentiment_analyzer.analyze(blob)
                item.sentiment = result.to_dict()
                age_hours = self._age_hours(item.published_at)
                recency = max(0.15, 1.0 - age_hours / (24 * 14))
                item.impact_score = round(
                    abs(result.score) * IMPACT_WEIGHTS.get(item.category, 0.4)
                    * result.confidence * recency * item.relevance, 4)

        payload = [i.to_dict() for i in items]
        cache.cache_set(key, payload, ttl=900)
        return payload

    @staticmethod
    def _age_hours(published_at: str) -> float:
        try:
            ts = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            return max((datetime.now(UTC) - ts).total_seconds() / 3600.0, 0.0)
        except Exception:
            return 72.0

    def sentiment_summary(self, symbol: str, limit: int = 20) -> dict:
        news = self.get_news(symbol, limit=limit, analyze=True)
        if not news:
            return {"symbol": symbol.upper(), "label": "neutral", "score": 0.0, "n": 0}

        from app.services.nlp.sentiment import SentimentResult

        results, weights = [], []
        for item in news:
            s = item["sentiment"]
            results.append(SentimentResult(s["label"], s["score"], s["confidence"],
                                           s["scores"], s["backend"], s.get("keywords", [])))
            age = self._age_hours(item["published_at"])
            weights.append(max(0.15, 1.0 - age / (24 * 14)) * item.get("relevance", 1.0))

        agg = sentiment_analyzer.aggregate(results, weights)
        by_category: dict[str, list[float]] = {}
        for item in news:
            by_category.setdefault(item["category"], []).append(item["sentiment"]["score"])

        top = sorted(news, key=lambda x: x.get("impact_score", 0), reverse=True)[:5]
        return {
            "symbol": symbol.upper(),
            **agg,
            "by_category": {k: round(sum(v) / len(v), 4) for k, v in by_category.items()},
            "backend": news[0]["sentiment"].get("backend", "lexicon"),
            "top_impact_news": [
                {"title": n["title"], "source": n["source"], "category": n["category"],
                 "sentiment": n["sentiment"]["label"], "impact_score": n["impact_score"],
                 "published_at": n["published_at"], "url": n["url"]}
                for n in top
            ],
        }

    def market_pulse(self, symbols: list[str]) -> dict:
        """Aggregate sentiment across a watchlist -> a market-wide mood reading."""
        per_symbol = {}
        for sym in symbols:
            try:
                per_symbol[sym.upper()] = self.sentiment_summary(sym, limit=8)
            except Exception as exc:
                logger.warning("pulse failed for %s: %s", sym, exc)
        if not per_symbol:
            return {"mood": "neutral", "score": 0.0, "symbols": {}}
        scores = [v["score"] for v in per_symbol.values()]
        avg = sum(scores) / len(scores)
        mood = "risk-on" if avg > 0.15 else "risk-off" if avg < -0.15 else "neutral"
        ranked = sorted(per_symbol.items(), key=lambda kv: kv[1]["score"], reverse=True)
        # Keep the two lists disjoint so the UI never shows a symbol as both.
        # A |score| below 0.01 rounds to "0.00" on screen: labelling that as
        # bullish or bearish reads as a signal when it is really just noise.
        MIN_CONVICTION = 0.01
        n_side = min(3, max(len(ranked) // 2, 1))
        bullish = [kv for kv in ranked[:n_side] if kv[1]["score"] >= MIN_CONVICTION]
        bearish = [kv for kv in ranked[::-1][:n_side] if kv[1]["score"] <= -MIN_CONVICTION]
        bearish = [kv for kv in bearish if kv[0] not in {k for k, _ in bullish}]
        return {
            "mood": mood, "score": round(avg, 4),
            "most_bullish": [{"symbol": k, "score": v["score"]} for k, v in bullish],
            "most_bearish": [{"symbol": k, "score": v["score"]} for k, v in bearish],
            "symbols": {k: {"label": v["label"], "score": v["score"], "n": v.get("n", 0)}
                        for k, v in per_symbol.items()},
        }


news_service = NewsService()
