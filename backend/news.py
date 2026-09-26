from __future__ import annotations

from urllib.parse import quote
from urllib.request import Request, urlopen

import html
import re
import time
import xml.etree.ElementTree as ET


CACHE = {}

POSITIVE_WORDS = {
    "surge",
    "rally",
    "gain",
    "gains",
    "growth",
    "approval",
    "approved",
    "inflow",
    "bullish",
    "record",
    "strong",
    "optimism",
    "easing",
    "cut",
    "cuts",
    "recovery",
    "breakout",
    "positive",
    "beat",
    "beats",
}

NEGATIVE_WORDS = {
    "fall",
    "falls",
    "drop",
    "drops",
    "loss",
    "losses",
    "bearish",
    "crash",
    "war",
    "sanction",
    "inflation",
    "hawkish",
    "hike",
    "hikes",
    "outflow",
    "weak",
    "recession",
    "risk",
    "selloff",
    "sell-off",
    "miss",
    "misses",
    "ban",
    "negative",
}


def _clean(value):

    text = html.unescape(
        str(value or "")
    )

    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    return text.strip()


def _sentiment(title):

    text = title.lower()

    positive = sum(
        1
        for word in POSITIVE_WORDS
        if word in text
    )

    negative = sum(
        1
        for word in NEGATIVE_WORDS
        if word in text
    )

    if positive > negative:

        return (
            "POSITIVE",
            min(
                1.0,
                (positive - negative) / 3,
            ),
        )

    if negative > positive:

        return (
            "NEGATIVE",
            min(
                1.0,
                (negative - positive) / 3,
            ),
        )

    return "NEUTRAL", 0.0


def _term(symbol):

    mapping = {
        "BTCUSDT": "Bitcoin crypto",
        "ETHUSDT": "Ethereum crypto",
        "SOLUSDT": "Solana crypto",
        "XAUUSD": "gold price",
        "EURUSD": "EUR USD forex",
        "GBPUSD": "GBP USD forex",
        "USDJPY": "USD JPY forex",
        "US30": "Dow Jones",
        "US500": "S&P 500",
        "USOIL": "WTI crude oil",
        "BTCUSD": "Bitcoin",
        "ETHUSD": "Ethereum",
    }

    return mapping.get(
        symbol,
        symbol,
    )


def fetch_news(
    symbol,
    limit=8,
):

    symbol = str(
        symbol or ""
    ).upper().strip()

    now = time.time()

    cache_key = (
        symbol,
        limit,
    )

    if cache_key in CACHE:

        cached_time, cached_data = CACHE[
            cache_key
        ]

        if now - cached_time < 120:

            return cached_data

    query = quote(
        _term(symbol)
    )

    url = (
        "https://news.google.com/rss/search"
        f"?q={query}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )

    try:

        request = Request(
            url,
            headers={
                "User-Agent":
                    "TraderAI-V7/1.0"
            },
        )

        raw = urlopen(
            request,
            timeout=8,
        ).read()

        root = ET.fromstring(raw)

        items = []

        for item in root.findall(
            ".//item"
        )[:limit]:

            title = _clean(
                item.findtext("title")
            )

            link = (
                item.findtext("link")
                or ""
            )

            published = (
                item.findtext("pubDate")
                or ""
            )

            sentiment, score = _sentiment(
                title
            )

            items.append(
                {
                    "title": title,
                    "link": link,
                    "published": published,
                    "sentiment": sentiment,
                    "score": round(
                        score,
                        3,
                    ),
                    "source":
                        "Google News RSS",
                }
            )

        total_score = 0.0

        for item in items:

            if item["sentiment"] == "POSITIVE":
                total_score += item["score"]

            elif item["sentiment"] == "NEGATIVE":
                total_score -= item["score"]

        sentiment_score = (
            total_score
            / max(1, len(items))
        )

        result = {
            "symbol": symbol,
            "items": items,
            "count": len(items),
            "sentiment_score": round(
                sentiment_score,
                3,
            ),
            "updated_at": now,
        }

    except Exception as exc:

        result = {
            "symbol": symbol,
            "items": [],
            "count": 0,
            "sentiment_score": 0.0,
            "updated_at": now,
            "error": str(exc),
        }

    CACHE[cache_key] = (
        now,
        result,
    )

    return result