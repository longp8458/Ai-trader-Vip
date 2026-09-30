from __future__ import annotations

import copy
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Any, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import ASSETS, TIMEFRAMES
from backend.data_adapter import (
    fetch_market_dataframe_cached,
    fetch_market_klines,
    get_latest_quote,
    get_latest_quotes_batch,
    get_symbol_info,
)
from backend.analyzer import analyze_market
from backend.history import (
    HISTORY_LIMIT,
    clear_history,
    get_history,
    get_stats,
    record_signal,
    refresh_history,
)
from backend.v7_engine import enrich, ENGINE_VERSION
from backend.news import fetch_news


# ============================================================
# APP
# ============================================================

APP_NAME = "TraderAI V7.2.8 Smart Cache Engine"
VERSION = "7.2.8"
MODE = "analysis_only"

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

class Candle(BaseModel):
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class AnalyzeRequest(BaseModel):
    candles: List[Candle] = Field(min_length=220)


# ============================================================
# SPEED / CACHE CONFIG
# ============================================================

MARKET_CACHE_SECONDS = 15.0

# Aggregate MTF cache.
MTF_CACHE_SECONDS = 60.0

SNAPSHOT_CACHE_SECONDS = 15.0

SNAPSHOT_REQUEST_TIMEOUT = 4.0

# Maximum time allowed for a complete MTF request.
MTF_REQUEST_TIMEOUT = 12.0


# ============================================================
# PER-TIMEFRAME CACHE
# ============================================================

FRAME_CACHE_SECONDS = {
    "M1": 15.0,
    "M5": 20.0,
    "M15": 30.0,
    "H1": 60.0,
    "H4": 120.0,
    "D1": 300.0,
    "W1": 600.0,
}


# Maximum age at which an old result may be considered stale.
FRAME_STALE_SECONDS = {
    "M1": 120.0,
    "M5": 180.0,
    "M15": 300.0,
    "H1": 600.0,
    "H4": 1200.0,
    "D1": 1800.0,
    "W1": 3600.0,
}


# ============================================================
# MEMORY CACHE
# ============================================================

_market_cache = {}

_mtf_cache = {}

_snapshot_cache = {}

_timeframe_analysis_cache = {}

_cache_lock = threading.RLock()


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(v):

    if v is None:
        return None

    if isinstance(v, dict):
        return {
            str(k): json_safe(x)
            for k, x in v.items()
        }

    if isinstance(v, (list, tuple)):
        return [
            json_safe(x)
            for x in v
        ]

    if isinstance(v, np.ndarray):
        return [
            json_safe(x)
            for x in v.tolist()
        ]

    if isinstance(v, np.generic):
        v = v.item()

    if isinstance(v, float):
        return (
            v
            if math.isfinite(v)
            else None
        )

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    return v


# ============================================================
# VALIDATE DATAFRAME
# ============================================================

def validate_dataframe(candles):

    df = pd.DataFrame(
        [
            c.model_dump()
            for c in candles
        ]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = df.reset_index(
        drop=True
    )

    if len(df) < 220:
        raise HTTPException(
            422,
            "CÃ¡ÂºÂ§n ÃƒÂ­t nhÃ¡ÂºÂ¥t 220 candle hÃ¡Â»Â£p lÃ¡Â»â€¡.",
        )

    return df


# ============================================================
# MARKET KLINE CACHE
# ============================================================

def fetch_market_klines_cached(
    symbol,
    timeframe="M1",
    limit=300,
):

    key = (
        symbol.upper().strip(),
        timeframe.upper().strip(),
        int(limit),
    )

    now = time.time()

    with _cache_lock:

        cached = _market_cache.get(key)

        if cached:

            cached_time, cached_data = cached

            if (
                now - cached_time
                <= MARKET_CACHE_SECONDS
            ):
                return copy.deepcopy(
                    cached_data
                )

    data = fetch_market_klines(
        symbol,
        timeframe,
        limit,
    )

    with _cache_lock:

        _market_cache[key] = (
            time.time(),
            copy.deepcopy(data),
        )

    return data


# ============================================================
# SINGLE TIMEFRAME ANALYSIS
# ============================================================

def analyze_one(
    symbol,
    timeframe,
    limit,
    with_news=False,
):

    symbol = symbol.upper().strip()
    timeframe = timeframe.upper().strip()

    key = (
        symbol,
        timeframe,
        int(limit),
    )

    now = time.time()

    ttl = FRAME_CACHE_SECONDS.get(
        timeframe,
        30.0,
    )

    # --------------------------------------------------------
    # CHECK TIMEFRAME ANALYSIS CACHE
    # --------------------------------------------------------

    with _cache_lock:

        cached = _timeframe_analysis_cache.get(
            key
        )

        if cached:

            cached_at, cached_result = cached

            age = max(
                0.0,
                now - cached_at,
            )

            # Fresh cache
            if age <= ttl:

                result = copy.deepcopy(
                    cached_result
                )

                result["_cache_hit"] = True
                result["_cache_state"] = "fresh"
                result["_cache_age_seconds"] = round(
                    age,
                    2,
                )
                result["_cache_ttl_seconds"] = ttl

                return result

            # Existing but expired
            cache_state = "stale_refresh"

        else:

            cache_state = "cold"


    # --------------------------------------------------------
    # FETCH DATA
    # --------------------------------------------------------

    df = fetch_market_dataframe_cached(
        symbol,
        timeframe,
        limit,
    )


    # --------------------------------------------------------
    # BASE TECHNICAL ANALYSIS
    # --------------------------------------------------------

    base = analyze_market(df)


    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    if with_news:

        try:

            news = fetch_news(
                symbol,
                8,
            )

        except Exception:

            news = {
                "items": [],
                "sentiment_score": 0.0,
            }

    else:

        news = {
            "items": [],
            "sentiment_score": 0.0,
        }


    # --------------------------------------------------------
    # HIGH-CONVICTION ENGINE
    # --------------------------------------------------------

    analysis = enrich(
        df,
        base,
        news,
    )

    analysis["timeframe"] = timeframe


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------
    #
    # Expose the complete analysis at the timeframe root.
    # This keeps the MTF API schema compatible with the frontend.
    #
    # =====================================================
    # NOVATRADE MTF ANALYSIS SCHEMA
    # Expose real analysis data at timeframe root.
    # Binance / XAUUSD / other symbols use same schema.
    # =====================================================
    _indicators = analysis.get(
        "indicators",
        {}
    ) or {}
    _structure = analysis.get(
        "structure",
        {}
    ) or {}
    _candlestick = analysis.get(
        "candlestick",
        {}
    ) or {}
    _levels = analysis.get(
        "levels",
        {}
    ) or {}
    _trade_levels = analysis.get(
        "trade_levels",
        {}
    ) or {}
    result = {
        # -----------------------------
        # Core
        # -----------------------------
        "signal": analysis.get(
            "signal",
            "NEUTRAL",
        ),
        "confidence": analysis.get(
            "confidence",
            0,
        ),
        "price": analysis.get(
            "price",
        ),
        # -----------------------------
        # Trade analysis
        # -----------------------------
        "entry": analysis.get(
            "entry",
            _trade_levels.get("entry"),
        ),
        "sl": analysis.get(
            "sl",
            _trade_levels.get("sl"),
        ),
        "tp1": analysis.get(
            "tp1",
            _trade_levels.get("tp1"),
        ),
        "tp2": analysis.get(
            "tp2",
            _trade_levels.get("tp2"),
        ),
        "tp3": analysis.get(
            "tp3",
            _trade_levels.get("tp3"),
        ),
        "rr": analysis.get(
            "rr",
        ),
        # -----------------------------
        # Levels
        # -----------------------------
        "trade_levels": _trade_levels,
        "levels": _levels,
        # -----------------------------
        # Technical analysis
        # -----------------------------
        "indicators": _indicators,
        "structure": _structure,
        "candlestick": _candlestick,
        # -----------------------------
        # Reasoning
        # -----------------------------
        "reasoning": analysis.get(
            "reasoning",
            [],
        ),
        # -----------------------------
        # Full original analysis
        # -----------------------------
        "analysis": analysis,
        # -----------------------------
        # Metadata
        # -----------------------------
        "source": get_symbol_info(
            symbol
        ),
        "timeframe": timeframe,
    }


    # --------------------------------------------------------
    # SAVE TIMEFRAME CACHE
    # --------------------------------------------------------

    with _cache_lock:

        _timeframe_analysis_cache[key] = (
            time.time(),
            copy.deepcopy(result),
        )


    result["_cache_hit"] = False

    result["_cache_state"] = cache_state

    result["_cache_age_seconds"] = None

    result["_cache_ttl_seconds"] = ttl

    return result


# ============================================================
# MTF CONSENSUS
# ============================================================

def build_consensus(results):

    weights = {
        "M1": 0.05,
        "M5": 0.08,
        "M15": 0.12,
        "H1": 0.22,
        "H4": 0.18,
        "D1": 0.25,
        "W1": 0.10,
    }

    buy = 0.0

    sell = 0.0


    # --------------------------------------------------------
    # WEIGHTED SCORE
    # --------------------------------------------------------

    for timeframe, result in results.items():

        confidence = float(
            result.get(
                "confidence",
                0,
            )
        )

        score = (
            weights.get(
                timeframe,
                0,
            )
            * confidence
            / 100
        )

        signal = result.get(
            "signal"
        )

        if signal == "BUY":

            buy += score

        elif signal == "SELL":

            sell += score


    total = max(
        buy + sell,
        1e-9,
    )

    edge = abs(
        buy - sell
    ) / total


    if buy > sell:

        signal = "BUY"

    elif sell > buy:

        signal = "SELL"

    else:

        signal = "NEUTRAL"


    # --------------------------------------------------------
    # STRONG DIRECTIONAL FRAMES
    # --------------------------------------------------------

    directional = []

    for result in results.values():

        signal_value = result.get(
            "signal"
        )

        confidence = float(
            result.get(
                "confidence",
                0,
            )
        )

        if (
            signal_value in {
                "BUY",
                "SELL",
            }
            and confidence >= 62
        ):

            directional.append(
                signal_value
            )


    same_direction = sum(
        x == signal
        for x in directional
    )


    # --------------------------------------------------------
    # HIGHER TIMEFRAME SUPPORT
    # --------------------------------------------------------

    higher = [
        results.get(
            timeframe,
            {},
        ).get(
            "signal"
        )
        for timeframe in (
            "H4",
            "D1",
            "W1",
        )
    ]

    higher_directional = [
        x
        for x in higher
        if x in {
            "BUY",
            "SELL",
        }
    ]


    if higher_directional:

        htf_support = (
            sum(
                x == signal
                for x in higher_directional
            )
            / len(
                higher_directional
            )
        )

    else:

        htf_support = 0.0


    # --------------------------------------------------------
    # HIGH CONVICTION
    # --------------------------------------------------------

    high_conviction = (
        signal != "NEUTRAL"
        and same_direction >= 2
        and edge >= 0.20
        and htf_support >= 0.50
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = min(
        97,
        50
        + edge * 45
        + same_direction * 2,
    )


    # --------------------------------------------------------
    # CONSERVATIVE FILTER
    # --------------------------------------------------------

    if not high_conviction:

        signal = "NEUTRAL"


    return {
        "consensus": signal,

        "signal": signal,

        "confidence": round(
            confidence,
            2,
        ),

        "buy_score": round(
            buy,
            4,
        ),

        "sell_score": round(
            sell,
            4,
        ),

        "directional_edge": round(
            edge,
            4,
        ),

        "high_conviction": (
            high_conviction
        ),

        "directional_frames": directional,

        "same_direction_frames": (
            same_direction
        ),

        "htf_support": round(
            htf_support,
            3,
        ),

        "engine_version": ENGINE_VERSION,
    }


# ============================================================
# HISTORY PAYLOAD
# ============================================================

def history_payload(
    symbol,
    consensus,
    results,
):

    signal = consensus.get(
        "signal",
        "NEUTRAL",
    )

    if signal not in {
        "BUY",
        "SELL",
    }:

        return None


    # Higher timeframe preferred
    preferred = (
        "H1",
        "M15",
        "H4",
        "D1",
        "W1",
        "M5",
        "M1",
    )


    for timeframe in preferred:

        analysis = (
            results
            .get(
                timeframe,
                {},
            )
            .get(
                "analysis"
            )
            or {}
        )

        if (
            analysis.get(
                "signal"
            )
            == signal
        ):

            return {
                "symbol": symbol,

                "timeframe": "MTF",

                "signal": signal,

                "entry": analysis.get(
                    "entry"
                ),

                "sl": analysis.get(
                    "sl"
                ),

                "tp1": analysis.get(
                    "tp1"
                ),

                "confidence": consensus.get(
                    "confidence"
                ),

                "signal_quality": analysis.get(
                    "signal_quality"
                ),
            }


    return None


# ============================================================
# RUN MULTI-TIMEFRAME
# ============================================================

def run_mtf(
    symbol,
    limit,
):

    started = time.perf_counter()

    results = {}

    errors = {}


    # --------------------------------------------------------
    # PARALLEL ANALYSIS
    # --------------------------------------------------------

    pool = ThreadPoolExecutor(
        max_workers=7
    )

    jobs = {
        pool.submit(
            analyze_one,
            symbol,
            timeframe,
            limit,
            timeframe in {
                "H1",
                "H4",
                "D1",
                "W1",
            },
        ): timeframe

        for timeframe in TIMEFRAMES
    }


    # --------------------------------------------------------
    # REQUEST TIMEOUT
    # --------------------------------------------------------

    done, pending = wait(
        list(jobs),
        timeout=MTF_REQUEST_TIMEOUT,
    )


    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    for job in done:

        timeframe = jobs[job]

        try:

            results[timeframe] = (
                job.result()
            )

        except Exception as exc:

            errors[timeframe] = str(
                exc
            )

            results[timeframe] = {
                "signal": "NEUTRAL",
                "confidence": 0,
                "analysis": None,
                "error": str(exc),
                "timeframe": timeframe,
                "_cache_hit": False,
                "_cache_state": "error",
                "_cache_age_seconds": None,
                "_cache_ttl_seconds": None,
            }


    # --------------------------------------------------------
    # TIMEOUT FRAMES
    # --------------------------------------------------------

    for job in pending:

        timeframe = jobs[job]

        errors[timeframe] = (
            f"Timeout after "
            f"{MTF_REQUEST_TIMEOUT:.1f}s"
        )

        results[timeframe] = {
            "signal": "NEUTRAL",
            "confidence": 0,
            "analysis": None,
            "error": errors[timeframe],
            "timeframe": timeframe,
            "_cache_hit": False,
            "_cache_state": "timeout",
            "_cache_age_seconds": None,
            "_cache_ttl_seconds": None,
        }

        job.cancel()


    pool.shutdown(
        wait=False,
        cancel_futures=True,
    )


    # --------------------------------------------------------
    # CONSENSUS
    # --------------------------------------------------------

    consensus = build_consensus(
        results
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_record = None

    try:

        payload = history_payload(
            symbol,
            consensus,
            results,
        )

        if payload:

            history_record = (
                record_signal(
                    payload
                )
            )

    except Exception as exc:

        errors["history"] = str(
            exc
        )


    # --------------------------------------------------------
    # CACHE STATISTICS
    # --------------------------------------------------------

    fresh_frames = [
        timeframe
        for timeframe, result
        in results.items()
        if result.get(
            "_cache_state"
        ) == "fresh"
    ]


    stale_frames = [
        timeframe
        for timeframe, result
        in results.items()
        if result.get(
            "_cache_state"
        ) == "stale_refresh"
    ]


    cold_frames = [
        timeframe
        for timeframe, result
        in results.items()
        if result.get(
            "_cache_state"
        ) == "cold"
    ]


    timeout_frames = [
        timeframe
        for timeframe, result
        in results.items()
        if result.get(
            "_cache_state"
        ) == "timeout"
    ]


    cache_hits = sum(
        1
        for result in results.values()
        if result.get(
            "_cache_hit"
        )
    )


    elapsed = round(
        time.perf_counter()
        - started,
        3,
    )


    # --------------------------------------------------------
    # CLEAN INTERNAL CACHE FIELDS
    # --------------------------------------------------------

    for result in results.values():

        result["cache_hit"] = bool(
            result.pop(
                "_cache_hit",
                False,
            )
        )

        result["cache_state"] = (
            result.pop(
                "_cache_state",
                "unknown",
            )
        )

        result["cache_age_seconds"] = (
            result.pop(
                "_cache_age_seconds",
                None,
            )
        )

        result["cache_ttl_seconds"] = (
            result.pop(
                "_cache_ttl_seconds",
                None,
            )
        )


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return json_safe({

        "status": "ok",

        "symbol": symbol,

        "timeframes": results,

        "consensus": consensus,
        "overall_signal": consensus.get("signal", "NEUTRAL"),
        "overall_confidence": consensus.get("confidence", 0),

        "errors": errors,

        "history_record": history_record,

        "analysis_only": True,

        "engine_version": ENGINE_VERSION,

        "source": get_symbol_info(
            symbol
        ),

        "speed_upgrade":
            "V7.2.8-SMART-CACHE",

        "cached_seconds":
            MTF_CACHE_SECONDS,

        "analysis_elapsed_seconds":
            elapsed,

        "partial":
            len(errors) > 0,

        "cache_summary": {

            "fresh_frames":
                fresh_frames,

            "stale_frames":
                stale_frames,

            "cold_frames":
                cold_frames,

            "timeout_frames":
                timeout_frames,

            "fresh_count":
                len(fresh_frames),

            "stale_count":
                len(stale_frames),

            "cold_count":
                len(cold_frames),

            "timeout_count":
                len(timeout_frames),

            "cache_hit_count":
                cache_hits,

            "cache_hit_ratio":
                round(
                    cache_hits
                    / max(
                        len(results),
                        1,
                    ),
                    3,
                ),

            "frame_cache_seconds":
                FRAME_CACHE_SECONDS,

            "frame_stale_seconds":
                FRAME_STALE_SECONDS,
        },
    })


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": APP_NAME,
        "version": VERSION,
        "status": "running",
        "mode": MODE,
        "analysis_only": True,
        "execution": False,
        "order_placement": False,
        "engine_version": ENGINE_VERSION,
        "speed_upgrade":
            "V7.2.8-SMART-CACHE",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "name": APP_NAME,

        "version": VERSION,

        "status": "running",

        "mode": MODE,

        "analysis_only": True,

        "execution": False,

        "order_placement": False,

        "engine_version": ENGINE_VERSION,

        "timeframes": TIMEFRAMES,

        "speed_upgrade":
            "V7.2.8-SMART-CACHE",

        "cache": {

            "market_seconds":
                MARKET_CACHE_SECONDS,

            "mtf_seconds":
                MTF_CACHE_SECONDS,

            "snapshot_seconds":
                SNAPSHOT_CACHE_SECONDS,

            "snapshot_timeout_seconds":
                SNAPSHOT_REQUEST_TIMEOUT,

            "mtf_timeout_seconds":
                MTF_REQUEST_TIMEOUT,

            "timeframe_cache_seconds":
                FRAME_CACHE_SECONDS,

            "timeframe_stale_seconds":
                FRAME_STALE_SECONDS,
        },
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    return {

        "name": APP_NAME,

        "version": VERSION,

        "status": "running",

        "mode": MODE,

        "analysis_only": True,

        "execution": False,

        "order_placement": False,

        "engine_version": ENGINE_VERSION,

        "assets": ASSETS,

        "speed_upgrade":
            "V7.2.8-SMART-CACHE",

        "cache": {

            "market_seconds":
                MARKET_CACHE_SECONDS,

            "mtf_seconds":
                MTF_CACHE_SECONDS,

            "snapshot_seconds":
                SNAPSHOT_CACHE_SECONDS,

            "snapshot_timeout_seconds":
                SNAPSHOT_REQUEST_TIMEOUT,

            "mtf_timeout_seconds":
                MTF_REQUEST_TIMEOUT,

            "timeframe_cache_seconds":
                FRAME_CACHE_SECONDS,

            "timeframe_stale_seconds":
                FRAME_STALE_SECONDS,
        },
    }


# ============================================================
# MARKET DATA
# ============================================================

@app.get("/api/market-data")
def market_data(
    symbol: str,
    timeframe: str = "M1",
    limit: int = Query(
        300,
        ge=1,
        le=1000,
    ),
):

    try:

        data = fetch_market_klines_cached(
            symbol,
            timeframe,
            limit,
        )

        return json_safe({

            "status": "ok",

            "symbol":
                symbol.upper(),

            "timeframe":
                timeframe.upper(),

            "data": data,

            "source":
                get_symbol_info(
                    symbol
                ),

            "cached_seconds":
                MARKET_CACHE_SECONDS,

            "speed_upgrade":
                "V7.2.8-SMART-CACHE",
        })

    except Exception as exc:

        raise HTTPException(
            500,
            f"Market data thÃ¡ÂºÂ¥t bÃ¡ÂºÂ¡i: {exc}",
        )


# ============================================================
# MARKET SNAPSHOT
# ============================================================

@app.get("/api/market-snapshot")
def market_snapshot(
    symbols: str,
):
    requested = list(
        dict.fromkeys(
            [
                x.strip().upper()
                for x in symbols.split(",")
                if x.strip()
            ]
        )
    )[:30]
    if not requested:
        raise HTTPException(
            422,
            "C?n ?t nh?t m?t symbol.",
        )
    key = tuple(
        sorted(requested)
    )
    now = time.time()
    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------
    with _cache_lock:
        cached = _snapshot_cache.get(
            key
        )
        if cached:
            cached_time, cached_data = cached
            if (
                now - cached_time
                <= SNAPSHOT_CACHE_SECONDS
            ):
                output = copy.deepcopy(
                    cached_data
                )
                output["cached"] = True
                output[
                    "cache_age_seconds"
                ] = round(
                    now - cached_time,
                    2,
                )
                return json_safe(
                    output
                )
    # --------------------------------------------------------
    # BULK FETCH
    # --------------------------------------------------------
    try:
        quotes, errors = (
            get_latest_quotes_batch(
                requested
            )
        )
    except Exception as exc:
        quotes = {}
        errors = {
            symbol: str(exc)
            for symbol in requested
        }
    missing = [
        symbol
        for symbol in requested
        if symbol not in quotes
    ]
    result = {
        "status": "ok",
        "data": quotes,
        "items": quotes,
        "errors": errors,
        "cached": False,
        "partial": bool(missing),
        "received": len(quotes),
        "requested": len(requested),
        "missing": missing,
        "updated_at": time.time(),
        "analysis_only": True,
        "speed_upgrade":
            "V7.2.8-SMART-CACHE",
        "request_timeout_seconds":
            SNAPSHOT_REQUEST_TIMEOUT,
    }
    # --------------------------------------------------------
    # IMPORTANT:
    # KH?NG CACHE SNAPSHOT THI?U
    # --------------------------------------------------------
    if quotes and not missing:
        with _cache_lock:
            _snapshot_cache[key] = (
                time.time(),
                copy.deepcopy(
                    result
                ),
            )
    return json_safe(
        result
    )

# ============================================================
# CLEAR CACHE
# ============================================================

@app.post("/api/cache/clear")
def cache_clear():

    with _cache_lock:

        _market_cache.clear()

        _snapshot_cache.clear()

        _mtf_cache.clear()

        _timeframe_analysis_cache.clear()


    return {

        "status": "ok",

        "message":
            "Ã„ÂÃƒÂ£ xÃƒÂ³a toÃƒÂ n bÃ¡Â»â„¢ cache V7.2.7",

        "analysis_only": True,

        "speed_upgrade":
            "V7.2.8-SMART-CACHE",
    }


# ============================================================
# NEWS
# ============================================================

@app.get("/api/news")
def api_news(
    symbol: str,
    limit: int = Query(
        8,
        ge=1,
        le=20,
    ),
):

    try:

        return json_safe({

            "status": "ok",

            **fetch_news(
                symbol,
                limit,
            ),
        })

    except Exception as exc:

        raise HTTPException(
            500,
            f"News thÃ¡ÂºÂ¥t bÃ¡ÂºÂ¡i: {exc}",
        )


# ============================================================
# MANUAL ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze(
    request: AnalyzeRequest,
):

    try:

        df = validate_dataframe(
            request.candles
        )

        base = analyze_market(
            df
        )

        result = enrich(
            df,
            base,
            {
                "items": [],
                "sentiment_score": 0,
            },
        )

        return json_safe({

            "status": "ok",

            "analysis": result,

            "analysis_only": True,

            "engine_version":
                ENGINE_VERSION,

            "speed_upgrade":
                "V7.2.8-SMART-CACHE",
        })

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            500,
            f"PhÃƒÂ¢n tÃƒÂ­ch thÃ¡ÂºÂ¥t bÃ¡ÂºÂ¡i: {exc}",
        )


# ============================================================
# LIVE MTF ANALYSIS
# ============================================================

@app.get("/api/analyze-live-mtf")
def analyze_live_mtf(
    symbol: str,
    limit: int = Query(
        300,
        ge=220,
        le=1000,
    ),
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    key = (
        symbol,
        int(limit),
    )

    now = time.time()


    # --------------------------------------------------------
    # AGGREGATE CACHE
    # --------------------------------------------------------

    with _cache_lock:

        cached = _mtf_cache.get(
            key
        )

        if cached:

            cached_time, cached_result = (
                cached
            )

            age = (
                now - cached_time
            )

            if age <= MTF_CACHE_SECONDS:

                output = copy.deepcopy(
                    cached_result
                )

                output["cached"] = True

                output[
                    "cache_age_seconds"
                ] = round(
                    age,
                    2,
                )

                output[
                    "speed_upgrade"
                ] = (
                    "V7.2.8-SMART-CACHE"
                )

                return json_safe(
                    output
                )


    # --------------------------------------------------------
    # COLD / EXPIRED MTF
    # --------------------------------------------------------

    result = run_mtf(
        symbol,
        int(limit),
    )


    # --------------------------------------------------------
    # SAVE AGGREGATE CACHE
    # --------------------------------------------------------

    with _cache_lock:

        _mtf_cache[key] = (
            time.time(),
            copy.deepcopy(
                result
            ),
        )


    result["cached"] = False

    result[
        "cache_age_seconds"
    ] = 0

    result[
        "speed_upgrade"
    ] = (
        "V7.2.8-SMART-CACHE"
    )


    return json_safe(
        result
    )


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
def api_history(
    limit: int = Query(
        HISTORY_LIMIT,
        ge=1,
        le=HISTORY_LIMIT,
    ),
):

    history = get_history(
        limit
    )

    return json_safe({

        "status": "ok",

        "history": history,

        "stats":
            get_stats(
                history
            ),

        "limit": limit,

        "analysis_only": True,
    })


# ============================================================
# HISTORY REFRESH
# ============================================================

@app.get("/api/history/refresh")
def api_history_refresh(
    limit: int = Query(
        HISTORY_LIMIT,
        ge=1,
        le=HISTORY_LIMIT,
    ),
):

    return json_safe({

        "status": "ok",

        **refresh_history(
            limit
        ),

        "analysis_only": True,
    })


# ============================================================
# HISTORY RECORD
# ============================================================

@app.post("/api/history/record")
def api_history_record(
    payload: dict,
):

    try:

        result = record_signal(
            payload
        )

        return json_safe({

            "status": "ok",

            **result,

            "stats":
                get_stats(),

            "analysis_only": True,
        })

    except Exception as exc:

        raise HTTPException(
            422,
            str(exc),
        )


# ============================================================
# HISTORY DELETE
# ============================================================

@app.delete("/api/history")
def api_history_delete():

    return json_safe({

        "status": "ok",

        **clear_history(),

        "analysis_only": True,
    })
