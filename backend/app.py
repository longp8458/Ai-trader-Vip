from __future__ import annotations

import math
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from typing import Any, List

import numpy as np
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    Field,
)

from backend.config import (
    ASSETS,
    TIMEFRAMES,
)

from backend.data_adapter import (
    fetch_market_dataframe,
    fetch_market_dataframe_cached,
    fetch_market_klines,
    check_binance_symbol,
    get_symbol_info,
    is_tradingview_symbol,
)

from backend.analyzer import (
    analyze_market,
)

from backend.mtf import (
    calculate_mtf_consensus,
)

from backend.history import (
    HISTORY_LIMIT,
    clear_history,
    get_history,
    get_stats,
    record_signal,
    refresh_history,
)

from backend.v7_engine import (
    enrich,
    ENGINE_VERSION,
)

from backend.news import (
    fetch_news,
)


# ============================================================
# TRADERAI V7.2 SPEED UPGRADE
# ============================================================

APP_NAME = "TraderAI V7.2 Speed Upgrade"
VERSION = "7.2.0"
MODE = "analysis_only"

# Cache thời gian tính bằng giây
MARKET_CACHE_SECONDS = 12
MTF_CACHE_SECONDS = 45
SNAPSHOT_CACHE_SECONDS = 12

# Giới hạn request
MAX_SNAPSHOT_SYMBOLS = 30
MAX_MTF_WORKERS = 7
MAX_SNAPSHOT_WORKERS = 8


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

    candles: List[Candle] = Field(
        min_length=220
    )


# ============================================================
# CACHE
# ============================================================

_cache_lock = threading.RLock()

_market_cache: dict[
    tuple,
    tuple[float, Any],
] = {}

_mtf_cache: dict[
    tuple,
    tuple[float, Any],
] = {}

_snapshot_cache: dict[
    tuple,
    tuple[float, Any],
] = {}


def _cache_get(
    cache: dict,
    key,
    ttl: float,
):

    now = time.monotonic()

    with _cache_lock:

        item = cache.get(key)

        if item is None:
            return None

        created_at, value = item

        if (
            now - created_at
            > ttl
        ):

            cache.pop(
                key,
                None,
            )

            return None

        return value


def _cache_set(
    cache: dict,
    key,
    value,
):

    with _cache_lock:

        cache[key] = (
            time.monotonic(),
            value,
        )


def _cache_clear():

    with _cache_lock:

        _market_cache.clear()
        _mtf_cache.clear()
        _snapshot_cache.clear()


# ============================================================
# JSON SAFETY
# ============================================================

def json_safe(value):

    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):

        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        np.generic,
    ):

        value = value.item()

    if isinstance(
        value,
        float,
    ):

        if math.isfinite(value):
            return value

        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    return value


# ============================================================
# DATAFRAME VALIDATION
# ============================================================

def validate_dataframe(
    candles,
):

    df = pd.DataFrame(
        [
            candle.model_dump()
            for candle in candles
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

    df = (
        df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index(drop=True)
    )

    if len(df) < 220:

        raise HTTPException(
            status_code=422,
            detail=(
                "Cần ít nhất 220 candle "
                "hợp lệ."
            ),
        )

    return df


# ============================================================
# MARKET DATA CACHE
# ============================================================

def fetch_market_klines_cached(
    symbol: str,
    timeframe: str,
    limit: int,
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    timeframe = (
        timeframe
        .upper()
        .strip()
    )

    key = (
        symbol,
        timeframe,
        int(limit),
    )

    cached = _cache_get(
        _market_cache,
        key,
        MARKET_CACHE_SECONDS,
    )

    if cached is not None:
        return cached

    data = fetch_market_klines(
        symbol,
        timeframe,
        limit,
    )

    data = json_safe(data)

    _cache_set(
        _market_cache,
        key,
        data,
    )

    return data


# ============================================================
# SINGLE TIMEFRAME ANALYSIS
# ============================================================

def analyze_one(
    symbol,
    timeframe,
    limit,
    include_news=False,
):

    df = fetch_market_dataframe_cached(
        symbol,
        timeframe,
        limit,
    )

    if df is None or len(df) < 220:

        raise ValueError(
            f"Không đủ dữ liệu cho "
            f"{symbol} {timeframe}"
        )

    base_analysis = analyze_market(
        df
    )

    if include_news:

        news = fetch_news(
            symbol,
            8,
        )

    else:

        news = {
            "items": [],
            "sentiment_score": 0.0,
        }

    analysis = enrich(
        df,
        base_analysis,
        news,
    )

    analysis["timeframe"] = timeframe

    return {
        "signal":
            analysis.get(
                "signal",
                "NEUTRAL",
            ),

        "confidence":
            analysis.get(
                "confidence",
                0,
            ),

        "analysis":
            analysis,

        "source":
            get_symbol_info(symbol),

        "timeframe":
            timeframe,
    }


# ============================================================
# V7 HIGH-CONVICTION CONSENSUS
# ============================================================

def build_v7_consensus(
    results,
):

    weights = {
        "M1": 0.05,
        "M5": 0.08,
        "M15": 0.12,
        "H1": 0.22,
        "H4": 0.18,
        "D1": 0.25,
        "W1": 0.10,
    }

    buy_score = 0.0
    sell_score = 0.0

    for timeframe, result in results.items():

        weight = weights.get(
            timeframe,
            0,
        )

        signal = result.get(
            "signal",
            "NEUTRAL",
        )

        try:

            confidence = float(
                result.get(
                    "confidence",
                    50,
                )
            ) / 100

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        if signal == "BUY":

            buy_score += (
                weight * confidence
            )

        elif signal == "SELL":

            sell_score += (
                weight * confidence
            )

    total = max(
        buy_score + sell_score,
        1e-9,
    )

    edge = (
        abs(
            buy_score - sell_score
        )
        / total
    )

    if buy_score > sell_score:

        signal = "BUY"

    elif sell_score > buy_score:

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    directional_frames = []

    for result in results.values():

        result_signal = result.get(
            "signal"
        )

        try:

            result_confidence = float(
                result.get(
                    "confidence",
                    0,
                )
            )

        except Exception:

            result_confidence = 0.0

        if (
            result_signal
            in {
                "BUY",
                "SELL",
            }
            and result_confidence >= 62
        ):

            directional_frames.append(
                result_signal
            )

    same_direction = sum(
        1
        for value in directional_frames
        if value == signal
    )

    higher_timeframes = [
        results.get(
            timeframe,
            {},
        ).get(
            "signal"
        )
        for timeframe in [
            "H4",
            "D1",
            "W1",
        ]
    ]

    directional_htf = [
        value
        for value in higher_timeframes
        if value in {
            "BUY",
            "SELL",
        }
    ]

    if directional_htf:

        htf_support = (
            sum(
                value == signal
                for value in directional_htf
            )
            / len(directional_htf)
        )

    else:

        htf_support = 0.0

    high_conviction = (
        signal != "NEUTRAL"
        and same_direction >= 2
        and edge >= 0.20
        and htf_support >= 0.50
    )

    confidence = min(
        97,
        50
        + edge * 45
        + same_direction * 2,
    )

    if not high_conviction:

        signal = "NEUTRAL"

    return {
        "consensus":
            signal,

        "signal":
            signal,

        "confidence":
            round(
                confidence,
                2,
            ),

        "buy_score":
            round(
                buy_score,
                4,
            ),

        "sell_score":
            round(
                sell_score,
                4,
            ),

        "directional_edge":
            round(
                edge,
                4,
            ),

        "high_conviction":
            high_conviction,

        "directional_frames":
            directional_frames,

        "same_direction_frames":
            same_direction,

        "htf_support":
            round(
                htf_support,
                3,
            ),

        "engine_version":
            ENGINE_VERSION,

        "speed_version":
            VERSION,
    }


# ============================================================
# HISTORY
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

    priority = [
        "H1",
        "M15",
        "H4",
        "D1",
        "W1",
        "M5",
        "M1",
    ]

    selected = None

    for timeframe in priority:

        item = results.get(
            timeframe,
            {},
        )

        analysis = (
            item.get(
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

            selected = analysis

            break

    selected = selected or {}

    return {
        "symbol":
            symbol,

        "timeframe":
            "MTF",

        "signal":
            signal,

        "entry":
            selected.get(
                "entry"
            ),

        "sl":
            selected.get(
                "sl"
            ),

        "tp1":
            selected.get(
                "tp1"
            ),

        "confidence":
            consensus.get(
                "confidence"
            ),

        "signal_quality":
            selected.get(
                "signal_quality"
            ),

        "analysis_only":
            True,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name":
            APP_NAME,

        "version":
            VERSION,

        "status":
            "running",

        "mode":
            MODE,

        "analysis_only":
            True,

        "engine_version":
            ENGINE_VERSION,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "name":
            APP_NAME,

        "version":
            VERSION,

        "status":
            "running",

        "mode":
            MODE,

        "analysis_only":
            True,

        "engine_version":
            ENGINE_VERSION,

        "speed_upgrade":
            True,

        "cache":
            {
                "market_seconds":
                    MARKET_CACHE_SECONDS,

                "mtf_seconds":
                    MTF_CACHE_SECONDS,

                "snapshot_seconds":
                    SNAPSHOT_CACHE_SECONDS,
            },

        "timeframes":
            TIMEFRAMES,
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    return {
        "name":
            APP_NAME,

        "version":
            VERSION,

        "status":
            "running",

        "mode":
            MODE,

        "analysis_only":
            True,

        "execution":
            False,

        "order_placement":
            False,

        "engine_version":
            ENGINE_VERSION,

        "speed_upgrade":
            True,

        "assets":
            ASSETS,
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

    symbol = (
        symbol
        .upper()
        .strip()
    )

    timeframe = (
        timeframe
        .upper()
        .strip()
    )

    try:

        data = fetch_market_klines_cached(
            symbol,
            timeframe,
            limit,
        )

        return json_safe(
            {
                "status":
                    "ok",

                "symbol":
                    symbol,

                "timeframe":
                    timeframe,

                "data":
                    data,

                "source":
                    get_symbol_info(
                        symbol
                    ),

                "cached":
                    True,

                "cache_seconds":
                    MARKET_CACHE_SECONDS,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Market data thất bại: "
                f"{exc}"
            ),
        )


# ============================================================
# FAST MARKET SNAPSHOT
# ============================================================

@app.get("/api/market-snapshot")
def market_snapshot(
    symbols: str,
):

    requested = [
        item.strip().upper()
        for item in symbols.split(",")
        if item.strip()
    ]

    requested = list(
        dict.fromkeys(
            requested
        )
    )[:MAX_SNAPSHOT_SYMBOLS]

    if not requested:

        raise HTTPException(
            status_code=422,
            detail=(
                "Cần ít nhất một symbol."
            ),
        )

    cache_key = tuple(
        sorted(requested)
    )

    cached = _cache_get(
        _snapshot_cache,
        cache_key,
        SNAPSHOT_CACHE_SECONDS,
    )

    if cached is not None:

        return json_safe(
            {
                **cached,
                "cached":
                    True,
            }
        )

    results = {}

    def get_quote(
        symbol,
    ):

        try:

            df = fetch_market_dataframe_cached(
                symbol,
                "M1",
                2,
            )

            if (
                df is None
                or df.empty
            ):

                return {
                    "symbol":
                        symbol,

                    "status":
                        "error",

                    "error":
                        "Không có dữ liệu",
                }

            row = df.iloc[-1]

            close = float(
                row.get(
                    "close",
                    np.nan,
                )
            )

            previous_close = None

            if len(df) >= 2:

                previous_close = float(
                    df.iloc[-2].get(
                        "close",
                        np.nan,
                    )
                )

            change = None
            change_percent = None

            if (
                previous_close
                and math.isfinite(
                    previous_close
                )
                and previous_close != 0
                and math.isfinite(close)
            ):

                change = (
                    close
                    - previous_close
                )

                change_percent = (
                    change
                    / previous_close
                    * 100
                )

            return {
                "symbol":
                    symbol,

                "status":
                    "ok",

                "price":
                    close,

                "previous_price":
                    previous_close,

                "change":
                    change,

                "change_percent":
                    change_percent,

                "source":
                    get_symbol_info(
                        symbol
                    ),
            }

        except Exception as exc:

            return {
                "symbol":
                    symbol,

                "status":
                    "error",

                "error":
                    str(exc),
            }

    with ThreadPoolExecutor(
        max_workers=min(
            MAX_SNAPSHOT_WORKERS,
            len(requested),
        )
    ) as pool:

        jobs = {
            pool.submit(
                get_quote,
                symbol,
            ):
                symbol

            for symbol in requested
        }

        for job in as_completed(
            jobs
        ):

            symbol = jobs[
                job
            ]

            try:

                results[
                    symbol
                ] = job.result()

            except Exception as exc:

                results[
                    symbol
                ] = {
                    "symbol":
                        symbol,

                    "status":
                        "error",

                    "error":
                        str(exc),
                }

    payload = {
        "status":
            "ok",

        "symbols":
            requested,

        "data":
            results,

        "analysis_only":
            True,

        "version":
            VERSION,
    }

    _cache_set(
        _snapshot_cache,
        cache_key,
        payload,
    )

    return json_safe(
        {
            **payload,
            "cached":
                False,
        }
    )


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

        result = fetch_news(
            symbol.upper().strip(),
            limit,
        )

        return json_safe(
            {
                "status":
                    "ok",

                **result,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "News thất bại: "
                f"{exc}"
            ),
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

        return json_safe(
            {
                "status":
                    "ok",

                "analysis":
                    result,

                "analysis_only":
                    True,

                "engine_version":
                    ENGINE_VERSION,
            }
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Phân tích thất bại: "
                f"{exc}"
            ),
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

    cache_key = (
        symbol,
        int(limit),
    )

    cached = _cache_get(
        _mtf_cache,
        cache_key,
        MTF_CACHE_SECONDS,
    )

    if cached is not None:

        return json_safe(
            {
                **cached,
                "cached":
                    True,
            }
        )

    results = {}
    errors = {}

    # --------------------------------------------------------
    # Chạy 7 timeframe song song
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_MTF_WORKERS
    ) as pool:

        jobs = {
            pool.submit(
                analyze_one,
                symbol,
                timeframe,
                limit,
                timeframe
                in {
                    "H1",
                    "H4",
                    "D1",
                    "W1",
                },
            ):
                timeframe

            for timeframe
            in TIMEFRAMES
        }

        for job in as_completed(
            jobs
        ):

            timeframe = jobs[
                job
            ]

            try:

                results[
                    timeframe
                ] = job.result()

            except Exception as exc:

                errors[
                    timeframe
                ] = str(exc)

                results[
                    timeframe
                ] = {
                    "signal":
                        "NEUTRAL",

                    "confidence":
                        0,

                    "analysis":
                        None,

                    "error":
                        str(exc),

                    "timeframe":
                        timeframe,
                }

    # --------------------------------------------------------
    # Consensus
    # --------------------------------------------------------

    consensus = build_v7_consensus(
        results
    )

    # --------------------------------------------------------
    # History
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

        errors[
            "history"
        ] = str(exc)

    response = json_safe(
        {
            "status":
                "ok",

            "symbol":
                symbol,

            "timeframes":
                results,

            "consensus":
                consensus,

            "errors":
                errors,

            "history_record":
                history_record,

            "analysis_only":
                True,

            "engine_version":
                ENGINE_VERSION,

            "source":
                get_symbol_info(
                    symbol
                ),

            "speed_version":
                VERSION,
        }
    )

    _cache_set(
        _mtf_cache,
        cache_key,
        response,
    )

    return {
        **response,
        "cached":
            False,
    }


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

    return json_safe(
        {
            "status":
                "ok",

            "history":
                history,

            "stats":
                get_stats(
                    history
                ),

            "limit":
                limit,

            "analysis_only":
                True,
        }
    )


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

    result = refresh_history(
        limit
    )

    return json_safe(
        {
            "status":
                "ok",

            **result,

            "analysis_only":
                True,
        }
    )


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

        return json_safe(
            {
                "status":
                    "ok",

                **result,

                "stats":
                    get_stats(),

                "analysis_only":
                    True,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.delete("/api/history")
def api_history_delete():

    return json_safe(
        {
            "status":
                "ok",

            **clear_history(),

            "analysis_only":
                True,
        }
    )


# ============================================================
# CACHE CONTROL
# ============================================================

@app.post("/api/cache/clear")
def api_cache_clear():

    _cache_clear()

    return {
        "status":
            "ok",

        "message":
            "Đã xóa toàn bộ cache.",

        "analysis_only":
            True,

        "version":
            VERSION,
    }