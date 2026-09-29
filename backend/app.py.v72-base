from __future__ import annotations

import math
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


APP_NAME = "TraderAI V7 High-Conviction"
VERSION = "7.0.0"
MODE = "analysis_only"


app = FastAPI(
    title=APP_NAME,
    version=VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def json_safe(value):

    if value is None:
        return None

    if isinstance(value, dict):

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


def analyze_one(
    symbol,
    timeframe,
    limit,
    include_news=False,
):

    df = fetch_market_dataframe(
        symbol,
        timeframe,
        limit,
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
            analysis["signal"],

        "confidence":
            analysis["confidence"],

        "analysis":
            analysis,

        "source":
            get_symbol_info(symbol),

        "timeframe":
            timeframe,
    }


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

        confidence = float(
            result.get(
                "confidence",
                50,
            )
        ) / 100

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

    directional_frames = [
        result.get("signal")
        for result in results.values()
        if result.get("signal")
        in {
            "BUY",
            "SELL",
        }
        and float(
            result.get(
                "confidence",
                0,
            )
        ) >= 62
    ]

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
    }


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
    }


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
    }


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

        "timeframes":
            TIMEFRAMES,
    }


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

        "assets":
            ASSETS,
    }


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

        return json_safe(
            {
                "status":
                    "ok",

                "symbol":
                    symbol.upper(),

                "timeframe":
                    timeframe.upper(),

                "data":
                    fetch_market_klines(
                        symbol,
                        timeframe,
                        limit,
                    ),

                "source":
                    get_symbol_info(
                        symbol
                    ),
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


@app.get("/api/news")
def api_news(
    symbol: str,
    limit: int = Query(
        8,
        ge=1,
        le=20,
    ),
):

    return json_safe(
        {
            "status":
                "ok",

            **fetch_news(
                symbol,
                limit,
            ),
        }
    )


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

    results = {}
    errors = {}

    with ThreadPoolExecutor(
        max_workers=7
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

    consensus = build_v7_consensus(
        results
    )

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

    return json_safe(
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
        }
    )


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