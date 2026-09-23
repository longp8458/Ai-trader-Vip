# backend/app.py

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import math

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    APP_NAME,
    VERSION,
    MODE,
    TIMEFRAMES,
)

from .analyzer import analyze_market

from .data_adapter import (
    fetch_market_dataframe,
    fetch_market_klines,
    fetch_binance_dataframe,
    check_binance_symbol,
    get_symbol_info,
    is_tradingview_symbol,
)

from .mtf import calculate_mtf_consensus


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description=(
        "NovaTrade AI V6 - "
        "Market analysis only. "
        "No automatic order execution."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONSTANTS
# ============================================================

ANALYSIS_ONLY = True


# ============================================================
# SAFE JSON
# ============================================================

def _json_safe(
    value: Any,
) -> Any:

    if isinstance(value, dict):

        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):

        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):

        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, float):

        if math.isfinite(value):
            return value

        return None

    return value


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

    symbol: str = "BTCUSDT"

    timeframe: str = "M15"

    limit: int = Field(
        default=300,
        ge=220,
        le=1000,
    )

    candles: list[Candle] = Field(
        min_length=220,
    )


class MTFRequest(BaseModel):

    symbol: str = "BTCUSDT"

    timeframes: list[str] = Field(
        default_factory=lambda: list(TIMEFRAMES)
    )

    limit: int = Field(
        default=300,
        ge=220,
        le=1000,
    )


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
        "analysis_only": ANALYSIS_ONLY,
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
        "analysis_only": ANALYSIS_ONLY,
    }


# ============================================================
# SYMBOL CHECK
# ============================================================

@app.get("/api/symbol-check")
def symbol_check(
    symbol: str,
):

    symbol = symbol.upper().strip()

    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="Symbol không được để trống.",
        )

    try:

        if is_tradingview_symbol(
            symbol
        ):

            return {
                "symbol": symbol,
                "available": True,
                "provider": "YAHOO_PROXY",
                "tradingview": True,
                "source": get_symbol_info(
                    symbol
                ),
                "analysis_only": True,
            }

        available = check_binance_symbol(
            symbol
        )

        return {
            "symbol": symbol,
            "available": available,
            "provider": "BINANCE",
            "tradingview": False,
            "source": get_symbol_info(
                symbol
            ),
            "analysis_only": True,
        }

    except Exception as exc:

        return {
            "symbol": symbol,
            "available": False,
            "error": str(exc),
            "analysis_only": True,
        }


# ============================================================
# MARKET DATA
# ============================================================

@app.get("/api/market-data")
def market_data(
    symbol: str,
    timeframe: str = "M15",
    limit: int = Query(
        300,
        ge=10,
        le=1000,
    ),
):

    symbol = symbol.upper().strip()
    timeframe = timeframe.upper().strip()

    try:

        df = fetch_market_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        candles = []

        for _, row in df.iterrows():

            candles.append(
                {
                    "timestamp": row[
                        "timestamp"
                    ].isoformat(),

                    "open": float(
                        row["open"]
                    ),

                    "high": float(
                        row["high"]
                    ),

                    "low": float(
                        row["low"]
                    ),

                    "close": float(
                        row["close"]
                    ),

                    "volume": float(
                        row["volume"]
                    ),
                }
            )

        return _json_safe(
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(candles),
                "candles": candles,
                "source": get_symbol_info(
                    symbol
                ),
                "analysis_only": True,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Không lấy được dữ liệu "
                f"{symbol} {timeframe}: {exc}"
            ),
        )


# ============================================================
# ANALYZE RAW CANDLES
# ============================================================

@app.post("/api/analyze")
def analyze(
    request: AnalyzeRequest,
):

    try:

        import pandas as pd

        rows = [
            candle.model_dump()
            for candle in request.candles
        ]

        df = pd.DataFrame(rows)

        result = analyze_market(
            df
        )

        return _json_safe(
            {
                "status": "ok",
                "symbol": request.symbol.upper(),
                "timeframe": request.timeframe.upper(),
                "analysis": result,
                "analysis_only": True,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# INTERNAL SINGLE-TIMEFRAME ANALYSIS
# ============================================================

def _analyze_one(
    symbol: str,
    timeframe: str,
    limit: int,
) -> dict[str, Any]:

    df = fetch_market_dataframe(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    analysis = analyze_market(
        df
    )

    return {
        "signal": analysis.get(
            "signal",
            "NEUTRAL",
        ),

        "confidence": analysis.get(
            "confidence",
            0.0,
        ),

        "analysis": analysis,

        "source": get_symbol_info(
            symbol
        ),
    }


# ============================================================
# SINGLE-TIMEFRAME LIVE ANALYSIS
# ============================================================

@app.get("/api/analyze-live")
def analyze_live(
    symbol: str,
    timeframe: str = "M15",
    limit: int = Query(
        300,
        ge=220,
        le=1000,
    ),
):

    symbol = symbol.upper().strip()
    timeframe = timeframe.upper().strip()

    try:

        result = _analyze_one(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        return _json_safe(
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                **result,
                "analysis_only": True,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )


# ============================================================
# MULTI-TIMEFRAME LIVE ANALYSIS
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

    symbol = symbol.upper().strip()

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    timeframes = list(
        TIMEFRAMES
    )

    # Không tạo quá nhiều thread
    max_workers = min(
        len(timeframes),
        7,
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:

        jobs = {
            pool.submit(
                _analyze_one,
                symbol,
                timeframe,
                limit,
            ): timeframe
            for timeframe in timeframes
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
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "analysis": None,
                    "source": get_symbol_info(
                        symbol
                    ),
                    "error": str(exc),
                }

    # ========================================================
    # CONSENSUS INPUT
    # ========================================================

    consensus_input = {}

    for timeframe in timeframes:

        item = results.get(
            timeframe,
            {},
        )

        consensus_input[
            timeframe
        ] = {
            "signal": item.get(
                "signal",
                "NEUTRAL",
            ),

            "confidence": item.get(
                "confidence",
                0.0,
            ),
        }

    try:

        consensus = calculate_mtf_consensus(
            consensus_input
        )

    except Exception as exc:

        consensus = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "buy_score": 0.0,
            "sell_score": 0.0,
            "neutral_score": 1.0,
            "details": {},
            "error": str(exc),
        }

    return _json_safe(
        {
            "status": "ok",

            "symbol": symbol,

            "timeframes": results,

            "consensus": consensus,

            "errors": errors,

            "source": get_symbol_info(
                symbol
            ),

            "analysis_only": True,
        }
    )


# ============================================================
# MTF ANALYSIS FROM REQUEST
# ============================================================

@app.post("/api/analyze-mtf")
def analyze_mtf(
    request: MTFRequest,
):

    symbol = request.symbol.upper().strip()

    requested_timeframes = [
        str(tf).upper().strip()
        for tf in request.timeframes
    ]

    valid_timeframes = [
        tf
        for tf in requested_timeframes
        if tf in TIMEFRAMES
    ]

    if not valid_timeframes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Không có timeframe hợp lệ. "
                f"Hỗ trợ: {TIMEFRAMES}"
            ),
        )

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(
        max_workers=min(
            len(valid_timeframes),
            7,
        )
    ) as pool:

        jobs = {
            pool.submit(
                _analyze_one,
                symbol,
                timeframe,
                request.limit,
            ): timeframe
            for timeframe in valid_timeframes
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
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "analysis": None,
                    "error": str(exc),
                }

    consensus_input = {
        timeframe: {
            "signal": results[
                timeframe
            ].get(
                "signal",
                "NEUTRAL",
            ),

            "confidence": results[
                timeframe
            ].get(
                "confidence",
                0.0,
            ),
        }

        for timeframe in valid_timeframes
    }

    try:

        consensus = calculate_mtf_consensus(
            consensus_input
        )

    except Exception as exc:

        consensus = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "error": str(exc),
        }

    return _json_safe(
        {
            "status": "ok",
            "symbol": symbol,
            "timeframes": results,
            "consensus": consensus,
            "errors": errors,
            "analysis_only": True,
        }
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.get("/api/status")
def status():

    return {
        "application": APP_NAME,
        "version": VERSION,
        "status": "online",
        "mode": MODE,
        "analysis_only": True,
        "automatic_order_execution": False,
    }