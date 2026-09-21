from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analyzer import analyze_market
from .mtf import calculate_mtf_consensus

from .data_engine import (
    normalize_candles,
    validate_candles,
    get_data_metadata
)

from .data_adapter import (
    fetch_binance_klines,
    check_binance_symbol
)


# =====================================================
# APP
# =====================================================

app = FastAPI(

    title="TraderAI V6",

    version="6.0.0"

)


# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# =====================================================
# DATA MODELS
# =====================================================

class Candle(BaseModel):

    timestamp: str

    open: float

    high: float

    low: float

    close: float

    volume: float


class AnalyzeRequest(BaseModel):

    symbol: str

    timeframe: str

    candles: list[Candle]


class MTFRequest(BaseModel):

    symbol: str

    timeframes: dict[
        str,
        list[Candle]
    ]


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {

        "app":
            "TraderAI V6",

        "version":
            "6.0.0",

        "mode":
            "analysis_only",

        "status":
            "running",

        "data_sources":
            [
                "Binance Public Market Data"
            ]

    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "ok",

        "app":
            "TraderAI V6",

        "data_engine":
            "enabled",

        "data_adapter":
            "binance"

    }


# =====================================================
# BINANCE SYMBOL CHECK
# =====================================================

@app.get("/api/symbol-check")
def symbol_check(

    symbol: str = "BTCUSDT"

):

    try:

        result = check_binance_symbol(
            symbol
        )

        return {

            "status":
                "ok",

            "result":
                result

        }

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)

        }


# =====================================================
# REAL MARKET DATA
# =====================================================

@app.get("/api/market-data")
def market_data(

    symbol: str = "BTCUSDT",

    timeframe: str = "M15",

    limit: int = 300

):

    try:

        candles = fetch_binance_klines(

            symbol=symbol,

            timeframe=timeframe,

            limit=limit

        )


        return {

            "status":
                "ok",

            "symbol":
                symbol.upper(),

            "timeframe":
                timeframe,

            "source":
                "Binance Public Market Data",

            "count":
                len(candles),

            "candles":
                candles

        }


    except Exception as error:

        return {

            "status":
                "error",

            "symbol":
                symbol.upper(),

            "timeframe":
                timeframe,

            "message":
                str(error)

        }


# =====================================================
# ANALYZE SINGLE TIMEFRAME
# =====================================================

@app.post("/api/analyze")
def analyze(

    request: AnalyzeRequest

):

    candles_data = [

        candle.model_dump()

        for candle in request.candles

    ]


    df = normalize_candles(
        candles_data
    )


    validation = validate_candles(
        df
    )


    metadata = get_data_metadata(

        df,

        request.symbol,

        request.timeframe,

        source="frontend"

    )


    if not validation["valid"]:

        return {

            "symbol":
                request.symbol,

            "timeframe":
                request.timeframe,

            "status":
                "invalid_data",

            "data_validation":
                validation,

            "metadata":
                metadata

        }


    result = analyze_market(
        df
    )


    return {

        "symbol":
            request.symbol,

        "timeframe":
            request.timeframe,

        "status":
            "ok",

        "metadata":
            metadata,

        "data_validation":
            validation,

        "analysis":
            result

    }


# =====================================================
# ANALYZE MULTI TIMEFRAME
# =====================================================

@app.post("/api/analyze-mtf")
def analyze_mtf(

    request: MTFRequest

):

    timeframe_results = {}

    data_metadata = {}

    data_validation = {}


    for (

        timeframe,

        candles

    ) in request.timeframes.items():


        candles_data = [

            candle.model_dump()

            for candle in candles

        ]


        df = normalize_candles(
            candles_data
        )


        validation = validate_candles(
            df
        )


        metadata = get_data_metadata(

            df,

            request.symbol,

            timeframe,

            source="frontend"

        )


        data_validation[
            timeframe
        ] = validation


        data_metadata[
            timeframe
        ] = metadata


        if not validation["valid"]:

            timeframe_results[
                timeframe
            ] = {

                "signal":
                    "NEUTRAL",

                "confidence":
                    0,

                "analysis": {

                    "signal":
                        "NEUTRAL",

                    "confidence":
                        0,

                    "price":
                        None,

                    "reasoning":
                        "Dữ liệu không hợp lệ."

                }

            }

            continue


        analysis = analyze_market(
            df
        )


        timeframe_results[
            timeframe
        ] = {

            "signal":
                analysis.get(
                    "signal",
                    "NEUTRAL"
                ),

            "confidence":
                analysis.get(
                    "confidence",
                    0
                ),

            "analysis":
                analysis

        }


    consensus = calculate_mtf_consensus(

        timeframe_results

    )


    return {

        "symbol":
            request.symbol,

        "status":
            "ok",

        "consensus":
            consensus,

        "timeframes":
            timeframe_results,

        "data_metadata":
            data_metadata,

        "data_validation":
            data_validation

    }


# =====================================================
# REAL-TIME MTF ANALYSIS
# =====================================================

@app.get("/api/analyze-live-mtf")
def analyze_live_mtf(

    symbol: str = "BTCUSDT",

    limit: int = 300

):

    symbol = symbol.upper().strip()


    timeframes = [

        "M1",

        "M5",

        "M15",

        "H1",

        "D1",

        "W1"

    ]


    timeframe_results = {}

    data_metadata = {}

    data_validation = {}


    for timeframe in timeframes:

        try:

            candles = fetch_binance_klines(

                symbol=symbol,

                timeframe=timeframe,

                limit=limit

            )


            df = normalize_candles(
                candles
            )


            validation = validate_candles(
                df
            )


            metadata = get_data_metadata(

                df,

                symbol,

                timeframe,

                source="binance_spot"

            )


            data_validation[
                timeframe
            ] = validation


            data_metadata[
                timeframe
            ] = metadata


            if not validation["valid"]:

                timeframe_results[
                    timeframe
                ] = {

                    "signal":
                        "NEUTRAL",

                    "confidence":
                        0,

                    "analysis": {

                        "signal":
                            "NEUTRAL",

                        "confidence":
                            0,

                        "price":
                            None,

                        "reasoning":
                            "Dữ liệu không hợp lệ."

                    }

                }

                continue


            analysis = analyze_market(
                df
            )


            timeframe_results[
                timeframe
            ] = {

                "signal":
                    analysis.get(
                        "signal",
                        "NEUTRAL"
                    ),

                "confidence":
                    analysis.get(
                        "confidence",
                        0
                    ),

                "analysis":
                    analysis

            }


        except Exception as error:

            timeframe_results[
                timeframe
            ] = {

                "signal":
                    "NEUTRAL",

                "confidence":
                    0,

                "analysis": {

                    "signal":
                        "NEUTRAL",

                    "confidence":
                        0,

                    "price":
                        None,

                    "reasoning":
                        str(error)

                }

            }


    consensus = calculate_mtf_consensus(

        timeframe_results

    )


    return {

        "status":
            "ok",

        "symbol":
            symbol,

        "source":
            "Binance Spot Public Market Data",

        "analysis_mode":
            "analysis_only",

        "consensus":
            consensus,

        "timeframes":
            timeframe_results,

        "data_metadata":
            data_metadata,

        "data_validation":
            data_validation

    }