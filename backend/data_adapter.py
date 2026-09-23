# backend/data_adapter.py

from __future__ import annotations

from typing import Any
import math
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE_URL = "https://data-api.binance.vision"

TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}


# TradingView symbol -> Yahoo proxy
# Lưu ý: đây là nguồn proxy, không đảm bảo giá giống 100% TradingView.
TV_SYMBOL_MAP = {
    "XAUUSD": {
        "tv": "OANDA:XAUUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "GC=F",
        "name": "Gold Futures",
    },
    "EURUSD": {
        "tv": "OANDA:EURUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "EURUSD=X",
        "name": "EUR/USD",
    },
    "GBPUSD": {
        "tv": "OANDA:GBPUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "GBPUSD=X",
        "name": "GBP/USD",
    },
    "US30": {
        "tv": "CAPITALCOM:US30",
        "provider": "YAHOO_PROXY",
        "ticker": "^DJI",
        "name": "Dow Jones Industrial Average",
    },
    "US500": {
        "tv": "CAPITALCOM:US500",
        "provider": "YAHOO_PROXY",
        "ticker": "^GSPC",
        "name": "S&P 500",
    },
    "USDJPY": {
        "tv": "OANDA:USDJPY",
        "provider": "YAHOO_PROXY",
        "ticker": "JPY=X",
        "name": "USD/JPY",
    },
    "BTCUSD": {
        "tv": "COINBASE:BTCUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "BTC-USD",
        "name": "Bitcoin USD",
    },
    "ETHUSD": {
        "tv": "COINBASE:ETHUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "ETH-USD",
        "name": "Ethereum USD",
    },
    "USOIL": {
        "tv": "TVC:USOIL",
        "provider": "YAHOO_PROXY",
        "ticker": "CL=F",
        "name": "WTI Crude Oil Futures",
    },
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise ValueError("Symbol phải là chuỗi.")

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol không được để trống.")

    return symbol


def _clean_timeframe(timeframe: str) -> str:
    if not isinstance(timeframe, str):
        raise ValueError("Timeframe phải là chuỗi.")

    timeframe = timeframe.strip().upper()

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Timeframe không hỗ trợ: {timeframe}. "
            f"Hỗ trợ: {', '.join(TIMEFRAME_MAP.keys())}"
        )

    return timeframe


def _finite(value: Any) -> float | None:
    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


# ============================================================
# OHLCV VALIDATION
# ============================================================

def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

    if df is None:
        raise ValueError("DataFrame không tồn tại.")

    if df.empty:
        raise ValueError("Không có dữ liệu OHLCV.")

    df = df.copy()

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Thiếu cột OHLCV: {column}"
            )

    # Timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    # Numeric columns
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

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Remove infinity
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=numeric_columns
    )

    # Giá phải hợp lệ
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    # High / Low hợp lệ
    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    # Sort
    df = df.sort_values(
        "timestamp"
    )

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError(
            "Sau khi làm sạch không còn candle OHLCV hợp lệ."
        )

    return df


# ============================================================
# BINANCE
# ============================================================

def _binance_to_dataframe(
    data: list[list[Any]],
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for item in data:

        if not isinstance(item, (list, tuple)):
            continue

        if len(item) < 6:
            continue

        try:

            timestamp = pd.to_datetime(
                int(item[0]),
                unit="ms",
                utc=True,
            )

            open_price = float(item[1])
            high_price = float(item[2])
            low_price = float(item[3])
            close_price = float(item[4])
            volume = float(item[5])

            if not all(
                math.isfinite(x)
                for x in [
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                ]
            ):
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

    if not rows:
        raise ValueError(
            "Binance không trả về candle OHLCV hợp lệ."
        )

    df = pd.DataFrame(rows)

    return _validate_ohlcv(df)


def fetch_binance_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> list[list[Any]]:

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    interval = TIMEFRAME_MAP[timeframe]

    limit = max(
        1,
        min(int(limit), 1000),
    )

    url = (
        f"{BINANCE_BASE_URL}/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        url,
        params=params,
        timeout=15,
        headers={
            "User-Agent": "NovaTradeAI/6.1",
        },
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            "Binance trả về dữ liệu không hợp lệ."
        )

    if not data:
        raise ValueError(
            f"Binance không có dữ liệu cho {symbol} {timeframe}."
        )

    return data


def fetch_binance_dataframe(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    data = fetch_binance_klines(
        symbol,
        timeframe,
        limit,
    )

    return _binance_to_dataframe(data)


# ============================================================
# YAHOO PROXY
# ============================================================

def _yahoo_range(
    timeframe: str,
) -> tuple[str, str]:

    timeframe = _clean_timeframe(timeframe)

    if timeframe == "M1":
        return "7d", "1m"

    if timeframe in {
        "M5",
        "M15",
    }:
        return "60d", TIMEFRAME_MAP[timeframe]

    if timeframe == "H1":
        return "730d", "1h"

    if timeframe == "H4":
        return "730d", "1h"

    if timeframe == "D1":
        return "10y", "1d"

    if timeframe == "W1":
        return "max", "1wk"

    return "60d", "1d"


def _resample_h4(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    temp = df.copy()

    temp = temp.set_index(
        "timestamp"
    )

    result = temp.resample(
        "4h",
        label="left",
        closed="left",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    result = result.dropna()

    result = result.reset_index()

    return result


def fetch_yahoo_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    if symbol not in TV_SYMBOL_MAP:
        raise ValueError(
            f"{symbol} không có trong TradingView symbol map."
        )

    ticker = TV_SYMBOL_MAP[symbol]["ticker"]

    period, interval = _yahoo_range(
        timeframe
    )

    url = (
        "https://query1.finance.yahoo.com"
        "/v8/finance/chart/"
        + ticker
    )

    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    response.raise_for_status()

    payload = response.json()

    chart = payload.get(
        "chart",
        {},
    )

    result = chart.get(
        "result"
    )

    if not result:
        raise ValueError(
            f"Yahoo không trả dữ liệu cho {symbol}."
        )

    result = result[0]

    timestamps = result.get(
        "timestamp",
        [],
    )

    indicators = result.get(
        "indicators",
        {},
    )

    quote_list = indicators.get(
        "quote",
        [],
    )

    if not quote_list:
        raise ValueError(
            f"Yahoo thiếu OHLCV cho {symbol}."
        )

    quote = quote_list[0]

    opens = quote.get(
        "open",
        [],
    )

    highs = quote.get(
        "high",
        [],
    )

    lows = quote.get(
        "low",
        [],
    )

    closes = quote.get(
        "close",
        [],
    )

    volumes = quote.get(
        "volume",
        [],
    )

    rows = []

    length = min(
        len(timestamps),
        len(opens),
        len(highs),
        len(lows),
        len(closes),
        len(volumes),
    )

    for i in range(length):

        try:

            timestamp = pd.to_datetime(
                int(timestamps[i]),
                unit="s",
                utc=True,
            )

            open_price = float(
                opens[i]
            )

            high_price = float(
                highs[i]
            )

            low_price = float(
                lows[i]
            )

            close_price = float(
                closes[i]
            )

            volume = (
                float(volumes[i])
                if volumes[i] is not None
                else 0.0
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

    if not rows:
        raise ValueError(
            f"Yahoo không có candle hợp lệ cho {symbol}."
        )

    df = pd.DataFrame(rows)

    df = _validate_ohlcv(df)

    if timeframe == "H4":
        df = _resample_h4(df)

    df = _validate_ohlcv(df)

    # Lấy limit candle cuối
    if len(df) > limit:
        df = df.tail(limit).reset_index(
            drop=True
        )

    return df


# ============================================================
# SYMBOL DETECTION
# ============================================================

def is_tradingview_symbol(
    symbol: str,
) -> bool:

    symbol = _clean_symbol(symbol)

    return symbol in TV_SYMBOL_MAP


# ============================================================
# MARKET DATA
# ============================================================

def fetch_market_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
):
    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    if is_tradingview_symbol(symbol):

        return fetch_yahoo_klines(
            symbol,
            timeframe,
            limit,
        )

    return fetch_binance_klines(
        symbol,
        timeframe,
        limit,
    )


def fetch_market_dataframe(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    if is_tradingview_symbol(symbol):

        df = fetch_yahoo_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

    else:

        # QUAN TRỌNG:
        # Binance trả list OHLCV dạng raw.
        # Không được dùng pd.DataFrame(data) trực tiếp.
        #
        # Phải chuyển qua _binance_to_dataframe()
        # để tạo timestamp/open/high/low/close/volume.

        data = fetch_binance_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        df = _binance_to_dataframe(
            data
        )

    return _validate_ohlcv(df)


# ============================================================
# SYMBOL INFO
# ============================================================

def check_binance_symbol(
    symbol: str,
) -> bool:

    symbol = _clean_symbol(symbol)

    try:

        url = (
            f"{BINANCE_BASE_URL}"
            "/api/v3/exchangeInfo"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
            },
            timeout=10,
            headers={
                "User-Agent": "NovaTradeAI/6.1",
            },
        )

        if response.status_code != 200:
            return False

        data = response.json()

        symbols = data.get(
            "symbols",
            [],
        )

        return bool(symbols)

    except Exception:
        return False


def get_symbol_info(
    symbol: str,
) -> dict[str, Any]:

    symbol = _clean_symbol(symbol)

    if symbol in TV_SYMBOL_MAP:

        item = TV_SYMBOL_MAP[
            symbol
        ]

        return {
            "symbol": symbol,
            "tv_symbol": item["tv"],
            "provider": item["provider"],
            "ticker": item["ticker"],
            "name": item["name"],
            "market_type": "tradingview_proxy",
            "analysis_only": True,
        }

    return {
        "symbol": symbol,
        "tv_symbol": None,
        "provider": "BINANCE",
        "ticker": symbol,
        "name": symbol,
        "market_type": "crypto",
        "analysis_only": True,
    }


# ============================================================
# COMPATIBILITY
# ============================================================

def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    return fetch_market_dataframe(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )


# ============================================================
# SIMPLE CACHE
# ============================================================

_DATA_CACHE: dict[
    tuple[str, str, int],
    tuple[float, pd.DataFrame],
] = {}

CACHE_SECONDS = 2.0


def fetch_market_dataframe_cached(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    key = (
        _clean_symbol(symbol),
        _clean_timeframe(timeframe),
        int(limit),
    )

    now = time.time()

    cached = _DATA_CACHE.get(
        key
    )

    if cached is not None:

        cached_time, cached_df = cached

        if now - cached_time <= CACHE_SECONDS:

            return cached_df.copy()

    df = fetch_market_dataframe(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    _DATA_CACHE[key] = (
        now,
        df.copy(),
    )

    return df.copy()