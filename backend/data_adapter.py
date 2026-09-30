from __future__ import annotations

from typing import Any
import math
import time
import threading
import json

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE_URL = "https://data-api.binance.vision"

YAHOO_BASE_URLS = [
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
]

BINANCE_TIMEOUT = (3, 7)
YAHOO_TIMEOUT = (3, 6)

CACHE_SECONDS = 15.0

TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}


# ============================================================
# TRADINGVIEW / YAHOO PROXY MAP
# ============================================================

TV_SYMBOL_MAP = {
    "XAUUSD": {
        "tv": "OANDA:XAUUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "GC=F",
        "name": "Gold Futures",
        "market_type": "commodity",
    },
    "EURUSD": {
        "tv": "OANDA:EURUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "EURUSD=X",
        "name": "EUR/USD",
        "market_type": "forex",
    },
    "GBPUSD": {
        "tv": "OANDA:GBPUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "GBPUSD=X",
        "name": "GBP/USD",
        "market_type": "forex",
    },
    "US30": {
        "tv": "CAPITALCOM:US30",
        "provider": "YAHOO_PROXY",
        "ticker": "^DJI",
        "name": "Dow Jones Industrial Average",
        "market_type": "index",
    },
    "US500": {
        "tv": "CAPITALCOM:US500",
        "provider": "YAHOO_PROXY",
        "ticker": "^GSPC",
        "name": "S&P 500",
        "market_type": "index",
    },
    "USDJPY": {
        "tv": "OANDA:USDJPY",
        "provider": "YAHOO_PROXY",
        "ticker": "JPY=X",
        "name": "USD/JPY",
        "market_type": "forex",
    },
    "BTCUSD": {
        "tv": "COINBASE:BTCUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "BTC-USD",
        "name": "Bitcoin USD",
        "market_type": "crypto",
    },
    "ETHUSD": {
        "tv": "COINBASE:ETHUSD",
        "provider": "YAHOO_PROXY",
        "ticker": "ETH-USD",
        "name": "Ethereum USD",
        "market_type": "crypto",
    },
    "USOIL": {
        "tv": "TVC:USOIL",
        "provider": "YAHOO_PROXY",
        "ticker": "CL=F",
        "name": "WTI Crude Oil Futures",
        "market_type": "commodity",
    },
}


# ============================================================
# CACHE
# ============================================================

_DATA_CACHE: dict[
    tuple[str, str, int],
    tuple[float, pd.DataFrame],
] = {}

_CACHE_LOCK = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def _clean_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise ValueError("Symbol pháº£i lÃ  chuá»—i.")

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol khÃ´ng Ä‘Æ°á»£c Ä‘á»ƒ trá»‘ng.")

    return symbol


def _clean_timeframe(timeframe: str) -> str:
    if not isinstance(timeframe, str):
        raise ValueError("Timeframe pháº£i lÃ  chuá»—i.")

    timeframe = timeframe.strip().upper()

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Timeframe khÃ´ng há»— trá»£: {timeframe}. "
            f"Há»— trá»£: {', '.join(TIMEFRAME_MAP)}"
        )

    return timeframe


def _safe_float(value: Any) -> float | None:
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
        raise ValueError("DataFrame khÃ´ng tá»“n táº¡i.")

    if df.empty:
        raise ValueError("KhÃ´ng cÃ³ dá»¯ liá»‡u OHLCV.")

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
                f"Thiáº¿u cá»™t OHLCV: {column}"
            )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
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
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df["volume"] = df["volume"].fillna(0.0)

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "KhÃ´ng cÃ²n candle OHLCV há»£p lá»‡."
        )

    return df


# ============================================================
# BINANCE
# ============================================================

def _binance_to_dataframe(
    data: list[list[Any]],
) -> pd.DataFrame:

    rows = []

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

            values = [
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
            ]

            if not all(
                math.isfinite(x)
                for x in values
            ):
                continue

            rows.append({
                "timestamp": timestamp,
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": values[4],
            })

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

    if not rows:
        raise ValueError(
            "Binance khÃ´ng tráº£ vá» OHLCV há»£p lá»‡."
        )

    return _validate_ohlcv(
        pd.DataFrame(rows)
    )


def fetch_binance_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
):

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    interval = TIMEFRAME_MAP[timeframe]

    limit = max(
        1,
        min(int(limit), 1000),
    )

    response = requests.get(
        f"{BINANCE_BASE_URL}/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        timeout=BINANCE_TIMEOUT,
        headers={
            "User-Agent": "NovaTradeAI/6.1",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            "Binance tráº£ dá»¯ liá»‡u khÃ´ng há»£p lá»‡."
        )

    if not data:
        raise ValueError(
            f"KhÃ´ng cÃ³ dá»¯ liá»‡u {symbol} {timeframe}."
        )

    return data


def fetch_binance_dataframe(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    return _binance_to_dataframe(
        fetch_binance_klines(
            symbol,
            timeframe,
            limit,
        )
    )


# ============================================================
# YAHOO CONFIG
# ============================================================

def _yahoo_range(
    timeframe: str,
) -> tuple[str, str]:
    timeframe = _clean_timeframe(timeframe)
    if timeframe == "M1":
        return "2d", "1m"
    if timeframe == "M5":
        return "7d", "5m"
    if timeframe == "M15":
        return "15d", "15m"
    if timeframe == "H1":
        return "45d", "1h"
    if timeframe == "H4":
        return "180d", "1h"
    if timeframe == "D1":
        return "2y", "1d"
    if timeframe == "W1":
        return "10y", "1wk"
    raise ValueError(
        f"Timeframe Yahoo không hỗ trợ: {timeframe}"
    )
# ============================================================
# H4 RESAMPLE
# ============================================================

def _resample_h4(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    temp = df.copy()

    temp = temp.set_index("timestamp")

    result = temp.resample(
        "4h",
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    result = result.reset_index()

    return result


# ============================================================
# YAHOO FETCH
# ============================================================

def _fetch_yahoo_payload(
    ticker: str,
    period: str,
    interval: str,
) -> dict:

    last_error = None

    for base_url in YAHOO_BASE_URLS:

        url = (
            f"{base_url}/v8/finance/chart/"
            f"{ticker}"
        )

        try:

            response = requests.get(
                url,
                params={
                    "range": period,
                    "interval": interval,
                    "includePrePost": "false",
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                },
                timeout=YAHOO_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/154.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "close",
                },
            )

            if response.status_code == 429:
                last_error = RuntimeError(
                    f"Yahoo rate limit 429: {ticker}"
                )
                continue

            response.raise_for_status()

            payload = response.json()

            chart = payload.get(
                "chart",
                {},
            )

            if chart.get("error"):
                raise ValueError(
                    str(chart["error"])
                )

            results = chart.get("result")

            if not results:
                raise ValueError(
                    f"Yahoo khÃ´ng cÃ³ result cho {ticker}."
                )

            return results[0]

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Yahoo khÃ´ng láº¥y Ä‘Æ°á»£c {ticker}: "
        f"{last_error}"
    )


def fetch_yahoo_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)

    if symbol not in TV_SYMBOL_MAP:
        raise ValueError(
            f"{symbol} khÃ´ng cÃ³ trong TV_SYMBOL_MAP."
        )

    ticker = TV_SYMBOL_MAP[symbol]["ticker"]

    period, interval = _yahoo_range(timeframe)

    result = _fetch_yahoo_payload(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    timestamps = result.get(
        "timestamp",
        [],
    )

    indicators = result.get(
        "indicators",
        {},
    )

    quotes = indicators.get(
        "quote",
        [],
    )

    if not quotes:
        raise ValueError(
            f"Yahoo thiáº¿u OHLCV {symbol}."
        )

    quote = quotes[0]

    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    length = min(
        len(timestamps),
        len(opens),
        len(highs),
        len(lows),
        len(closes),
        len(volumes),
    )

    rows = []

    for i in range(length):

        try:

            timestamp = pd.to_datetime(
                int(timestamps[i]),
                unit="s",
                utc=True,
            )

            open_price = _safe_float(opens[i])
            high_price = _safe_float(highs[i])
            low_price = _safe_float(lows[i])
            close_price = _safe_float(closes[i])
            volume = _safe_float(volumes[i])

            if (
                open_price is None
                or high_price is None
                or low_price is None
                or close_price is None
            ):
                continue

            rows.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": (
                    volume
                    if volume is not None
                    else 0.0
                ),
            })

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

    if not rows:
        raise ValueError(
            f"Yahoo khÃ´ng cÃ³ candle há»£p lá»‡ "
            f"cho {symbol} {timeframe}."
        )

    df = _validate_ohlcv(
        pd.DataFrame(rows)
    )

    if timeframe == "H4":

        df = _resample_h4(df)

        df = _validate_ohlcv(df)

    if len(df) > int(limit):

        df = (
            df.tail(int(limit))
            .reset_index(drop=True)
        )

    if len(df) < 10:
        raise ValueError(
            f"{symbol} {timeframe} chá»‰ cÃ³ "
            f"{len(df)} candle."
        )

    return df


# ============================================================
# SYMBOL DETECTION
# ============================================================

def is_tradingview_symbol(
    symbol: str,
) -> bool:

    return (
        _clean_symbol(symbol)
        in TV_SYMBOL_MAP
    )


# ============================================================
# MAIN MARKET DATA ROUTER
# ============================================================

def fetch_market_klines(
    symbol: str,
    timeframe: str,
    limit: int = 300,
):

    symbol = _clean_symbol(symbol)

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

        return fetch_yahoo_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

    return _binance_to_dataframe(
        fetch_binance_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
    )


# ============================================================
# CACHED MARKET DATA
# ============================================================

def fetch_market_dataframe_cached(
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:

    symbol = _clean_symbol(symbol)
    timeframe = _clean_timeframe(timeframe)
    limit = int(limit)

    key = (
        symbol,
        timeframe,
        limit,
    )

    now = time.time()

    with _CACHE_LOCK:

        cached = _DATA_CACHE.get(key)

        if cached is not None:

            cached_time, cached_df = cached

            if (
                now - cached_time
                <= CACHE_SECONDS
            ):
                return cached_df.copy()

    df = fetch_market_dataframe(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    with _CACHE_LOCK:

        _DATA_CACHE[key] = (
            time.time(),
            df.copy(),
        )

    return df.copy()


# ============================================================
# QUOTE
# ============================================================


def get_latest_quotes_batch(
    symbols: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    requested = list(
        dict.fromkeys(
            _clean_symbol(x)
            for x in symbols
            if isinstance(x, str) and x.strip()
        )
    )
    quotes = {}
    errors = {}
    binance_symbols = [
        x for x in requested
        if x not in TV_SYMBOL_MAP
    ]
    # --------------------------------------------------------
    # BINANCE: 1 BULK REQUEST FOR ALL CRYPTO
    # --------------------------------------------------------
    if binance_symbols:
        try:
            response = requests.get(
                f"{BINANCE_BASE_URL}/api/v3/ticker/24hr",
                params={
                    "symbols": json.dumps(
                        binance_symbols,
                        separators=(",", ":"),
                    )
                },
                timeout=(3, 8),
                headers={
                    "User-Agent":
                        "NovaTradeAI/7.2.8",
                    "Accept":
                        "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(
                    "Binance bulk ticker kh?ng h?p l?."
                )
            for item in payload:
                try:
                    symbol = _clean_symbol(
                        item.get("symbol", "")
                    )
                    price = _safe_float(
                        item.get("lastPrice")
                    )
                    change = _safe_float(
                        item.get("priceChangePercent")
                    )
                    if price is None:
                        continue
                    previous = price
                    if (
                        change is not None
                        and (1.0 + change / 100.0) != 0
                    ):
                        previous = (
                            price
                            / (1.0 + change / 100.0)
                        )
                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": price,
                        "previous": previous,
                        "change_percent": change,
                        "timestamp":
                            pd.Timestamp.now(
                                tz="UTC"
                            ).isoformat(),
                        "source":
                            get_symbol_info(symbol),
                        "analysis_only": True,
                    }
                except Exception as exc:
                    errors[
                        str(item.get(
                            "symbol",
                            "UNKNOWN"
                        ))
                    ] = str(exc)
            for symbol in binance_symbols:
                if (
                    symbol not in quotes
                    and symbol not in errors
                ):
                    errors[symbol] = (
                        "Binance kh?ng tr? quote."
                    )
        except Exception as exc:
            for symbol in binance_symbols:
                errors[symbol] = str(exc)
    # --------------------------------------------------------
    # TRADINGVIEW / YAHOO SYMBOLS
    # --------------------------------------------------------
    for symbol in requested:
        if symbol in TV_SYMBOL_MAP:
            try:
                quotes[symbol] = get_latest_quote(
                    symbol
                )
            except Exception as exc:
                errors[symbol] = str(exc)
    return quotes, errors
def get_latest_quote(
    symbol: str,
) -> dict[str, Any]:

    symbol = _clean_symbol(symbol)

    df = fetch_market_dataframe_cached(
        symbol=symbol,
        timeframe="M1",
        limit=2,
    )

    if df.empty:
        raise ValueError(
            f"KhÃ´ng cÃ³ quote cho {symbol}."
        )

    last = df.iloc[-1]

    previous = (
        df.iloc[-2]
        if len(df) >= 2
        else last
    )

    price = float(last["close"])
    previous_price = float(previous["close"])

    change = None

    if previous_price != 0:

        change = (
            (price - previous_price)
            / previous_price
        ) * 100.0

    return {
        "symbol": symbol,
        "price": price,
        "previous": previous_price,
        "change_percent": change,
        "timestamp": last["timestamp"].isoformat(),
        "source": get_symbol_info(symbol),
        "analysis_only": True,
    }


# ============================================================
# SYMBOL CHECK
# ============================================================

def check_binance_symbol(
    symbol: str,
) -> bool:

    symbol = _clean_symbol(symbol)

    try:

        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/exchangeInfo",
            params={
                "symbol": symbol
            },
            timeout=(3, 5),
            headers={
                "User-Agent":
                    "NovaTradeAI/6.1",
                "Accept":
                    "application/json",
            },
        )

        if response.status_code != 200:
            return False

        data = response.json()

        return bool(
            data.get(
                "symbols",
                [],
            )
        )

    except Exception:
        return False


# ============================================================
# SYMBOL INFO
# ============================================================

def get_symbol_info(
    symbol: str,
) -> dict[str, Any]:

    symbol = _clean_symbol(symbol)

    if symbol in TV_SYMBOL_MAP:

        item = TV_SYMBOL_MAP[symbol]

        return {
            "symbol": symbol,
            "tv_symbol": item["tv"],
            "provider": item["provider"],
            "ticker": item["ticker"],
            "name": item["name"],
            "market_type": item["market_type"],
            "proxy": True,
            "analysis_only": True,
        }

    return {
        "symbol": symbol,
        "tv_symbol": None,
        "provider": "BINANCE",
        "ticker": symbol,
        "name": symbol,
        "market_type": "crypto",
        "proxy": False,
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

    return fetch_market_dataframe_cached(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )







