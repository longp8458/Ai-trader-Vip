# ============================================================
# NovaTrade AI V6 - TECHNICAL INDICATORS
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def finite_number(value):
    """
    Chuyển giá trị về float an toàn.
    NaN / Inf sẽ thành None.
    """
    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


# ============================================================
# SMA
# ============================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(
        window=period,
        min_periods=period
    ).mean()


# ============================================================
# EMA
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    result = pd.Series(
        index=series.index,
        dtype=float
    )

    normal = avg_loss > 0

    result.loc[normal] = (
        100
        - (
            100
            / (
                1
                + avg_gain.loc[normal]
                / avg_loss.loc[normal]
            )
        )
    )

    only_gain = (avg_loss == 0) & (avg_gain > 0)
    result.loc[only_gain] = 100

    only_loss = (avg_gain == 0) & (avg_loss > 0)
    result.loc[only_loss] = 0

    flat = (avg_gain == 0) & (avg_loss == 0)
    result.loc[flat] = 50

    return result


# ============================================================
# ATR
# ============================================================

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# MACD
# ============================================================

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
):

    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal
    ).mean()

    histogram = macd_line - signal_line

    return (
        macd_line,
        signal_line,
        histogram
    )


# ============================================================
# Bollinger Bands
# ============================================================

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
):

    middle = series.rolling(
        period,
        min_periods=period
    ).mean()

    std = series.rolling(
        period,
        min_periods=period
    ).std()

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return (
        upper,
        middle,
        lower
    )


# ============================================================
# Volume SMA
# ============================================================

def volume_sma(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:

    return volume.rolling(
        period,
        min_periods=period
    ).mean()


# ============================================================
# Add all indicators
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Thiếu cột dữ liệu: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # EMA
    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)

    # SMA
    df["sma20"] = sma(df["close"], 20)
    df["sma50"] = sma(df["close"], 50)
    df["sma200"] = sma(df["close"], 200)

    # RSI
    df["rsi"] = rsi(
        df["close"],
        14
    )

    # MACD
    (
        df["macd"],
        df["macd_signal"],
        df["macd_histogram"],
    ) = macd(
        df["close"]
    )

    # ATR
    df["atr"] = atr(
        df["high"],
        df["low"],
        df["close"],
        14
    )

    # Bollinger Bands
    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"],
    ) = bollinger_bands(
        df["close"],
        20,
        2.0
    )

    # Volume
    df["volume_sma20"] = volume_sma(
        df["volume"],
        20
    )

    return df


# ============================================================
# Sanitize
# ============================================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    return result