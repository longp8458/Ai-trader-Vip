import pandas as pd
import numpy as np


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


def macd(series):

    ema12 = ema(series, 12)
    ema26 = ema(series, 26)

    macd_line = ema12 - ema26

    signal = macd_line.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd_line - signal

    return macd_line, signal, histogram


def bollinger_bands(series, period=20):

    middle = sma(series, period)

    std = series.rolling(period).std()

    upper = middle + 2 * std
    lower = middle - 2 * std

    return upper, middle, lower


def add_indicators(df):

    df = df.copy()

    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)

    df["sma20"] = sma(df["close"], 20)
    df["sma50"] = sma(df["close"], 50)
    df["sma200"] = sma(df["close"], 200)

    df["rsi"] = rsi(df["close"])

    df["atr"] = atr(df)

    macd_line, signal, histogram = macd(
        df["close"]
    )

    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_histogram"] = histogram

    upper, middle, lower = bollinger_bands(
        df["close"]
    )

    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    df["volume_sma20"] = sma(
        df["volume"], 20
    )

    return df