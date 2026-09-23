import numpy as np
import pandas as pd

EPS = 1e-12

def ema(series, period):
    return pd.to_numeric(series, errors="coerce").ewm(span=period, adjust=False, min_periods=period).mean()

def sma(series, period):
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).mean()

def rsi(series, period=14):
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # Rising-only windows => RSI 100; falling-only windows => RSI 0.
    result = result.mask((avg_loss <= EPS) & (avg_gain > EPS), 100.0)
    result = result.mask((avg_gain <= EPS) & (avg_loss > EPS), 0.0)
    result = result.mask((avg_gain <= EPS) & (avg_loss <= EPS), 50.0)
    return result.clip(0, 100)

def atr(df, period=14):
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def macd(series):
    close = pd.to_numeric(series, errors="coerce")
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line, signal, line - signal

def bollinger_bands(series, period=20, std_mult=2.0):
    middle = sma(series, period)
    std = pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).std()
    return middle + std_mult * std, middle, middle - std_mult * std

def add_indicators(df):
    data = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["ema9"] = ema(data["close"], 9)
    data["ema20"] = ema(data["close"], 20)
    data["ema50"] = ema(data["close"], 50)
    data["ema100"] = ema(data["close"], 100)
    data["ema200"] = ema(data["close"], 200)

    data["sma20"] = sma(data["close"], 20)
    data["sma50"] = sma(data["close"], 50)
    data["sma200"] = sma(data["close"], 200)

    data["rsi"] = rsi(data["close"], 14)
    data["atr"] = atr(data, 14)

    line, signal, hist = macd(data["close"])
    data["macd"] = line
    data["macd_signal"] = signal
    data["macd_histogram"] = hist

    upper, middle, lower = bollinger_bands(data["close"], 20, 2.0)
    data["bb_upper"] = upper
    data["bb_middle"] = middle
    data["bb_lower"] = lower
    data["volume_sma20"] = sma(data["volume"], 20)

    return data

def finite_number(value, default=None):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default

def sanitize_dataframe(df):
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return data
