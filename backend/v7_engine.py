from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from backend.levels import calculate_levels


ENGINE_VERSION = "7.1-HIGH-CONVICTION"


# =========================================================
# SAFE HELPERS
# =========================================================

def _num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)

        if math.isfinite(x):
            return x

    except (TypeError, ValueError):
        pass

    return default


def _finite(value: Any) -> float | None:
    try:
        x = float(value)

        if math.isfinite(x):
            return x

    except (TypeError, ValueError):
        pass

    return None


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    data.columns = [
        str(c).lower()
        for c in data.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:

        if column not in data.columns:
            data[column] = 0.0

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return data.reset_index(drop=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def _ema(series: pd.Series, period: int) -> float:

    value = (
        series
        .ewm(
            span=period,
            adjust=False,
            min_periods=period,
        )
        .mean()
        .iloc[-1]
    )

    return _num(value)


def _sma(series: pd.Series, period: int) -> float:

    value = (
        series
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
        .iloc[-1]
    )

    return _num(value)


def _rsi(
    close: pd.Series,
    period: int = 14,
) -> float:

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    last_gain = _num(avg_gain.iloc[-1])
    last_loss = _num(avg_loss.iloc[-1])

    if last_loss == 0:

        if last_gain > 0:
            return 100.0

        return 50.0

    rs = last_gain / last_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def _atr(
    data: pd.DataFrame,
    period: int = 14,
) -> float:

    high = data["high"]
    low = data["low"]
    close = data["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    value = (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
        .iloc[-1]
    )

    atr = _finite(value)

    if atr is not None and atr > 0:
        return atr

    price = max(
        _num(close.iloc[-1]),
        1.0,
    )

    return price * 0.005


def _macd(
    close: pd.Series,
) -> tuple[float, float, float]:

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    line = ema12 - ema26

    signal = (
        line
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    histogram = line - signal

    return (
        _num(line.iloc[-1]),
        _num(signal.iloc[-1]),
        _num(histogram.iloc[-1]),
    )


def _bollinger(
    close: pd.Series,
    period: int = 20,
) -> tuple[float, float, float]:

    middle = (
        close
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
        .iloc[-1]
    )

    std = (
        close
        .rolling(
            period,
            min_periods=period,
        )
        .std()
        .iloc[-1]
    )

    middle = _num(middle)
    std = _num(std)

    upper = middle + 2 * std
    lower = middle - 2 * std

    return (
        lower,
        middle,
        upper,
    )


def _adx(
    data: pd.DataFrame,
    period: int = 14,
) -> tuple[float, float, float]:

    high = data["high"]
    low = data["low"]
    close = data["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=data.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=data.index,
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    plus_di = (
        100
        * plus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
        / atr.replace(0, np.nan)
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan,
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    adx = (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    return (
        _num(adx.iloc[-1]),
        _num(plus_di.iloc[-1]),
        _num(minus_di.iloc[-1]),
    )


def _obv(
    data: pd.DataFrame,
) -> tuple[float, str, float]:

    close = data["close"]
    volume = data["volume"].fillna(0)

    direction = np.sign(
        close.diff().fillna(0)
    )

    obv_series = (
        direction * volume
    ).cumsum()

    obv = _num(
        obv_series.iloc[-1]
    )

    if len(obv_series) >= 10:

        recent = (
            obv_series
            .iloc[-5:]
            .mean()
        )

        previous = (
            obv_series
            .iloc[-10:-5]
            .mean()
        )

        if recent > previous:
            trend = "RISING"

        elif recent < previous:
            trend = "FALLING"

        else:
            trend = "FLAT"

    else:
        trend = "FLAT"

    volume_sma = _sma(
        volume,
        20,
    )

    current_volume = _num(
        volume.iloc[-1]
    )

    ratio = (
        current_volume
        / max(volume_sma, 1e-9)
    )

    return (
        obv,
        trend,
        ratio,
    )


def _technical_indicators(
    data: pd.DataFrame,
) -> dict[str, Any]:

    close = data["close"]

    price = _num(
        close.iloc[-1]
    )

    ema20 = _ema(
        close,
        20,
    )

    ema50 = _ema(
        close,
        50,
    )

    ema200 = _ema(
        close,
        200,
    )

    sma20 = _sma(
        close,
        20,
    )

    sma50 = _sma(
        close,
        50,
    )

    sma200 = _sma(
        close,
        200,
    )

    rsi = _rsi(
        close,
        14,
    )

    macd_line, macd_signal, macd_hist = _macd(
        close
    )

    atr = _atr(
        data,
        14,
    )

    bb_lower, bb_middle, bb_upper = _bollinger(
        close,
        20,
    )

    adx, plus_di, minus_di = _adx(
        data,
        14,
    )

    obv, obv_trend, volume_ratio = _obv(
        data
    )

    vwap = (
        (
            data["close"]
            * data["volume"]
        ).rolling(
            20,
            min_periods=1,
        ).sum()
        /
        data["volume"]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
        .replace(0, np.nan)
    ).iloc[-1]

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "atr": atr,
        "bb_lower": bb_lower,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "volume": _num(
            data["volume"].iloc[-1]
        ),
        "volume_sma20": _sma(
            data["volume"],
            20,
        ),
        "volume_ratio": volume_ratio,
        "obv": obv,
        "obv_trend": obv_trend,
        "vwap": _num(vwap),
    }


# =========================================================
# FALLBACK MARKET STRUCTURE
# =========================================================

def _fallback_structure(
    data: pd.DataFrame,
) -> dict[str, Any]:

    if len(data) < 20:

        return {
            "structure": "UNKNOWN",
            "trend": "NEUTRAL",
        }

    recent = data.tail(60)

    highs = (
        recent["high"]
        .rolling(
            5,
            center=True,
        )
        .max()
    )

    lows = (
        recent["low"]
        .rolling(
            5,
            center=True,
        )
        .min()
    )

    swing_highs = []

    swing_lows = []

    for i in range(
        2,
        len(recent) - 2,
    ):

        h = _num(
            recent["high"].iloc[i]
        )

        l = _num(
            recent["low"].iloc[i]
        )

        local_h = _num(
            highs.iloc[i]
        )

        local_l = _num(
            lows.iloc[i]
        )

        if h >= local_h:
            swing_highs.append(h)

        if l <= local_l:
            swing_lows.append(l)

    if len(swing_highs) >= 2:

        high_structure = (
            "HH"
            if swing_highs[-1]
            > swing_highs[-2]
            else "LH"
        )

    else:
        high_structure = "UNKNOWN"

    if len(swing_lows) >= 2:

        low_structure = (
            "HL"
            if swing_lows[-1]
            > swing_lows[-2]
            else "LL"
        )

    else:
        low_structure = "UNKNOWN"

    if (
        high_structure == "HH"
        and low_structure == "HL"
    ):
        trend = "BULLISH"
        structure = "HH_HL"

    elif (
        high_structure == "LH"
        and low_structure == "LL"
    ):
        trend = "BEARISH"
        structure = "LH_LL"

    else:
        trend = "NEUTRAL"
        structure = "MIXED"

    return {
        "structure": structure,
        "trend": trend,
        "high_structure": high_structure,
        "low_structure": low_structure,
        "last_swing_high": (
            swing_highs[-1]
            if swing_highs
            else None
        ),
        "last_swing_low": (
            swing_lows[-1]
            if swing_lows
            else None
        ),
    }


# =========================================================
# FALLBACK CANDLESTICK
# =========================================================

def _fallback_candle(
    data: pd.DataFrame,
) -> dict[str, Any]:

    if len(data) < 2:

        return {
            "pattern": "UNKNOWN",
            "signal": "NEUTRAL",
            "strength": 0,
        }

    last = data.iloc[-1]
    prev = data.iloc[-2]

    o = _num(last["open"])
    h = _num(last["high"])
    l = _num(last["low"])
    c = _num(last["close"])

    po = _num(prev["open"])
    pc = _num(prev["close"])

    body = abs(c - o)

    candle_range = max(
        h - l,
        1e-9,
    )

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    bullish = c > o
    bearish = c < o

    pattern = "NORMAL"
    signal = "NEUTRAL"
    strength = 0

    # Bullish engulfing
    if (
        pc < po
        and c > o
        and c >= po
        and o <= pc
    ):

        pattern = "BULLISH_ENGULFING"
        signal = "BUY"
        strength = 2

    # Bearish engulfing
    elif (
        pc > po
        and c < o
        and o >= pc
        and c <= po
    ):

        pattern = "BEARISH_ENGULFING"
        signal = "SELL"
        strength = 2

    # Hammer
    elif (
        lower_wick >= body * 2
        and upper_wick <= body
        and bullish
    ):

        pattern = "HAMMER"
        signal = "BUY"
        strength = 1

    # Shooting star
    elif (
        upper_wick >= body * 2
        and lower_wick <= body
        and bearish
    ):

        pattern = "SHOOTING_STAR"
        signal = "SELL"
        strength = 1

    # Doji
    elif (
        body <= candle_range * 0.10
    ):

        pattern = "DOJI"
        signal = "NEUTRAL"
        strength = 1

    return {
        "pattern": pattern,
        "signal": signal,
        "strength": strength,
    }


# =========================================================
# NORMALIZE ANALYZER OUTPUT
# =========================================================

def _extract_structure(
    analysis: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, Any]:

    value = (
        analysis.get("structure")
        if isinstance(analysis, dict)
        else None
    )

    if isinstance(value, dict):

        return value

    return _fallback_structure(
        data
    )


def _extract_candle(
    analysis: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, Any]:

    value = (
        analysis.get("candlestick")
        if isinstance(analysis, dict)
        else None
    )

    if isinstance(value, dict):

        return value

    return _fallback_candle(
        data
    )


# =========================================================
# MAIN ENGINE
# =========================================================

def enrich(
    df: pd.DataFrame,
    analysis: dict[str, Any] | None,
    news: dict[str, Any] | None = None,
) -> dict[str, Any]:

    data = _clean_df(df)

    if len(data) < 50:

        return {
            "engine_version": ENGINE_VERSION,
            "signal": "NEUTRAL",
            "final_signal": "NEUTRAL",
            "confidence": 50.0,
            "signal_quality": 0.0,
            "directional_edge": 0.0,
            "buy_score": 0.0,
            "sell_score": 0.0,
            "price": None,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "trade_levels": {},
            "levels": {},
            "high_conviction": False,
            "reasoning": [
                "Không đủ dữ liệu để phân tích."
            ],
            "analysis_only": True,
        }

    analysis = (
        analysis
        if isinstance(analysis, dict)
        else {}
    )

    news = (
        news
        if isinstance(news, dict)
        else {}
    )

    indicators = _technical_indicators(
        data
    )

    structure = _extract_structure(
        analysis,
        data,
    )

    candle = _extract_candle(
        analysis,
        data,
    )

    levels = calculate_levels(
        data
    )

    price = indicators["price"]
    atr = max(
        indicators["atr"],
        price * 0.001,
    )

    buy = 0.0
    sell = 0.0

    reasons: list[str] = []

    # -----------------------------------------------------
    # 1. EMA TREND
    # -----------------------------------------------------

    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    ema200 = indicators["ema200"]

    if ema20 > ema50:

        buy += 1.5

        reasons.append(
            f"EMA20 > EMA50 "
            f"({ema20:.2f} > {ema50:.2f})"
        )

    elif ema20 < ema50:

        sell += 1.5

        reasons.append(
            f"EMA20 < EMA50 "
            f"({ema20:.2f} < {ema50:.2f})"
        )

    if price > ema200:

        buy += 0.8

        reasons.append(
            "Giá nằm trên EMA200."
        )

    elif price < ema200:

        sell += 0.8

        reasons.append(
            "Giá nằm dưới EMA200."
        )

    # -----------------------------------------------------
    # 2. RSI
    # -----------------------------------------------------

    rsi = indicators["rsi"]

    if 52 <= rsi <= 68:

        buy += 1.0

        reasons.append(
            f"RSI bullish: {rsi:.2f}"
        )

    elif 32 <= rsi <= 48:

        sell += 1.0

        reasons.append(
            f"RSI bearish: {rsi:.2f}"
        )

    elif rsi > 72:

        # Không coi overbought là SELL tự động.
        # Chỉ giảm nhẹ bullish conviction.
        buy += 0.25

        reasons.append(
            f"RSI cao: {rsi:.2f}, "
            "động lượng tăng nhưng vùng nóng."
        )

    elif rsi < 28:

        sell += 0.25

        reasons.append(
            f"RSI thấp: {rsi:.2f}, "
            "động lượng yếu nhưng vùng quá bán."
        )

    # -----------------------------------------------------
    # 3. MACD
    # -----------------------------------------------------

    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    macd_hist = indicators["macd_histogram"]

    if macd > macd_signal:

        buy += 1.2

        reasons.append(
            f"MACD bullish "
            f"({macd:.4f} > {macd_signal:.4f})"
        )

    elif macd < macd_signal:

        sell += 1.2

        reasons.append(
            f"MACD bearish "
            f"({macd:.4f} < {macd_signal:.4f})"
        )

    if macd_hist > 0:

        buy += 0.3

    elif macd_hist < 0:

        sell += 0.3

    # -----------------------------------------------------
    # 4. ADX + DI
    # -----------------------------------------------------

    adx = indicators["adx"]
    plus_di = indicators["plus_di"]
    minus_di = indicators["minus_di"]

    if adx >= 20:

        if plus_di > minus_di:

            buy += 0.9

            reasons.append(
                f"ADX {adx:.2f}: "
                "trend strength nghiêng bullish."
            )

        elif minus_di > plus_di:

            sell += 0.9

            reasons.append(
                f"ADX {adx:.2f}: "
                "trend strength nghiêng bearish."
            )

    else:

        reasons.append(
            f"ADX {adx:.2f}: thị trường chưa có trend mạnh."
        )

    # -----------------------------------------------------
    # 5. MARKET STRUCTURE
    # -----------------------------------------------------

    structure_name = str(
        structure.get("structure", "")
    ).upper()

    structure_trend = str(
        structure.get("trend", "")
    ).upper()

    if (
        "HH_HL" in structure_name
        or structure_trend == "BULLISH"
    ):

        buy += 1.8

        reasons.append(
            "Market Structure bullish."
        )

    elif (
        "LH_LL" in structure_name
        or structure_trend == "BEARISH"
    ):

        sell += 1.8

        reasons.append(
            "Market Structure bearish."
        )

    # BOS
    bos_direction = str(
        structure.get(
            "bos_direction",
            "",
        )
    ).upper()

    if bos_direction == "BULLISH":

        buy += 0.7

        reasons.append(
            "Có Bullish BOS."
        )

    elif bos_direction == "BEARISH":

        sell += 0.7

        reasons.append(
            "Có Bearish BOS."
        )

    # -----------------------------------------------------
    # 6. CANDLESTICK
    # -----------------------------------------------------

    candle_signal = str(
        candle.get(
            "signal",
            "",
        )
    ).upper()

    candle_pattern = str(
        candle.get(
            "pattern",
            "",
        )
    ).upper()

    candle_strength = _num(
        candle.get(
            "strength",
            0,
        )
    )

    if candle_signal == "BUY":

        buy += min(
            1.5,
            0.75
            * max(
                1,
                candle_strength,
            ),
        )

        reasons.append(
            f"Nến bullish: {candle_pattern}"
        )

    elif candle_signal == "SELL":

        sell += min(
            1.5,
            0.75
            * max(
                1,
                candle_strength,
            ),
        )

        reasons.append(
            f"Nến bearish: {candle_pattern}"
        )

    elif candle_pattern:

        reasons.append(
            f"Nến: {candle_pattern}"
        )

    # -----------------------------------------------------
    # 7. VOLUME / OBV / VWAP
    # -----------------------------------------------------

    volume_ratio = indicators[
        "volume_ratio"
    ]

    obv_trend = indicators[
        "obv_trend"
    ]

    vwap = indicators[
        "vwap"
    ]

    if obv_trend == "RISING":

        buy += 0.6

        reasons.append(
            "OBV đang tăng."
        )

    elif obv_trend == "FALLING":

        sell += 0.6

        reasons.append(
            "OBV đang giảm."
        )

    if price > vwap:

        buy += 0.35

        reasons.append(
            "Giá nằm trên VWAP."
        )

    elif price < vwap:

        sell += 0.35

        reasons.append(
            "Giá nằm dưới VWAP."
        )

    if volume_ratio >= 1.20:

        if price > vwap:
            buy += 0.35

        elif price < vwap:
            sell += 0.35

    # -----------------------------------------------------
    # 8. BOLLINGER
    # -----------------------------------------------------

    bb_upper = indicators[
        "bb_upper"
    ]

    bb_lower = indicators[
        "bb_lower"
    ]

    bb_middle = indicators[
        "bb_middle"
    ]

    if price > bb_middle:

        buy += 0.25

    elif price < bb_middle:

        sell += 0.25

    # Không tạo BUY/SELL chỉ vì chạm band.
    # Dùng BB như context.

    # -----------------------------------------------------
    # 9. SUPPORT / RESISTANCE
    # -----------------------------------------------------

    nearest_support = levels.get(
        "nearest_support"
    )

    nearest_resistance = levels.get(
        "nearest_resistance"
    )

    distance_support = levels.get(
        "distance_to_support_atr"
    )

    distance_resistance = levels.get(
        "distance_to_resistance_atr"
    )

    if (
        nearest_support
        and distance_support is not None
        and distance_support <= 1.25
    ):

        if buy > sell:

            buy += 1.0

            reasons.append(
                "Giá đang gần vùng hỗ trợ."
            )

    if (
        nearest_resistance
        and distance_resistance is not None
        and distance_resistance <= 1.25
    ):

        if sell > buy:

            sell += 1.0

            reasons.append(
                "Giá đang gần vùng kháng cự."
            )

    # -----------------------------------------------------
    # 10. NEWS
    # -----------------------------------------------------

    news_score = _num(
        news.get(
            "sentiment_score",
            0,
        )
    )

    if news_score > 0.15:

        buy += 0.5

        reasons.append(
            "News sentiment đang nghiêng tích cực."
        )

    elif news_score < -0.15:

        sell += 0.5

        reasons.append(
            "News sentiment đang nghiêng tiêu cực."
        )

    # -----------------------------------------------------
    # 11. FINAL SCORE
    # -----------------------------------------------------

    total = max(
        buy + sell,
        1e-9,
    )

    edge = (
        abs(buy - sell)
        / total
    )

    if buy > sell:

        signal = "BUY"

    elif sell > buy:

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    # Không ép tín hiệu khi chênh lệch quá nhỏ.

    if edge < 0.12:

        signal = "NEUTRAL"

    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    agreement = (
        max(buy, sell)
        / total
    )

    quality = (
        35
        + agreement * 35
        + edge * 30
    )

    quality = min(
        100,
        max(
            0,
            quality,
        ),
    )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = (
        45
        + edge * 40
        + agreement * 15
    )

    if adx >= 25:
        confidence += 4

    if volume_ratio >= 1.20:
        confidence += 2

    confidence = min(
        96,
        max(
            45,
            confidence,
        ),
    )

    # -----------------------------------------------------
    # HIGH CONVICTION
    # -----------------------------------------------------

    high_conviction = (
        signal in {
            "BUY",
            "SELL",
        }
        and quality >= 68
        and confidence >= 64
        and edge >= 0.22
    )

    if not high_conviction:

        if signal != "NEUTRAL":

            reasons.append(
                "Chưa đạt ngưỡng High-Conviction."
            )

        signal = "NEUTRAL"

    # -----------------------------------------------------
    # TRADE REFERENCE LEVELS
    # -----------------------------------------------------

    entry = price

    sl = None
    tp1 = None
    tp2 = None
    tp3 = None

    if (
        signal == "BUY"
        and nearest_support
    ):

        support_low = _num(
            nearest_support.get(
                "low",
                nearest_support.get(
                    "price",
                    price - atr,
                ),
            )
        )

        sl = (
            support_low
            - 0.35 * atr
        )

        risk = max(
            entry - sl,
            0.60 * atr,
        )

        tp1 = entry + risk
        tp2 = entry + 2 * risk
        tp3 = entry + 3 * risk

    elif (
        signal == "SELL"
        and nearest_resistance
    ):

        resistance_high = _num(
            nearest_resistance.get(
                "high",
                nearest_resistance.get(
                    "price",
                    price + atr,
                ),
            )
        )

        sl = (
            resistance_high
            + 0.35 * atr
        )

        risk = max(
            sl - entry,
            0.60 * atr,
        )

        tp1 = entry - risk
        tp2 = entry - 2 * risk
        tp3 = entry - 3 * risk

    # -----------------------------------------------------
    # FINAL REASONING
    # -----------------------------------------------------

    if not reasons:

        reasons.append(
            "Chưa có bằng chứng đủ mạnh."
        )

    return {
        "engine_version": ENGINE_VERSION,

        "signal": signal,

        "final_signal": signal,

        "confidence": round(
            confidence,
            2,
        ),

        "signal_quality": round(
            quality,
            2,
        ),

        "directional_edge": round(
            edge,
            4,
        ),

        "buy_score": round(
            buy,
            3,
        ),

        "sell_score": round(
            sell,
            3,
        ),

        "price": price,

        "entry": entry,

        "sl": sl,

        "tp1": tp1,

        "tp2": tp2,

        "tp3": tp3,

        "trade_levels": {
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
        },

        "indicators": indicators,

        "structure": structure,

        "candlestick": candle,

        "levels": levels,

        "high_conviction": high_conviction,

        "news_context": news,

        "reasoning": reasons[:12],

        "analysis_only": True,
    }