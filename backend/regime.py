import pandas as pd


def detect_market_regime(df):
    """
    Xác định trạng thái thị trường dựa trên:
    - EMA
    - ADX nếu có
    - ATR
    - biến động giá
    """

    if df is None or len(df) < 50:
        return {
            "regime": "UNKNOWN",
            "trend": "NEUTRAL",
            "volatility": "UNKNOWN",
            "reason": "Không đủ dữ liệu"
        }

    data = df.copy()

    close = data["close"]

    # =====================================================
    # EMA
    # =====================================================

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================================
    # ATR
    # =====================================================

    high_low = (
        data["high"]
        - data["low"]
    )

    high_close = (
        data["high"]
        - data["close"].shift()
    ).abs()

    low_close = (
        data["low"]
        - data["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    price = close.iloc[-1]

    if price == 0:
        return {
            "regime": "UNKNOWN",
            "trend": "NEUTRAL",
            "volatility": "UNKNOWN",
            "reason": "Giá không hợp lệ"
        }

    # =====================================================
    # ATR %
    # =====================================================

    atr_percent = (
        atr / price
    ) * 100

    # =====================================================
    # TREND
    # =====================================================

    if (
        price > ema20
        and ema20 > ema50
    ):

        trend = "BULLISH"

    elif (
        price < ema20
        and ema20 < ema50
    ):

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"

    # =====================================================
    # VOLATILITY
    # =====================================================

    if atr_percent >= 3:

        volatility = "HIGH"

    elif atr_percent <= 0.5:

        volatility = "LOW"

    else:

        volatility = "NORMAL"

    # =====================================================
    # MARKET REGIME
    # =====================================================

    if (
        trend == "BULLISH"
        and volatility == "HIGH"
    ):

        regime = "TRENDING_BULLISH_HIGH_VOLATILITY"

    elif (
        trend == "BULLISH"
    ):

        regime = "TRENDING_BULLISH"

    elif (
        trend == "BEARISH"
        and volatility == "HIGH"
    ):

        regime = "TRENDING_BEARISH_HIGH_VOLATILITY"

    elif (
        trend == "BEARISH"
    ):

        regime = "TRENDING_BEARISH"

    elif (
        volatility == "HIGH"
    ):

        regime = "HIGH_VOLATILITY"

    elif (
        volatility == "LOW"
    ):

        regime = "LOW_VOLATILITY"

    else:

        regime = "RANGING"

    # =====================================================
    # REASON
    # =====================================================

    reasons = []

    if trend == "BULLISH":

        reasons.append(
            "Giá nằm trên EMA20 và EMA20 nằm trên EMA50."
        )

    elif trend == "BEARISH":

        reasons.append(
            "Giá nằm dưới EMA20 và EMA20 nằm dưới EMA50."
        )

    else:

        reasons.append(
            "EMA20 và EMA50 chưa xác nhận xu hướng rõ."
        )

    reasons.append(
        f"ATR = {atr:.4f} "
        f"({atr_percent:.2f}% giá)."
    )

    return {
        "regime": regime,
        "trend": trend,
        "volatility": volatility,
        "atr": round(float(atr), 6),
        "atr_percent": round(
            float(atr_percent),
            4
        ),
        "ema20": round(
            float(ema20),
            6
        ),
        "ema50": round(
            float(ema50),
            6
        ),
        "reason": " ".join(reasons)
    }