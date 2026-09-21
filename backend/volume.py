import pandas as pd
import numpy as np


def calculate_obv(df):
    """
    Tính On-Balance Volume (OBV).
    """

    close = df["close"]
    volume = df["volume"]

    direction = np.sign(
        close.diff()
    ).fillna(0)

    obv = (
        direction * volume
    ).cumsum()

    return obv


def calculate_vwap(df):
    """
    Tính VWAP dựa trên giá điển hình.
    """

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    cumulative_volume = (
        df["volume"].cumsum()
    )

    cumulative_value = (
        typical_price * df["volume"]
    ).cumsum()

    return (
        cumulative_value
        /
        cumulative_volume.replace(
            0,
            np.nan
        )
    )


def analyze_volume(df):
    """
    Phân tích Volume:

    - Volume SMA
    - Volume Ratio
    - Volume Spike
    - OBV
    - OBV Trend
    - VWAP
    - Volume Signal
    """

    if df is None or len(df) < 20:

        return {
            "volume": None,
            "volume_sma20": None,
            "volume_ratio": None,
            "volume_spike": False,
            "obv": None,
            "obv_trend": "UNKNOWN",
            "vwap": None,
            "signal": "NEUTRAL",
            "strength": 0,
            "reason": "Không đủ dữ liệu Volume."
        }

    data = df.copy()

    data["volume_sma20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["obv"] = calculate_obv(
        data
    )

    data["vwap"] = calculate_vwap(
        data
    )

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    volume = float(
        latest["volume"]
    )

    volume_sma20 = float(
        latest["volume_sma20"]
    )

    if volume_sma20 <= 0:

        return {
            "volume": volume,
            "volume_sma20": volume_sma20,
            "volume_ratio": None,
            "volume_spike": False,
            "obv": float(latest["obv"]),
            "obv_trend": "UNKNOWN",
            "vwap": float(latest["vwap"]),
            "signal": "NEUTRAL",
            "strength": 0,
            "reason": "Volume trung bình không hợp lệ."
        }

    # =====================================================
    # VOLUME RATIO
    # =====================================================

    volume_ratio = (
        volume
        /
        volume_sma20
    )

    # =====================================================
    # VOLUME SPIKE
    # =====================================================

    volume_spike = (
        volume_ratio >= 1.5
    )

    # =====================================================
    # OBV TREND
    # =====================================================

    current_obv = float(
        latest["obv"]
    )

    previous_obv = float(
        previous["obv"]
    )

    if current_obv > previous_obv:
        obv_trend = "RISING"
    elif current_obv < previous_obv:
        obv_trend = "FALLING"
    else:
        obv_trend = "FLAT"

    # =====================================================
    # VWAP
    # =====================================================

    vwap = float(
        latest["vwap"]
    )

    price = float(
        latest["close"]
    )

    if price > vwap:
        price_vwap = "ABOVE"
    elif price < vwap:
        price_vwap = "BELOW"
    else:
        price_vwap = "AT_VWAP"

    # =====================================================
    # VOLUME SIGNAL
    # =====================================================

    buy_score = 0
    sell_score = 0

    reasons = []

    if volume_spike:

        reasons.append(
            f"Volume Spike "
            f"({volume_ratio:.2f}x trung bình)"
        )

        if price > vwap:
            buy_score += 1
            reasons.append(
                "Giá trên VWAP"
            )

        elif price < vwap:
            sell_score += 1
            reasons.append(
                "Giá dưới VWAP"
            )

    if obv_trend == "RISING":

        buy_score += 1

        reasons.append(
            "OBV đang tăng"
        )

    elif obv_trend == "FALLING":

        sell_score += 1

        reasons.append(
            "OBV đang giảm"
        )

    if price > vwap:

        buy_score += 1

        reasons.append(
            "Giá nằm trên VWAP"
        )

    elif price < vwap:

        sell_score += 1

        reasons.append(
            "Giá nằm dưới VWAP"
        )

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    if buy_score > sell_score:

        signal = "BUY"

    elif sell_score > buy_score:

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    strength = abs(
        buy_score - sell_score
    )

    if not reasons:

        reasons.append(
            "Volume chưa cung cấp tín hiệu rõ."
        )

    return {
        "volume": round(
            volume,
            6
        ),

        "volume_sma20": round(
            volume_sma20,
            6
        ),

        "volume_ratio": round(
            volume_ratio,
            4
        ),

        "volume_spike": bool(
            volume_spike
        ),

        "obv": round(
            current_obv,
            6
        ),

        "obv_trend": obv_trend,

        "vwap": round(
            vwap,
            6
        ),

        "price_vs_vwap": price_vwap,

        "signal": signal,

        "strength": strength,

        "buy_score": buy_score,

        "sell_score": sell_score,

        "reason": " | ".join(
            reasons
        )
    }