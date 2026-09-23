# backend/volume.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

VOLUME_SMA_PERIOD = 20
OBV_SMA_PERIOD = 20
SPIKE_RATIO = 1.8
STRONG_VOLUME_RATIO = 1.3


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def _prepare_dataframe(
    data: pd.DataFrame,
) -> pd.DataFrame:

    if data is None:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame(data)

    df = data.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if any(
        column not in df.columns
        for column in required
    ):
        return pd.DataFrame()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required
    )

    return df.reset_index(drop=True)


# ============================================================
# OBV
# ============================================================

def _calculate_obv(
    df: pd.DataFrame,
) -> pd.Series:

    close = df["close"]
    volume = df["volume"]

    direction = np.sign(
        close.diff()
    ).fillna(0.0)

    obv = (
        direction
        * volume
    ).cumsum()

    return obv


# ============================================================
# VWAP
# ============================================================

def _calculate_vwap(
    df: pd.DataFrame,
) -> pd.Series:

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    cumulative_volume = (
        df["volume"]
        .cumsum()
    )

    cumulative_pv = (
        typical_price
        * df["volume"]
    ).cumsum()

    vwap = (
        cumulative_pv
        / cumulative_volume.replace(
            0,
            np.nan,
        )
    )

    return vwap


# ============================================================
# MAIN VOLUME ANALYZER
# ============================================================

def analyze_volume(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    neutral_result = {
        "signal": "NEUTRAL",
        "direction": "NEUTRAL",

        "volume": 0.0,
        "volume_sma20": 0.0,
        "volume_ratio": 0.0,

        "volume_spike": False,
        "strong_volume": False,

        "obv": 0.0,
        "obv_sma20": 0.0,
        "obv_trend": "NEUTRAL",

        "vwap": 0.0,
        "price_vs_vwap": "NEUTRAL",

        "buy_score": 0.0,
        "sell_score": 0.0,

        "strength": 0.0,

        "analysis_only": True,
    }

    try:

        df = _prepare_dataframe(data)

        minimum_length = (
            VOLUME_SMA_PERIOD + 5
        )

        if len(df) < minimum_length:
            return neutral_result

        # ----------------------------------------------------
        # Volume SMA
        # ----------------------------------------------------

        volume_sma = (
            df["volume"]
            .rolling(
                VOLUME_SMA_PERIOD,
                min_periods=VOLUME_SMA_PERIOD,
            )
            .mean()
        )

        # ----------------------------------------------------
        # OBV
        # ----------------------------------------------------

        obv = _calculate_obv(df)

        obv_sma = (
            obv
            .rolling(
                OBV_SMA_PERIOD,
                min_periods=OBV_SMA_PERIOD,
            )
            .mean()
        )

        # ----------------------------------------------------
        # VWAP
        # ----------------------------------------------------

        vwap = _calculate_vwap(df)

        # ----------------------------------------------------
        # Latest values
        # ----------------------------------------------------

        close = _safe_float(
            df["close"].iloc[-1]
        )

        current_volume = _safe_float(
            df["volume"].iloc[-1]
        )

        current_volume_sma = _safe_float(
            volume_sma.iloc[-1]
        )

        current_obv = _safe_float(
            obv.iloc[-1]
        )

        current_obv_sma = _safe_float(
            obv_sma.iloc[-1]
        )

        current_vwap = _safe_float(
            vwap.iloc[-1]
        )

        # ----------------------------------------------------
        # Volume ratio
        # ----------------------------------------------------

        if current_volume_sma > 0:

            volume_ratio = (
                current_volume
                / current_volume_sma
            )

        else:
            volume_ratio = 0.0

        volume_ratio = _safe_float(
            volume_ratio
        )

        volume_spike = (
            volume_ratio
            >= SPIKE_RATIO
        )

        strong_volume = (
            volume_ratio
            >= STRONG_VOLUME_RATIO
        )

        # ----------------------------------------------------
        # OBV trend
        # ----------------------------------------------------

        if current_obv > current_obv_sma:
            obv_trend = "BULLISH"

        elif current_obv < current_obv_sma:
            obv_trend = "BEARISH"

        else:
            obv_trend = "NEUTRAL"

        # ----------------------------------------------------
        # Price vs VWAP
        # ----------------------------------------------------

        if (
            current_vwap > 0
            and close > current_vwap
        ):
            price_vs_vwap = "ABOVE"

        elif (
            current_vwap > 0
            and close < current_vwap
        ):
            price_vs_vwap = "BELOW"

        else:
            price_vs_vwap = "NEUTRAL"

        # ----------------------------------------------------
        # Directional scoring
        # ----------------------------------------------------

        buy_score = 0.0
        sell_score = 0.0

        # OBV
        if obv_trend == "BULLISH":
            buy_score += 2.0

        elif obv_trend == "BEARISH":
            sell_score += 2.0

        # VWAP
        if price_vs_vwap == "ABOVE":
            buy_score += 1.5

        elif price_vs_vwap == "BELOW":
            sell_score += 1.5

        # Strong volume confirms direction
        if strong_volume:

            if close > current_vwap:
                buy_score += 1.0

            elif close < current_vwap:
                sell_score += 1.0

        # Volume spike
        if volume_spike:

            if obv_trend == "BULLISH":
                buy_score += 1.0

            elif obv_trend == "BEARISH":
                sell_score += 1.0

        # ----------------------------------------------------
        # Final signal
        # ----------------------------------------------------

        if buy_score > sell_score:
            signal = "BUY"
            direction = "BULLISH"

        elif sell_score > buy_score:
            signal = "SELL"
            direction = "BEARISH"

        else:
            signal = "NEUTRAL"
            direction = "NEUTRAL"

        total_score = (
            buy_score
            + sell_score
        )

        if total_score > 0:

            strength = (
                max(
                    buy_score,
                    sell_score,
                )
                / total_score
                * 100.0
            )

        else:
            strength = 0.0

        return {
            "signal": signal,
            "direction": direction,

            "volume": round(
                current_volume,
                8,
            ),

            "volume_sma20": round(
                current_volume_sma,
                8,
            ),

            "volume_ratio": round(
                volume_ratio,
                4,
            ),

            "volume_spike": volume_spike,
            "strong_volume": strong_volume,

            "obv": round(
                current_obv,
                8,
            ),

            "obv_sma20": round(
                current_obv_sma,
                8,
            ),

            "obv_trend": obv_trend,

            "vwap": round(
                current_vwap,
                8,
            ),

            "price_vs_vwap": price_vs_vwap,

            "buy_score": round(
                buy_score,
                2,
            ),

            "sell_score": round(
                sell_score,
                2,
            ),

            "strength": round(
                strength,
                2,
            ),

            "analysis_only": True,
        }

    except Exception:
        return neutral_result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def detect_volume(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return analyze_volume(data)


def get_volume_analysis(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return analyze_volume(data)


__all__ = [
    "analyze_volume",
    "detect_volume",
    "get_volume_analysis",
]