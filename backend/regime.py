# backend/regime.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

EMA_FAST = 20
EMA_SLOW = 50

ATR_PERIOD = 14

LOW_VOLATILITY_ATR_PCT = 0.50
HIGH_VOLATILITY_ATR_PCT = 3.00

TREND_DISTANCE_PCT = 0.20


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

    if "close" not in df.columns:
        return pd.DataFrame()

    if "high" not in df.columns:
        return pd.DataFrame()

    if "low" not in df.columns:
        return pd.DataFrame()

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:

        if column in df.columns:

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
            "high",
            "low",
            "close",
        ]
    )

    return df.reset_index(drop=True)


# ============================================================
# EMA
# ============================================================

def _ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# ATR
# ============================================================

def _atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================
# TREND DETECTION
# ============================================================

def _detect_trend(
    close: float,
    ema_fast: float,
    ema_slow: float,
) -> str:

    if (
        close <= 0
        or ema_fast <= 0
        or ema_slow <= 0
    ):
        return "NEUTRAL"

    distance = (
        abs(ema_fast - ema_slow)
        / ema_slow
        * 100.0
    )

    if distance < TREND_DISTANCE_PCT:
        return "RANGING"

    if (
        close > ema_fast
        and ema_fast > ema_slow
    ):
        return "BULLISH"

    if (
        close < ema_fast
        and ema_fast < ema_slow
    ):
        return "BEARISH"

    return "RANGING"


# ============================================================
# VOLATILITY
# ============================================================

def _detect_volatility(
    atr_pct: float,
) -> str:

    if atr_pct >= HIGH_VOLATILITY_ATR_PCT:
        return "HIGH"

    if atr_pct <= LOW_VOLATILITY_ATR_PCT:
        return "LOW"

    return "NORMAL"


# ============================================================
# MAIN REGIME ANALYZER
# ============================================================

def detect_regime(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    neutral_result = {
        "regime": "NEUTRAL",
        "trend": "NEUTRAL",
        "volatility": "NORMAL",

        "score": 0.0,

        "close": 0.0,
        "ema20": 0.0,
        "ema50": 0.0,

        "atr": 0.0,
        "atr_pct": 0.0,

        "trend_strength": 0.0,
        "volatility_score": 0.0,

        "trending": False,
        "ranging": False,

        "high_volatility": False,
        "low_volatility": False,

        "analysis_only": True,
    }

    try:

        df = _prepare_dataframe(data)

        minimum_length = max(
            EMA_SLOW,
            ATR_PERIOD,
        ) + 5

        if len(df) < minimum_length:
            return neutral_result

        close_series = df["close"]

        ema20_series = _ema(
            close_series,
            EMA_FAST,
        )

        ema50_series = _ema(
            close_series,
            EMA_SLOW,
        )

        atr_series = _atr(
            df,
            ATR_PERIOD,
        )

        close = _safe_float(
            close_series.iloc[-1]
        )

        ema20 = _safe_float(
            ema20_series.iloc[-1]
        )

        ema50 = _safe_float(
            ema50_series.iloc[-1]
        )

        atr = _safe_float(
            atr_series.iloc[-1]
        )

        if close <= 0:
            return neutral_result

        atr_pct = (
            atr
            / close
            * 100.0
        )

        trend = _detect_trend(
            close,
            ema20,
            ema50,
        )

        volatility = _detect_volatility(
            atr_pct
        )

        # ----------------------------------------------------
        # Trend strength
        # ----------------------------------------------------

        if ema50 > 0:

            trend_strength = (
                abs(ema20 - ema50)
                / ema50
                * 100.0
            )

        else:
            trend_strength = 0.0

        trend_strength = min(
            trend_strength,
            100.0,
        )

        # ----------------------------------------------------
        # Volatility score
        # ----------------------------------------------------

        volatility_score = min(
            atr_pct / 5.0 * 100.0,
            100.0,
        )

        # ----------------------------------------------------
        # Regime classification
        # ----------------------------------------------------

        if trend == "BULLISH":

            if volatility == "HIGH":
                regime = "TRENDING_BULLISH_HIGH_VOLATILITY"

            elif volatility == "LOW":
                regime = "TRENDING_BULLISH_LOW_VOLATILITY"

            else:
                regime = "TRENDING_BULLISH"

        elif trend == "BEARISH":

            if volatility == "HIGH":
                regime = "TRENDING_BEARISH_HIGH_VOLATILITY"

            elif volatility == "LOW":
                regime = "TRENDING_BEARISH_LOW_VOLATILITY"

            else:
                regime = "TRENDING_BEARISH"

        elif trend == "RANGING":

            if volatility == "HIGH":
                regime = "RANGING_HIGH_VOLATILITY"

            elif volatility == "LOW":
                regime = "RANGING_LOW_VOLATILITY"

            else:
                regime = "RANGING"

        else:
            regime = "NEUTRAL"

        # ----------------------------------------------------
        # Directional score
        # ----------------------------------------------------

        directional_score = 0.0

        if trend == "BULLISH":
            directional_score = trend_strength

        elif trend == "BEARISH":
            directional_score = -trend_strength

        # ----------------------------------------------------
        # Final regime score
        # ----------------------------------------------------

        score = (
            directional_score * 0.70
            + volatility_score * 0.30
        )

        score = max(
            -100.0,
            min(
                100.0,
                score,
            ),
        )

        return {
            "regime": regime,

            "trend": trend,

            "volatility": volatility,

            "score": round(
                score,
                2,
            ),

            "close": round(
                close,
                8,
            ),

            "ema20": round(
                ema20,
                8,
            ),

            "ema50": round(
                ema50,
                8,
            ),

            "atr": round(
                atr,
                8,
            ),

            "atr_pct": round(
                atr_pct,
                4,
            ),

            "trend_strength": round(
                trend_strength,
                2,
            ),

            "volatility_score": round(
                volatility_score,
                2,
            ),

            "trending": trend in {
                "BULLISH",
                "BEARISH",
            },

            "ranging": trend == "RANGING",

            "high_volatility": (
                volatility == "HIGH"
            ),

            "low_volatility": (
                volatility == "LOW"
            ),

            "analysis_only": True,
        }

    except Exception:
        return neutral_result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def analyze_regime(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_regime(data)


def get_market_regime(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_regime(data)


__all__ = [
    "detect_regime",
    "analyze_regime",
    "get_market_regime",
]