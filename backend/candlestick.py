# backend/candlestick.py

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

EPSILON = 1e-12


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def _prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
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
# CANDLE GEOMETRY
# ============================================================

def _candle_metrics(row: pd.Series) -> Dict[str, float]:

    open_price = _safe_float(row["open"])
    high = _safe_float(row["high"])
    low = _safe_float(row["low"])
    close = _safe_float(row["close"])

    body = abs(close - open_price)

    candle_range = max(
        high - low,
        EPSILON,
    )

    upper_wick = max(
        high - max(open_price, close),
        0.0,
    )

    lower_wick = max(
        min(open_price, close) - low,
        0.0,
    )

    body_ratio = body / candle_range

    upper_ratio = upper_wick / candle_range
    lower_ratio = lower_wick / candle_range

    bullish = close > open_price
    bearish = close < open_price
    doji = body_ratio <= 0.10

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "body": body,
        "range": candle_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body_ratio": body_ratio,
        "upper_ratio": upper_ratio,
        "lower_ratio": lower_ratio,
        "bullish": float(bullish),
        "bearish": float(bearish),
        "doji": float(doji),
    }


def _direction(
    metrics: Dict[str, float],
) -> str:

    if metrics["bullish"]:
        return "BULLISH"

    if metrics["bearish"]:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# SINGLE CANDLE PATTERNS
# ============================================================

def _detect_doji(
    current: Dict[str, float],
) -> bool:

    return (
        current["body_ratio"] <= 0.10
    )


def _detect_hammer(
    current: Dict[str, float],
) -> bool:

    return (
        current["lower_wick"]
        >= current["body"] * 2.0
        and current["upper_wick"]
        <= max(
            current["body"] * 0.75,
            EPSILON,
        )
        and current["body_ratio"] <= 0.45
    )


def _detect_inverted_hammer(
    current: Dict[str, float],
) -> bool:

    return (
        current["upper_wick"]
        >= current["body"] * 2.0
        and current["lower_wick"]
        <= max(
            current["body"] * 0.75,
            EPSILON,
        )
        and current["body_ratio"] <= 0.45
    )


def _detect_shooting_star(
    current: Dict[str, float],
) -> bool:

    return (
        current["upper_wick"]
        >= current["body"] * 2.0
        and current["lower_wick"]
        <= max(
            current["body"] * 0.75,
            EPSILON,
        )
        and current["body_ratio"] <= 0.45
    )


def _detect_marubozu(
    current: Dict[str, float],
) -> bool:

    return (
        current["body_ratio"] >= 0.85
        and current["upper_ratio"] <= 0.08
        and current["lower_ratio"] <= 0.08
    )


# ============================================================
# TWO-CANDLE PATTERNS
# ============================================================

def _detect_bullish_engulfing(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> bool:

    return (
        previous["bearish"]
        and current["bullish"]
        and current["open"]
        <= previous["close"]
        and current["close"]
        >= previous["open"]
        and current["body"]
        > previous["body"]
    )


def _detect_bearish_engulfing(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> bool:

    return (
        previous["bullish"]
        and current["bearish"]
        and current["open"]
        >= previous["close"]
        and current["close"]
        <= previous["open"]
        and current["body"]
        > previous["body"]
    )


def _detect_piercing(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> bool:

    if not (
        previous["bearish"]
        and current["bullish"]
    ):
        return False

    previous_mid = (
        previous["open"]
        + previous["close"]
    ) / 2.0

    return (
        current["open"]
        < previous["close"]
        and current["close"]
        > previous_mid
        and current["close"]
        < previous["open"]
    )


def _detect_dark_cloud(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> bool:

    if not (
        previous["bullish"]
        and current["bearish"]
    ):
        return False

    previous_mid = (
        previous["open"]
        + previous["close"]
    ) / 2.0

    return (
        current["open"]
        > previous["close"]
        and current["close"]
        < previous_mid
        and current["close"]
        > previous["open"]
    )


# ============================================================
# THREE-CANDLE PATTERNS
# ============================================================

def _detect_morning_star(
    first: Dict[str, float],
    second: Dict[str, float],
    third: Dict[str, float],
) -> bool:

    if not (
        first["bearish"]
        and third["bullish"]
    ):
        return False

    small_middle = (
        second["body_ratio"] <= 0.35
    )

    first_mid = (
        first["open"]
        + first["close"]
    ) / 2.0

    return (
        small_middle
        and third["close"] > first_mid
    )


def _detect_evening_star(
    first: Dict[str, float],
    second: Dict[str, float],
    third: Dict[str, float],
) -> bool:

    if not (
        first["bullish"]
        and third["bearish"]
    ):
        return False

    small_middle = (
        second["body_ratio"] <= 0.35
    )

    first_mid = (
        first["open"]
        + first["close"]
    ) / 2.0

    return (
        small_middle
        and third["close"] < first_mid
    )


# ============================================================
# INSIDE BAR
# ============================================================

def _detect_inside_bar(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> bool:

    return (
        current["high"]
        <= previous["high"]
        and current["low"]
        >= previous["low"]
    )


# ============================================================
# THREE SOLDIERS / THREE CROWS
# ============================================================

def _detect_three_white_soldiers(
    candles: List[Dict[str, float]],
) -> bool:

    if len(candles) < 3:
        return False

    a, b, c = candles[-3:]

    return (
        a["bullish"]
        and b["bullish"]
        and c["bullish"]
        and b["close"] > a["close"]
        and c["close"] > b["close"]
        and b["open"] >= a["open"]
        and c["open"] >= b["open"]
        and a["body_ratio"] >= 0.50
        and b["body_ratio"] >= 0.50
        and c["body_ratio"] >= 0.50
    )


def _detect_three_black_crows(
    candles: List[Dict[str, float]],
) -> bool:

    if len(candles) < 3:
        return False

    a, b, c = candles[-3:]

    return (
        a["bearish"]
        and b["bearish"]
        and c["bearish"]
        and b["close"] < a["close"]
        and c["close"] < b["close"]
        and b["open"] <= a["open"]
        and c["open"] <= b["open"]
        and a["body_ratio"] >= 0.50
        and b["body_ratio"] >= 0.50
        and c["body_ratio"] >= 0.50
    )


# ============================================================
# PATTERN SCORING
# ============================================================

PATTERN_SCORES = {
    "THREE_WHITE_SOLDIERS": 3.0,
    "THREE_BLACK_CROWS": 3.0,

    "MORNING_STAR": 3.0,
    "EVENING_STAR": 3.0,

    "BULLISH_ENGULFING": 2.5,
    "BEARISH_ENGULFING": 2.5,

    "PIERCING": 2.0,
    "DARK_CLOUD": 2.0,

    "HAMMER": 1.5,
    "SHOOTING_STAR": 1.5,
    "INVERTED_HAMMER": 1.2,

    "MARUBOZU": 1.0,
    "DOJI": 0.5,
    "INSIDE_BAR": 0.5,
}


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_candle(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    neutral_result = {
        "pattern": "NONE",
        "signal": "NEUTRAL",
        "direction": "NEUTRAL",
        "strength": 0.0,

        "patterns": [],
        "bullish_patterns": [],
        "bearish_patterns": [],

        "body_ratio": 0.0,
        "upper_wick_ratio": 0.0,
        "lower_wick_ratio": 0.0,

        "analysis_only": True,
    }

    try:

        df = _prepare_dataframe(data)

        if len(df) < 3:
            return neutral_result

        # ----------------------------------------------------
        # Convert latest candles
        # ----------------------------------------------------

        metrics = [
            _candle_metrics(df.iloc[i])
            for i in range(len(df))
        ]

        current = metrics[-1]
        previous = metrics[-2]
        first = metrics[-3]

        detected: List[Dict[str, Any]] = []

        def add_pattern(
            name: str,
            direction: str,
            strength: float | None = None,
        ):

            score = (
                strength
                if strength is not None
                else PATTERN_SCORES.get(
                    name,
                    1.0,
                )
            )

            detected.append(
                {
                    "name": name,
                    "direction": direction,
                    "strength": float(score),
                }
            )

        # ----------------------------------------------------
        # Single candle
        # ----------------------------------------------------

        if _detect_doji(current):

            add_pattern(
                "DOJI",
                "NEUTRAL",
            )

        if _detect_hammer(current):

            add_pattern(
                "HAMMER",
                "BULLISH",
            )

        if _detect_inverted_hammer(current):

            add_pattern(
                "INVERTED_HAMMER",
                "BULLISH",
            )

        if _detect_shooting_star(current):

            add_pattern(
                "SHOOTING_STAR",
                "BEARISH",
            )

        if _detect_marubozu(current):

            if current["bullish"]:
                add_pattern(
                    "MARUBOZU",
                    "BULLISH",
                )

            elif current["bearish"]:
                add_pattern(
                    "MARUBOZU",
                    "BEARISH",
                )

        # ----------------------------------------------------
        # Two candles
        # ----------------------------------------------------

        if _detect_bullish_engulfing(
            previous,
            current,
        ):

            add_pattern(
                "BULLISH_ENGULFING",
                "BULLISH",
            )

        if _detect_bearish_engulfing(
            previous,
            current,
        ):

            add_pattern(
                "BEARISH_ENGULFING",
                "BEARISH",
            )

        if _detect_piercing(
            previous,
            current,
        ):

            add_pattern(
                "PIERCING",
                "BULLISH",
            )

        if _detect_dark_cloud(
            previous,
            current,
        ):

            add_pattern(
                "DARK_CLOUD",
                "BEARISH",
            )

        if _detect_inside_bar(
            previous,
            current,
        ):

            add_pattern(
                "INSIDE_BAR",
                "NEUTRAL",
            )

        # ----------------------------------------------------
        # Three candles
        # ----------------------------------------------------

        if _detect_morning_star(
            first,
            previous,
            current,
        ):

            add_pattern(
                "MORNING_STAR",
                "BULLISH",
            )

        if _detect_evening_star(
            first,
            previous,
            current,
        ):

            add_pattern(
                "EVENING_STAR",
                "BEARISH",
            )

        recent_three = metrics[-3:]

        if _detect_three_white_soldiers(
            recent_three,
        ):

            add_pattern(
                "THREE_WHITE_SOLDIERS",
                "BULLISH",
            )

        if _detect_three_black_crows(
            recent_three,
        ):

            add_pattern(
                "THREE_BLACK_CROWS",
                "BEARISH",
            )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        bullish_score = sum(
            item["strength"]
            for item in detected
            if item["direction"] == "BULLISH"
        )

        bearish_score = sum(
            item["strength"]
            for item in detected
            if item["direction"] == "BEARISH"
        )

        total_score = (
            bullish_score
            + bearish_score
        )

        if bullish_score > bearish_score:
            signal = "BUY"
            direction = "BULLISH"

        elif bearish_score > bullish_score:
            signal = "SELL"
            direction = "BEARISH"

        else:
            signal = "NEUTRAL"
            direction = "NEUTRAL"

        if total_score > 0:
            strength = (
                max(
                    bullish_score,
                    bearish_score,
                )
                / total_score
                * 100.0
            )
        else:
            strength = 0.0

        # ----------------------------------------------------
        # Pattern priority
        # ----------------------------------------------------

        priority = {
            "THREE_WHITE_SOLDIERS": 100,
            "THREE_BLACK_CROWS": 100,

            "MORNING_STAR": 95,
            "EVENING_STAR": 95,

            "BULLISH_ENGULFING": 90,
            "BEARISH_ENGULFING": 90,

            "PIERCING": 80,
            "DARK_CLOUD": 80,

            "HAMMER": 70,
            "SHOOTING_STAR": 70,
            "INVERTED_HAMMER": 65,

            "MARUBOZU": 50,
            "INSIDE_BAR": 30,
            "DOJI": 20,
        }

        detected.sort(
            key=lambda item: (
                priority.get(
                    item["name"],
                    0,
                ),
                item["strength"],
            ),
            reverse=True,
        )

        primary_pattern = (
            detected[0]["name"]
            if detected
            else "NONE"
        )

        bullish_patterns = [
            item["name"]
            for item in detected
            if item["direction"] == "BULLISH"
        ]

        bearish_patterns = [
            item["name"]
            for item in detected
            if item["direction"] == "BEARISH"
        ]

        return {
            "pattern": primary_pattern,
            "signal": signal,
            "direction": direction,
            "strength": round(
                float(strength),
                2,
            ),

            "patterns": detected,

            "bullish_patterns": bullish_patterns,
            "bearish_patterns": bearish_patterns,

            "body_ratio": round(
                current["body_ratio"],
                4,
            ),

            "upper_wick_ratio": round(
                current["upper_ratio"],
                4,
            ),

            "lower_wick_ratio": round(
                current["lower_ratio"],
                4,
            ),

            "analysis_only": True,
        }

    except Exception:
        return neutral_result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def detect_candlestick(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_candle(data)


def analyze_candlestick(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_candle(data)


__all__ = [
    "detect_candle",
    "detect_candlestick",
    "analyze_candlestick",
]