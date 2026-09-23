# backend/levels.py

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SWING_LOOKBACK = 3
RECENT_WINDOW = 50

MAX_LEVELS = 8

DEFAULT_TOLERANCE_PCT = 0.005


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
# SWING DETECTION
# ============================================================

def _find_swings(
    df: pd.DataFrame,
    lookback: int = SWING_LOOKBACK,
) -> Dict[str, List[float]]:

    highs: List[float] = []
    lows: List[float] = []

    if len(df) < (
        lookback * 2 + 1
    ):
        return {
            "highs": highs,
            "lows": lows,
        }

    high_values = df["high"].to_numpy(
        dtype=float
    )

    low_values = df["low"].to_numpy(
        dtype=float
    )

    for i in range(
        lookback,
        len(df) - lookback,
    ):

        current_high = high_values[i]
        current_low = low_values[i]

        if not (
            np.isfinite(current_high)
            and np.isfinite(current_low)
        ):
            continue

        left_high = high_values[
            i - lookback:i
        ]

        right_high = high_values[
            i + 1:i + lookback + 1
        ]

        left_low = low_values[
            i - lookback:i
        ]

        right_low = low_values[
            i + 1:i + lookback + 1
        ]

        if (
            current_high
            > np.max(left_high)
            and current_high
            >= np.max(right_high)
        ):
            highs.append(
                _safe_float(current_high)
            )

        if (
            current_low
            < np.min(left_low)
            and current_low
            <= np.min(right_low)
        ):
            lows.append(
                _safe_float(current_low)
            )

    return {
        "highs": highs,
        "lows": lows,
    }


# ============================================================
# LEVEL CLUSTERING
# ============================================================

def _cluster_levels(
    prices: List[float],
    tolerance: float,
) -> List[Dict[str, Any]]:

    if not prices:
        return []

    valid_prices = [
        _safe_float(price)
        for price in prices
        if _safe_float(price) > 0
    ]

    if not valid_prices:
        return []

    valid_prices.sort()

    clusters: List[List[float]] = []

    for price in valid_prices:

        placed = False

        for cluster in clusters:

            center = (
                sum(cluster)
                / len(cluster)
            )

            if abs(
                price - center
            ) <= tolerance:

                cluster.append(price)
                placed = True
                break

        if not placed:
            clusters.append(
                [price]
            )

    levels: List[Dict[str, Any]] = []

    for cluster in clusters:

        center = (
            sum(cluster)
            / len(cluster)
        )

        strength = len(cluster)

        levels.append(
            {
                "price": round(
                    center,
                    8,
                ),
                "touches": strength,
                "strength": strength,
            }
        )

    levels.sort(
        key=lambda x: (
            x["strength"],
            x["price"],
        ),
        reverse=True,
    )

    return levels


# ============================================================
# LEVEL DISTANCE
# ============================================================

def _distance_pct(
    price: float,
    level: float,
) -> float:

    if price <= 0:
        return 0.0

    return abs(
        price - level
    ) / price * 100.0


# ============================================================
# MAIN LEVEL ANALYZER
# ============================================================

def detect_levels(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    neutral_result = {
        "support": None,
        "resistance": None,

        "supports": [],
        "resistances": [],

        "liquidity_highs": [],
        "liquidity_lows": [],

        "nearest_support": None,
        "nearest_resistance": None,

        "support_distance_pct": 0.0,
        "resistance_distance_pct": 0.0,

        "price": 0.0,

        "analysis_only": True,
    }

    try:

        df = _prepare_dataframe(data)

        if len(df) < 20:
            return neutral_result

        current_price = _safe_float(
            df["close"].iloc[-1]
        )

        if current_price <= 0:
            return neutral_result

        # ----------------------------------------------------
        # Dynamic tolerance
        # ----------------------------------------------------

        tolerance = (
            current_price
            * DEFAULT_TOLERANCE_PCT
        )

        # ----------------------------------------------------
        # Swings
        # ----------------------------------------------------

        swings = _find_swings(
            df,
            SWING_LOOKBACK,
        )

        swing_highs = swings["highs"]
        swing_lows = swings["lows"]

        # ----------------------------------------------------
        # Recent extreme levels
        # ----------------------------------------------------

        recent = df.tail(
            RECENT_WINDOW
        )

        recent_high = _safe_float(
            recent["high"].max()
        )

        recent_low = _safe_float(
            recent["low"].min()
        )

        high_candidates = (
            swing_highs
            + [recent_high]
        )

        low_candidates = (
            swing_lows
            + [recent_low]
        )

        # ----------------------------------------------------
        # Cluster
        # ----------------------------------------------------

        resistance_levels = (
            _cluster_levels(
                high_candidates,
                tolerance,
            )
        )

        support_levels = (
            _cluster_levels(
                low_candidates,
                tolerance,
            )
        )

        # ----------------------------------------------------
        # Split levels around current price
        # ----------------------------------------------------

        resistances = [
            level
            for level in resistance_levels
            if level["price"]
            >= current_price
        ]

        supports = [
            level
            for level in support_levels
            if level["price"]
            <= current_price
        ]

        # ----------------------------------------------------
        # Nearest support
        # ----------------------------------------------------

        supports.sort(
            key=lambda x: (
                current_price
                - x["price"]
            )
        )

        resistances.sort(
            key=lambda x: (
                x["price"]
                - current_price
            )
        )

        nearest_support = (
            supports[0]
            if supports
            else None
        )

        nearest_resistance = (
            resistances[0]
            if resistances
            else None
        )

        # ----------------------------------------------------
        # Liquidity pools
        # ----------------------------------------------------

        # Liquidity above price:
        # recent/swing highs.

        liquidity_highs = sorted(
            [
                price
                for price in high_candidates
                if price > current_price
            ]
        )

        # Liquidity below price:
        # recent/swing lows.

        liquidity_lows = sorted(
            [
                price
                for price in low_candidates
                if price < current_price
            ],
            reverse=True,
        )

        # Remove duplicates within tolerance
        def _deduplicate(
            values: List[float],
        ) -> List[float]:

            result: List[float] = []

            for value in values:

                if not result:

                    result.append(value)
                    continue

                if abs(
                    value
                    - result[-1]
                ) > tolerance:

                    result.append(value)

            return result

        liquidity_highs = _deduplicate(
            liquidity_highs
        )[:MAX_LEVELS]

        liquidity_lows = _deduplicate(
            liquidity_lows
        )[:MAX_LEVELS]

        # ----------------------------------------------------
        # Distances
        # ----------------------------------------------------

        support_distance_pct = 0.0

        if nearest_support:

            support_distance_pct = (
                _distance_pct(
                    current_price,
                    nearest_support["price"],
                )
            )

        resistance_distance_pct = 0.0

        if nearest_resistance:

            resistance_distance_pct = (
                _distance_pct(
                    current_price,
                    nearest_resistance["price"],
                )
            )

        # ----------------------------------------------------
        # Output levels
        # ----------------------------------------------------

        supports = supports[:MAX_LEVELS]
        resistances = resistances[:MAX_LEVELS]

        return {
            "support": (
                nearest_support["price"]
                if nearest_support
                else None
            ),

            "resistance": (
                nearest_resistance["price"]
                if nearest_resistance
                else None
            ),

            "supports": supports,

            "resistances": resistances,

            "liquidity_highs": [
                round(
                    price,
                    8,
                )
                for price
                in liquidity_highs
            ],

            "liquidity_lows": [
                round(
                    price,
                    8,
                )
                for price
                in liquidity_lows
            ],

            "nearest_support": (
                nearest_support
            ),

            "nearest_resistance": (
                nearest_resistance
            ),

            "support_distance_pct": round(
                support_distance_pct,
                4,
            ),

            "resistance_distance_pct": round(
                resistance_distance_pct,
                4,
            ),

            "price": round(
                current_price,
                8,
            ),

            "analysis_only": True,
        }

    except Exception:
        return neutral_result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def analyze_levels(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_levels(data)


def get_support_resistance(
    data: pd.DataFrame,
) -> Dict[str, Any]:

    return detect_levels(data)


__all__ = [
    "detect_levels",
    "analyze_levels",
    "get_support_resistance",
]