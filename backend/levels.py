from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


def _f(value: Any, default: float | None = None):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass

    return default


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data.columns = [
        str(column).lower()
        for column in data.columns
    ]

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
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
            "high",
            "low",
            "close",
        ]
    )

    return data.reset_index(drop=True)


def _atr(
    df: pd.DataFrame,
    period: int = 14,
) -> float:

    high = pd.to_numeric(
        df["high"],
        errors="coerce",
    )

    low = pd.to_numeric(
        df["low"],
        errors="coerce",
    )

    close = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
        .iloc[-1]
    )

    value = _f(atr)

    if value is not None and value > 0:
        return value

    price = _f(
        close.iloc[-1],
        1.0,
    )

    return max(
        price * 0.005,
        1e-9,
    )


def _cluster(
    values: list[float],
    tolerance: float,
) -> list[dict[str, Any]]:

    valid_values = sorted(
        [
            float(value)
            for value in values
            if _f(value) is not None
        ]
    )

    if not valid_values:
        return []

    groups: list[dict[str, Any]] = []

    for value in valid_values:

        if not groups:
            groups.append(
                {
                    "price": value,
                    "touches": 1,
                }
            )
            continue

        group = groups[-1]

        if abs(
            value - group["price"]
        ) > tolerance:

            groups.append(
                {
                    "price": value,
                    "touches": 1,
                }
            )

        else:

            old_touches = group["touches"]

            group["price"] = (
                (
                    group["price"]
                    * old_touches
                )
                + value
            ) / (
                old_touches + 1
            )

            group["touches"] = (
                old_touches + 1
            )

    return groups


def calculate_levels(
    df: pd.DataFrame,
) -> dict[str, Any]:

    data = _prepare_dataframe(df)

    empty_result = {
        "price": None,
        "atr": None,
        "support": [],
        "resistance": [],
        "zones": [],
        "fibonacci": [],
        "nearest_support": None,
        "nearest_resistance": None,
        "distance_to_support_atr": None,
        "distance_to_resistance_atr": None,
        "range_high": None,
        "range_low": None,
    }

    if len(data) < 30:
        return empty_result

    price = _f(
        data["close"].iloc[-1]
    )

    if price is None:
        return empty_result

    atr = _atr(data)

    lookback = min(
        len(data),
        200,
    )

    x = (
        data
        .tail(lookback)
        .reset_index(drop=True)
    )

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(
        2,
        len(x) - 2,
    ):

        high = _f(
            x["high"].iloc[i]
        )

        low = _f(
            x["low"].iloc[i]
        )

        if high is None or low is None:
            continue

        local_high = float(
            x["high"]
            .iloc[i - 2:i + 3]
            .max()
        )

        local_low = float(
            x["low"]
            .iloc[i - 2:i + 3]
            .min()
        )

        if high >= local_high:
            swing_highs.append(high)

        if low <= local_low:
            swing_lows.append(low)

    tolerance = max(
        atr * 0.45,
        price * 0.0015,
    )

    resistance_clusters = _cluster(
        swing_highs,
        tolerance,
    )

    support_clusters = _cluster(
        swing_lows,
        tolerance,
    )

    resistance = sorted(
        [
            group
            for group in resistance_clusters
            if group["price"] > price
        ],
        key=lambda group: group["price"],
    )[:5]

    support = sorted(
        [
            group
            for group in support_clusters
            if group["price"] < price
        ],
        key=lambda group: group["price"],
        reverse=True,
    )[:5]

    range_high = _f(
        x["high"].max()
    )

    range_low = _f(
        x["low"].min()
    )

    fibonacci = []

    if (
        range_high is not None
        and range_low is not None
        and range_high > range_low
    ):

        price_range = (
            range_high - range_low
        )

        for ratio in [
            0.236,
            0.382,
            0.500,
            0.618,
            0.786,
        ]:

            fibonacci.append(
                {
                    "ratio": ratio,
                    "price": (
                        range_high
                        - price_range * ratio
                    ),
                }
            )

    zones = []

    for group in support:

        zone_price = group["price"]

        strength = min(
            100,
            40 + group["touches"] * 12,
        )

        zones.append(
            {
                "type": "SUPPORT",
                "low": (
                    zone_price
                    - tolerance
                ),
                "high": (
                    zone_price
                    + tolerance
                ),
                "price": zone_price,
                "touches": group["touches"],
                "strength": strength,
            }
        )

    for group in resistance:

        zone_price = group["price"]

        strength = min(
            100,
            40 + group["touches"] * 12,
        )

        zones.append(
            {
                "type": "RESISTANCE",
                "low": (
                    zone_price
                    - tolerance
                ),
                "high": (
                    zone_price
                    + tolerance
                ),
                "price": zone_price,
                "touches": group["touches"],
                "strength": strength,
            }
        )

    support_zones = [
        zone
        for zone in zones
        if zone["type"] == "SUPPORT"
    ]

    resistance_zones = [
        zone
        for zone in zones
        if zone["type"] == "RESISTANCE"
    ]

    nearest_support = max(
        support_zones,
        key=lambda zone: zone["price"],
        default=None,
    )

    nearest_resistance = min(
        resistance_zones,
        key=lambda zone: zone["price"],
        default=None,
    )

    def distance_in_atr(zone):

        if zone is None:
            return None

        return (
            abs(
                price - zone["price"]
            )
            / max(
                atr,
                1e-9,
            )
        )

    return {
        "price": price,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "zones": zones,
        "fibonacci": fibonacci,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support_atr": distance_in_atr(
            nearest_support
        ),
        "distance_to_resistance_atr": distance_in_atr(
            nearest_resistance
        ),
        "range_high": range_high,
        "range_low": range_low,
    }


def detect_levels(
    df: pd.DataFrame,
) -> dict[str, Any]:

    return calculate_levels(df)