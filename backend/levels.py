import numpy as np


def _cluster_levels(levels, tolerance):
    """
    Gom các mức giá gần nhau thành một vùng.
    """

    if not levels:
        return []

    levels = sorted(levels)

    clusters = []

    current = [levels[0]]

    for price in levels[1:]:

        average = sum(current) / len(current)

        if abs(price - average) <= tolerance:

            current.append(price)

        else:

            clusters.append(current)

            current = [price]

    clusters.append(current)

    zones = []

    for cluster in clusters:

        zones.append(
            sum(cluster) / len(cluster)
        )

    return zones


def detect_support_resistance(
    df,
    swing_highs=None,
    swing_lows=None
):

    if df is None or len(df) < 30:

        return {
            "support": [],
            "resistance": []
        }

    data = df.copy()

    current_price = float(
        data["close"].iloc[-1]
    )

    # =====================================================
    # TOLERANCE
    # =====================================================

    tolerance = (
        current_price * 0.005
    )

    # =====================================================
    # SWING LEVELS
    # =====================================================

    resistance_candidates = []
    support_candidates = []

    if swing_highs:

        resistance_candidates.extend(
            [
                float(item["price"])
                for item in swing_highs
            ]
        )

    if swing_lows:

        support_candidates.extend(
            [
                float(item["price"])
                for item in swing_lows
            ]
        )

    # =====================================================
    # RECENT HIGH / LOW
    # =====================================================

    resistance_candidates.extend(
        data["high"]
        .tail(50)
        .tolist()
    )

    support_candidates.extend(
        data["low"]
        .tail(50)
        .tolist()
    )

    # =====================================================
    # CLUSTER
    # =====================================================

    resistance_zones = _cluster_levels(
        resistance_candidates,
        tolerance
    )

    support_zones = _cluster_levels(
        support_candidates,
        tolerance
    )

    # =====================================================
    # KEEP RELEVANT LEVELS
    # =====================================================

    resistance_zones = sorted(
        [
            level
            for level in resistance_zones
            if level > current_price
        ]
    )

    support_zones = sorted(
        [
            level
            for level in support_zones
            if level < current_price
        ],
        reverse=True
    )

    # =====================================================
    # LIQUIDITY
    # =====================================================

    liquidity_highs = [
        level
        for level in resistance_zones
        if abs(
            level - current_price
        ) / current_price <= 0.03
    ]

    liquidity_lows = [
        level
        for level in support_zones
        if abs(
            level - current_price
        ) / current_price <= 0.03
    ]

    return {

        "support": [
            round(level, 6)
            for level in support_zones[:5]
        ],

        "resistance": [
            round(level, 6)
            for level in resistance_zones[:5]
        ],

        "liquidity_highs": [
            round(level, 6)
            for level in liquidity_highs[:5]
        ],

        "liquidity_lows": [
            round(level, 6)
            for level in liquidity_lows[:5]
        ]
    }