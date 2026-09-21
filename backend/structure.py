import pandas as pd


def find_swing_points(df, left=3, right=3):
    """
    Tìm các swing high / swing low cơ bản.
    """

    highs = []
    lows = []

    if len(df) < left + right + 1:
        return highs, lows

    for i in range(left, len(df) - right):

        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]

        left_highs = df["high"].iloc[
            i - left:i
        ]

        right_highs = df["high"].iloc[
            i + 1:i + right + 1
        ]

        left_lows = df["low"].iloc[
            i - left:i
        ]

        right_lows = df["low"].iloc[
            i + 1:i + right + 1
        ]

        # Swing High
        if (
            current_high > left_highs.max()
            and
            current_high > right_highs.max()
        ):
            highs.append({
                "index": i,
                "price": float(current_high)
            })

        # Swing Low
        if (
            current_low < left_lows.min()
            and
            current_low < right_lows.min()
        ):
            lows.append({
                "index": i,
                "price": float(current_low)
            })

    return highs, lows


def detect_structure(df):

    if df is None or len(df) < 30:

        return {
            "structure": "UNKNOWN",
            "trend": "NEUTRAL",
            "bos": False,
            "choch": False,
            "last_swing_high": None,
            "last_swing_low": None,
            "swing_highs": [],
            "swing_lows": []
        }

    data = df.reset_index(
        drop=True
    ).copy()

    swing_highs, swing_lows = find_swing_points(
        data,
        left=3,
        right=3
    )

    # =====================================================
    # KHÔNG ĐỦ SWING
    # =====================================================

    if len(swing_highs) < 2 or len(swing_lows) < 2:

        return {
            "structure": "RANGE",
            "trend": "NEUTRAL",
            "bos": False,
            "choch": False,
            "last_swing_high": (
                swing_highs[-1]["price"]
                if swing_highs
                else None
            ),
            "last_swing_low": (
                swing_lows[-1]["price"]
                if swing_lows
                else None
            ),
            "swing_highs": swing_highs[-5:],
            "swing_lows": swing_lows[-5:]
        }

    # =====================================================
    # 2 SWING HIGH GẦN NHẤT
    # =====================================================

    previous_high = swing_highs[-2]
    last_high = swing_highs[-1]

    # =====================================================
    # 2 SWING LOW GẦN NHẤT
    # =====================================================

    previous_low = swing_lows[-2]
    last_low = swing_lows[-1]

    # =====================================================
    # HH / LH
    # =====================================================

    if last_high["price"] > previous_high["price"]:

        high_structure = "HH"

    else:

        high_structure = "LH"

    # =====================================================
    # HL / LL
    # =====================================================

    if last_low["price"] > previous_low["price"]:

        low_structure = "HL"

    else:

        low_structure = "LL"

    # =====================================================
    # MARKET STRUCTURE
    # =====================================================

    if (
        high_structure == "HH"
        and
        low_structure == "HL"
    ):

        structure = "HH_HL"
        trend = "BULLISH"

    elif (
        high_structure == "LH"
        and
        low_structure == "LL"
    ):

        structure = "LH_LL"
        trend = "BEARISH"

    else:

        structure = (
            f"{high_structure}_{low_structure}"
        )

        trend = "NEUTRAL"

    # =====================================================
    # CURRENT PRICE
    # =====================================================

    current_price = float(
        data["close"].iloc[-1]
    )

    # =====================================================
    # BOS
    # =====================================================

    bullish_bos = (
        current_price
        >
        last_high["price"]
    )

    bearish_bos = (
        current_price
        <
        last_low["price"]
    )

    bos = (
        bullish_bos
        or
        bearish_bos
    )

    if bullish_bos:

        bos_direction = "BULLISH"

    elif bearish_bos:

        bos_direction = "BEARISH"

    else:

        bos_direction = "NONE"

    # =====================================================
    # CHoCH
    # =====================================================
    #
    # CHoCH được đánh dấu khi cấu trúc hiện tại
    # đi ngược xu hướng trước đó.
    #

    choch = False
    choch_direction = "NONE"

    if (
        high_structure == "HH"
        and
        low_structure == "LL"
    ):

        choch = True

        choch_direction = "BEARISH"

    elif (
        high_structure == "LH"
        and
        low_structure == "HL"
    ):

        choch = True

        choch_direction = "BULLISH"

    # =====================================================
    # OUTPUT
    # =====================================================

    return {

        "structure": structure,

        "trend": trend,

        "high_structure": high_structure,

        "low_structure": low_structure,

        "bos": bool(bos),

        "bos_direction": bos_direction,

        "choch": bool(choch),

        "choch_direction": choch_direction,

        "last_swing_high": round(
            last_high["price"],
            6
        ),

        "last_swing_low": round(
            last_low["price"],
            6
        ),

        "swing_highs": swing_highs[-5:],

        "swing_lows": swing_lows[-5:]
    }