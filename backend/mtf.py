TIMEFRAME_WEIGHTS = {
    "M1": 0.05,
    "M5": 0.10,
    "M15": 0.15,
    "H1": 0.25,
    "D1": 0.30,
    "W1": 0.15
}


def calculate_mtf_consensus(timeframe_results):
    """
    Tính consensus từ kết quả phân tích độc lập
    của từng timeframe.

    timeframe_results:

    {
        "M1": {
            "signal": "BUY",
            "confidence": 70
        },

        "M5": {
            "signal": "SELL",
            "confidence": 60
        }
    }
    """

    buy_score = 0.0
    sell_score = 0.0
    neutral_score = 0.0

    details = {}

    for timeframe in TIMEFRAME_WEIGHTS:

        result = timeframe_results.get(
            timeframe,
            {}
        )

        signal = result.get(
            "signal",
            "NEUTRAL"
        )

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        weight = TIMEFRAME_WEIGHTS[
            timeframe
        ]

        # Chuẩn hóa confidence
        confidence = max(
            0.0,
            min(
                confidence,
                100.0
            )
        )

        # Điểm tác động:
        # confidence × trọng số
        weighted_score = (
            confidence / 100.0
        ) * weight

        if signal == "BUY":

            buy_score += weighted_score

        elif signal == "SELL":

            sell_score += weighted_score

        else:

            neutral_score += weighted_score

        details[timeframe] = {

            "signal": signal,

            "confidence": round(
                confidence,
                2
            ),

            "weight": weight,

            "weighted_score": round(
                weighted_score,
                4
            )
        }

    # =====================================================
    # CONSENSUS
    # =====================================================

    if (
        buy_score > sell_score
        and
        buy_score > neutral_score
    ):

        consensus = "BUY"

    elif (
        sell_score > buy_score
        and
        sell_score > neutral_score
    ):

        consensus = "SELL"

    else:

        consensus = "NEUTRAL"

    # =====================================================
    # CONSENSUS CONFIDENCE
    # =====================================================

    total_score = (
        buy_score
        +
        sell_score
        +
        neutral_score
    )

    if total_score > 0:

        if consensus == "BUY":

            confidence = (
                buy_score /
                total_score
            ) * 100

        elif consensus == "SELL":

            confidence = (
                sell_score /
                total_score
            ) * 100

        else:

            confidence = (
                neutral_score /
                total_score
            ) * 100

    else:

        confidence = 0

    return {

        "consensus": consensus,

        "confidence": round(
            confidence,
            2
        ),

        "buy_score": round(
            buy_score,
            4
        ),

        "sell_score": round(
            sell_score,
            4
        ),

        "neutral_score": round(
            neutral_score,
            4
        ),

        "timeframes": details
    }