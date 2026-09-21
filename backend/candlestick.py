def detect_candle(df):
    """
    Phân tích mô hình nến.

    Bao gồm:
    - Doji
    - Hammer
    - Shooting Star
    - Bullish Engulfing
    - Bearish Engulfing
    - Morning Star
    - Evening Star
    - Inside Bar
    - Marubozu
    - Bullish / Bearish Candle
    """

    if df is None or len(df) < 3:

        return {
            "pattern": "UNKNOWN",
            "signal": "NEUTRAL",
            "strength": 0
        }

    current = df.iloc[-1]
    previous = df.iloc[-2]
    previous2 = df.iloc[-3]

    # =====================================================
    # CURRENT CANDLE
    # =====================================================

    open_price = float(
        current["open"]
    )

    close = float(
        current["close"]
    )

    high = float(
        current["high"]
    )

    low = float(
        current["low"]
    )

    body = abs(
        close - open_price
    )

    candle_range = high - low

    if candle_range <= 0:

        return {
            "pattern": "FLAT",
            "signal": "NEUTRAL",
            "strength": 0
        }

    upper_wick = (
        high
        -
        max(
            open_price,
            close
        )
    )

    lower_wick = (
        min(
            open_price,
            close
        )
        -
        low
    )

    body_ratio = (
        body
        /
        candle_range
    )

    # =====================================================
    # PREVIOUS CANDLE
    # =====================================================

    prev_open = float(
        previous["open"]
    )

    prev_close = float(
        previous["close"]
    )

    prev_high = float(
        previous["high"]
    )

    prev_low = float(
        previous["low"]
    )

    prev_body = abs(
        prev_close - prev_open
    )

    # =====================================================
    # 1. DOJI
    # =====================================================

    if body_ratio < 0.10:

        return {
            "pattern": "DOJI",
            "signal": "NEUTRAL",
            "strength": 1
        }

    # =====================================================
    # 2. HAMMER
    # =====================================================

    if (
        lower_wick > body * 2
        and
        upper_wick < body
        and
        close >= open_price
    ):

        return {
            "pattern": "HAMMER",
            "signal": "BUY",
            "strength": 2
        }

    # =====================================================
    # 3. SHOOTING STAR
    # =====================================================

    if (
        upper_wick > body * 2
        and
        lower_wick < body
        and
        close <= open_price
    ):

        return {
            "pattern": "SHOOTING_STAR",
            "signal": "SELL",
            "strength": 2
        }

    # =====================================================
    # 4. BULLISH ENGULFING
    # =====================================================

    previous_bearish = (
        prev_close < prev_open
    )

    current_bullish = (
        close > open_price
    )

    bullish_engulfing = (
        previous_bearish
        and
        current_bullish
        and
        open_price <= prev_close
        and
        close >= prev_open
    )

    if bullish_engulfing:

        return {
            "pattern": "BULLISH_ENGULFING",
            "signal": "BUY",
            "strength": 3
        }

    # =====================================================
    # 5. BEARISH ENGULFING
    # =====================================================

    previous_bullish = (
        prev_close > prev_open
    )

    current_bearish = (
        close < open_price
    )

    bearish_engulfing = (
        previous_bullish
        and
        current_bearish
        and
        open_price >= prev_close
        and
        close <= prev_open
    )

    if bearish_engulfing:

        return {
            "pattern": "BEARISH_ENGULFING",
            "signal": "SELL",
            "strength": 3
        }

    # =====================================================
    # 6. INSIDE BAR
    # =====================================================

    inside_bar = (
        high <= prev_high
        and
        low >= prev_low
    )

    if inside_bar:

        return {
            "pattern": "INSIDE_BAR",
            "signal": "NEUTRAL",
            "strength": 1
        }

    # =====================================================
    # 7. MARUBOZU BULLISH
    # =====================================================

    bullish_marubozu = (
        current_bullish
        and
        body_ratio >= 0.90
        and
        upper_wick <= candle_range * 0.05
        and
        lower_wick <= candle_range * 0.05
    )

    if bullish_marubozu:

        return {
            "pattern": "BULLISH_MARUBOZU",
            "signal": "BUY",
            "strength": 2
        }

    # =====================================================
    # 8. MARUBOZU BEARISH
    # =====================================================

    bearish_marubozu = (
        current_bearish
        and
        body_ratio >= 0.90
        and
        upper_wick <= candle_range * 0.05
        and
        lower_wick <= candle_range * 0.05
    )

    if bearish_marubozu:

        return {
            "pattern": "BEARISH_MARUBOZU",
            "signal": "SELL",
            "strength": 2
        }

    # =====================================================
    # 9. MORNING STAR
    # =====================================================

    prev2_open = float(
        previous2["open"]
    )

    prev2_close = float(
        previous2["close"]
    )

    prev2_body = abs(
        prev2_close
        -
        prev2_open
    )

    morning_star = (
        prev2_close < prev2_open
        and
        prev_body < prev2_body * 0.5
        and
        current_bullish
        and
        close
        >
        (prev2_open + prev2_close) / 2
    )

    if morning_star:

        return {
            "pattern": "MORNING_STAR",
            "signal": "BUY",
            "strength": 3
        }

    # =====================================================
    # 10. EVENING STAR
    # =====================================================

    evening_star = (
        prev2_close > prev2_open
        and
        prev_body < prev2_body * 0.5
        and
        current_bearish
        and
        close
        <
        (prev2_open + prev2_close) / 2
    )

    if evening_star:

        return {
            "pattern": "EVENING_STAR",
            "signal": "SELL",
            "strength": 3
        }

    # =====================================================
    # DEFAULT
    # =====================================================

    if current_bullish:

        return {
            "pattern": "BULLISH",
            "signal": "BUY",
            "strength": 1
        }

    return {
        "pattern": "BEARISH",
        "signal": "SELL",
        "strength": 1
    }