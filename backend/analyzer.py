import pandas as pd

from .indicators import add_indicators
from .structure import detect_structure
from .candlestick import detect_candle
from .regime import detect_market_regime
from .levels import detect_support_resistance
from .volume import analyze_volume


def calculate_trade_levels(
    price,
    atr_value,
    signal
):
    """
    Tạo các mức giá tham chiếu cho phân tích.

    Lưu ý:
    - Không đặt lệnh.
    - Không gửi lệnh.
    - Chỉ dùng làm mức tham chiếu phân tích.
    """

    if price is None:
        return {
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr": None,
            "method": "ATR_1.5R_REFERENCE"
        }

    if atr_value is None or atr_value <= 0:
        atr_value = price * 0.01

    risk_distance = atr_value * 1.5

    # =========================
    # BUY
    # =========================

    if signal == "BUY":

        entry = price

        sl = (
            entry
            -
            risk_distance
        )

        tp1 = (
            entry
            +
            risk_distance
        )

        tp2 = (
            entry
            +
            risk_distance * 2
        )

        tp3 = (
            entry
            +
            risk_distance * 3
        )

    # =========================
    # SELL
    # =========================

    elif signal == "SELL":

        entry = price

        sl = (
            entry
            +
            risk_distance
        )

        tp1 = (
            entry
            -
            risk_distance
        )

        tp2 = (
            entry
            -
            risk_distance * 2
        )

        tp3 = (
            entry
            -
            risk_distance * 3
        )

    # =========================
    # NEUTRAL
    # =========================

    else:

        return {
            "entry": round(
                price,
                6
            ),
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr": None,
            "method": "ATR_1.5R_REFERENCE"
        }

    return {
        "entry": round(
            entry,
            6
        ),

        "sl": round(
            sl,
            6
        ),

        "tp1": round(
            tp1,
            6
        ),

        "tp2": round(
            tp2,
            6
        ),

        "tp3": round(
            tp3,
            6
        ),

        "rr": {
            "tp1": 1.0,
            "tp2": 2.0,
            "tp3": 3.0
        },

        "method": "ATR_1.5R_REFERENCE"
    }


def analyze_market(df):

    # =====================================================
    # KIỂM TRA DỮ LIỆU
    # =====================================================

    if df is None or len(df) < 220:

        return {
            "signal": "NEUTRAL",
            "confidence": 0,
            "price": None,
            "reasoning": (
                "Không đủ dữ liệu. "
                "Cần ít nhất 220 nến."
            )
        }

    data = df.copy()

    # =====================================================
    # TECHNICAL INDICATORS
    # =====================================================

    data = add_indicators(
        data
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    if len(data) == 0:

        return {
            "signal": "NEUTRAL",
            "confidence": 0,
            "price": None,
            "reasoning": (
                "Dữ liệu indicator "
                "không hợp lệ."
            )
        }

    latest = data.iloc[-1]

    price = float(
        latest["close"]
    )

    # =====================================================
    # MARKET STRUCTURE
    # =====================================================

    structure = detect_structure(
        data
    )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================

    levels = detect_support_resistance(
        data,
        structure.get(
            "swing_highs",
            []
        ),
        structure.get(
            "swing_lows",
            []
        )
    )

    # =====================================================
    # CANDLESTICK
    # =====================================================

    candlestick = detect_candle(
        data
    )

    # =====================================================
    # MARKET REGIME
    # =====================================================

    regime = detect_market_regime(
        data
    )

    # =====================================================
    # VOLUME ENGINE
    # =====================================================

    volume = analyze_volume(
        data
    )

    # =====================================================
    # SCORE
    # =====================================================

    buy_points = 0
    sell_points = 0

    reasons = []

    # =====================================================
    # EMA20 / EMA50
    # =====================================================

    if (
        latest["ema20"]
        >
        latest["ema50"]
    ):

        buy_points += 1

        reasons.append(
            "EMA20 > EMA50"
        )

    elif (
        latest["ema20"]
        <
        latest["ema50"]
    ):

        sell_points += 1

        reasons.append(
            "EMA20 < EMA50"
        )

    # =====================================================
    # EMA200
    # =====================================================

    if (
        latest["close"]
        >
        latest["ema200"]
    ):

        buy_points += 1

        reasons.append(
            "Giá > EMA200"
        )

    elif (
        latest["close"]
        <
        latest["ema200"]
    ):

        sell_points += 1

        reasons.append(
            "Giá < EMA200"
        )

    # =====================================================
    # RSI
    # =====================================================

    rsi = float(
        latest["rsi"]
    )

    if rsi >= 55:

        buy_points += 1

        reasons.append(
            f"RSI tăng ({rsi:.2f})"
        )

    elif rsi <= 45:

        sell_points += 1

        reasons.append(
            f"RSI giảm ({rsi:.2f})"
        )

    # =====================================================
    # MACD
    # =====================================================

    macd = float(
        latest["macd"]
    )

    macd_signal = float(
        latest["macd_signal"]
    )

    if macd > macd_signal:

        buy_points += 1

        reasons.append(
            "MACD > Signal"
        )

    elif macd < macd_signal:

        sell_points += 1

        reasons.append(
            "MACD < Signal"
        )

    # =====================================================
    # MARKET STRUCTURE TREND
    # =====================================================

    structure_trend = structure.get(
        "trend",
        "NEUTRAL"
    )

    if structure_trend == "BULLISH":

        buy_points += 1

        reasons.append(
            "Market Structure bullish"
        )

    elif structure_trend == "BEARISH":

        sell_points += 1

        reasons.append(
            "Market Structure bearish"
        )

    # =====================================================
    # BOS
    # =====================================================

    bos_direction = structure.get(
        "bos_direction",
        "NONE"
    )

    if bos_direction == "BULLISH":

        buy_points += 1

        reasons.append(
            "Bullish BOS"
        )

    elif bos_direction == "BEARISH":

        sell_points += 1

        reasons.append(
            "Bearish BOS"
        )

    # =====================================================
    # CHoCH
    # =====================================================

    choch_direction = structure.get(
        "choch_direction",
        "NONE"
    )

    if choch_direction == "BULLISH":

        buy_points += 1

        reasons.append(
            "Bullish CHoCH"
        )

    elif choch_direction == "BEARISH":

        sell_points += 1

        reasons.append(
            "Bearish CHoCH"
        )

    # =====================================================
    # CANDLESTICK
    # =====================================================

    candle_signal = candlestick.get(
        "signal",
        "NEUTRAL"
    )

    candle_pattern = candlestick.get(
        "pattern",
        "UNKNOWN"
    )

    candle_strength = candlestick.get(
        "strength",
        0
    )

    if candle_signal == "BUY":

        buy_points += min(
            candle_strength,
            2
        )

        reasons.append(
            f"Candlestick BUY: "
            f"{candle_pattern}"
        )

    elif candle_signal == "SELL":

        sell_points += min(
            candle_strength,
            2
        )

        reasons.append(
            f"Candlestick SELL: "
            f"{candle_pattern}"
        )

    else:

        reasons.append(
            f"Candlestick: "
            f"{candle_pattern}"
        )

    # =====================================================
    # VOLUME ENGINE
    # =====================================================

    volume_signal = volume.get(
        "signal",
        "NEUTRAL"
    )

    volume_strength = volume.get(
        "strength",
        0
    )

    volume_ratio = volume.get(
        "volume_ratio"
    )

    volume_spike = volume.get(
        "volume_spike",
        False
    )

    if volume_signal == "BUY":

        buy_points += min(
            volume_strength,
            2
        )

        reasons.append(
            "Volume xác nhận BUY: "
            +
            volume.get(
                "reason",
                ""
            )
        )

    elif volume_signal == "SELL":

        sell_points += min(
            volume_strength,
            2
        )

        reasons.append(
            "Volume xác nhận SELL: "
            +
            volume.get(
                "reason",
                ""
            )
        )

    else:

        reasons.append(
            "Volume chưa xác nhận xu hướng."
        )

    if volume_spike:

        reasons.append(
            f"Volume Spike "
            f"{volume_ratio:.2f}x"
        )

    # =====================================================
    # MARKET REGIME
    # =====================================================

    regime_trend = regime.get(
        "trend",
        "NEUTRAL"
    )

    regime_name = regime.get(
        "regime",
        "UNKNOWN"
    )

    if regime_trend == "BULLISH":

        buy_points += 1

        reasons.append(
            f"Regime bullish: "
            f"{regime_name}"
        )

    elif regime_trend == "BEARISH":

        sell_points += 1

        reasons.append(
            f"Regime bearish: "
            f"{regime_name}"
        )

    else:

        reasons.append(
            f"Regime: "
            f"{regime_name}"
        )

    # =====================================================
    # SUPPORT
    # =====================================================

    supports = levels.get(
        "support",
        []
    )

    if supports:

        nearest_support = supports[0]

        reasons.append(
            "Support gần nhất: "
            f"{nearest_support:.4f}"
        )

    # =====================================================
    # RESISTANCE
    # =====================================================

    resistances = levels.get(
        "resistance",
        []
    )

    if resistances:

        nearest_resistance = resistances[0]

        reasons.append(
            "Resistance gần nhất: "
            f"{nearest_resistance:.4f}"
        )

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    total_points = (
        buy_points
        +
        sell_points
    )

    if (
        buy_points > sell_points
        and
        buy_points >= 3
    ):

        signal = "BUY"

    elif (
        sell_points > buy_points
        and
        sell_points >= 3
    ):

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    if total_points > 0:

        if signal == "BUY":

            confidence = (
                buy_points
                /
                total_points
            ) * 100

        elif signal == "SELL":

            confidence = (
                sell_points
                /
                total_points
            ) * 100

        else:

            confidence = 50

    else:

        confidence = 0

    confidence = min(
        confidence,
        100
    )

    # =====================================================
    # ATR
    # =====================================================

    atr_value = float(
        latest["atr"]
    )

    # =====================================================
    # TRADE LEVELS
    # =====================================================

    trade_levels = calculate_trade_levels(
        price,
        atr_value,
        signal
    )

    # =====================================================
    # INDICATORS
    # =====================================================

    indicators = {

        "ema20": float(
            latest["ema20"]
        ),

        "ema50": float(
            latest["ema50"]
        ),

        "ema200": float(
            latest["ema200"]
        ),

        "sma20": float(
            latest["sma20"]
        ),

        "sma50": float(
            latest["sma50"]
        ),

        "sma200": float(
            latest["sma200"]
        ),

        "rsi": float(
            latest["rsi"]
        ),

        "macd": float(
            latest["macd"]
        ),

        "macd_signal": float(
            latest["macd_signal"]
        ),

        "macd_histogram": float(
            latest["macd_histogram"]
        ),

        "atr": float(
            latest["atr"]
        ),

        "volume": float(
            latest["volume"]
        ),

        "volume_sma20": float(
            latest["volume_sma20"]
        )
    }

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "signal": signal,

        "confidence": round(
            confidence,
            2
        ),

        "price": price,

        "buy_points": buy_points,

        "sell_points": sell_points,

        "indicators": indicators,

        "structure": structure,

        "levels": levels,

        "candlestick": candlestick,

        "volume": volume,

        "regime": regime,

        "trade_levels": trade_levels,

        "reasoning": (
            " | ".join(
                reasons
            )
        )
    }