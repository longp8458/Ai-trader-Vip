from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from backend.indicators import add_indicators
from backend.structure import detect_structure
from backend.candlestick import detect_candle
from backend.regime import detect_regime
from backend.levels import detect_levels
from backend.volume import analyze_volume


MIN_CANDLES = 220

# Risk/reference levels only — KHÔNG đặt lệnh.
RISK_ATR_MULTIPLIER = 1.5

# Validation thresholds
SIGNAL_QUALITY_MIN = 65.0
CONFIDENCE_MIN = 55.0

# Không cho tín hiệu nếu xung đột quá lớn.
MAX_CONFLICT = 0.35

# Chênh lệch BUY/SELL tối thiểu.
MIN_DIRECTIONAL_EDGE = 0.08


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)

        if not math.isfinite(x):
            return default

        return x

    except Exception:
        return default


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value)


def _to_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (int, float, np.integer, np.floating)):
        x = _safe(value, math.nan)

        if math.isfinite(x):
            return [x]

        return []

    return []


def _clean_number(value: Any):
    x = _safe(value, math.nan)

    if math.isfinite(x):
        return x

    return None


# ============================================================
# DATAFRAME
# ============================================================

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame rỗng")

    data = df.copy()

    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            f"Thiếu cột OHLCV: {', '.join(missing)}"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=required
    )

    if len(data) < MIN_CANDLES:
        raise ValueError(
            f"Không đủ dữ liệu: {len(data)}/{MIN_CANDLES} candles"
        )

    return data.reset_index(drop=True)


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def _component(result: Any) -> dict:
    if isinstance(result, dict):
        return result

    return {}


def _signal_from_component(result: dict) -> str:
    for key in (
        "signal",
        "direction",
        "bias",
        "trend",
    ):
        value = result.get(key)

        if value is not None:
            text = str(value).upper()

            if text in ("BUY", "BULLISH", "LONG"):
                return "BUY"

            if text in ("SELL", "BEARISH", "SHORT"):
                return "SELL"

            if text in ("NEUTRAL", "RANGE", "RANGING"):
                return "NEUTRAL"

    return "NEUTRAL"


def _score_component(result: dict) -> float:
    for key in (
        "score",
        "strength",
        "trend_strength",
        "confidence",
    ):
        if key in result:
            return max(
                0.0,
                min(
                    100.0,
                    _safe(result.get(key), 0.0),
                ),
            )

    return 0.0


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def _validate_signal(
    *,
    raw_signal: str,
    buy_score: float,
    sell_score: float,
    quality: float,
    confidence: float,
    conflict: float,
    structure_signal: str,
    regime_signal: str,
    candle_signal: str,
    volume_signal: str,
    mtf_bias: str = "NEUTRAL",
) -> dict:

    raw_signal = _text(
        raw_signal,
        "NEUTRAL",
    ).upper()

    if raw_signal not in (
        "BUY",
        "SELL",
        "NEUTRAL",
    ):
        raw_signal = "NEUTRAL"

    total = buy_score + sell_score

    if total > 0:
        directional_edge = abs(
            buy_score - sell_score
        ) / total
    else:
        directional_edge = 0.0

    reasons = []

    # --------------------------------------------------------
    # Quality gate
    # --------------------------------------------------------

    if quality < SIGNAL_QUALITY_MIN:
        reasons.append(
            f"Quality thấp ({quality:.1f} < {SIGNAL_QUALITY_MIN:.1f})"
        )

    # --------------------------------------------------------
    # Confidence gate
    # --------------------------------------------------------

    if confidence < CONFIDENCE_MIN:
        reasons.append(
            f"Confidence thấp ({confidence:.1f} < {CONFIDENCE_MIN:.1f})"
        )

    # --------------------------------------------------------
    # Conflict gate
    # --------------------------------------------------------

    if conflict > MAX_CONFLICT:
        reasons.append(
            f"Xung đột module cao ({conflict:.2f})"
        )

    # --------------------------------------------------------
    # Directional edge
    # --------------------------------------------------------

    if raw_signal in ("BUY", "SELL"):

        if directional_edge < MIN_DIRECTIONAL_EDGE:
            reasons.append(
                f"BUY/SELL edge quá thấp ({directional_edge:.2f})"
            )

    # --------------------------------------------------------
    # Structure validation
    # --------------------------------------------------------

    if raw_signal == "BUY":

        if structure_signal == "SELL":
            reasons.append(
                "Market Structure đang chống lại BUY"
            )

    elif raw_signal == "SELL":

        if structure_signal == "BUY":
            reasons.append(
                "Market Structure đang chống lại SELL"
            )

    # --------------------------------------------------------
    # Regime validation
    # --------------------------------------------------------

    if raw_signal == "BUY":

        if regime_signal == "SELL":
            reasons.append(
                "Market Regime đang chống lại BUY"
            )

    elif raw_signal == "SELL":

        if regime_signal == "BUY":
            reasons.append(
                "Market Regime đang chống lại SELL"
            )

    # --------------------------------------------------------
    # MTF validation
    # --------------------------------------------------------

    mtf_conflict = False

    if raw_signal == "BUY" and mtf_bias == "SELL":
        mtf_conflict = True

    if raw_signal == "SELL" and mtf_bias == "BUY":
        mtf_conflict = True

    if mtf_conflict:
        reasons.append(
            f"MTF bias ({mtf_bias}) chống lại tín hiệu"
        )

    # --------------------------------------------------------
    # Candlestick / volume
    # --------------------------------------------------------

    if raw_signal == "BUY" and candle_signal == "SELL":
        reasons.append(
            "Candlestick đang chống lại BUY"
        )

    if raw_signal == "SELL" and candle_signal == "BUY":
        reasons.append(
            "Candlestick đang chống lại SELL"
        )

    if raw_signal == "BUY" and volume_signal == "SELL":
        reasons.append(
            "Volume đang chống lại BUY"
        )

    if raw_signal == "SELL" and volume_signal == "BUY":
        reasons.append(
            "Volume đang chống lại SELL"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    final_signal = raw_signal

    if reasons:
        final_signal = "NEUTRAL"

    return {
        "raw_signal": raw_signal,
        "final_signal": final_signal,
        "validated": final_signal != "NEUTRAL",
        "directional_edge": round(
            directional_edge,
            4,
        ),
        "validation_reasons": reasons,
        "validation_passed": len(reasons) == 0,
        "mtf_bias": mtf_bias,
    }


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    signal: str,
    entry: float,
    atr: float,
    support: list | None = None,
    resistance: list | None = None,
) -> dict:

    signal = _text(
        signal,
        "NEUTRAL",
    ).upper()

    entry = _safe(entry)
    atr = _safe(atr)

    if signal not in ("BUY", "SELL"):
        return {
            "valid": False,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "risk_distance": None,
            "rr": {
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
        }

    if entry <= 0 or atr <= 0:
        return {
            "valid": False,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "risk_distance": None,
            "rr": {
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
        }

    support = _to_list(support)
    resistance = _to_list(resistance)

    distance = atr * RISK_ATR_MULTIPLIER

    if signal == "BUY":

        sl = entry - distance

        if support:
            valid_support = [
                x for x in support
                if _safe(x) < entry
            ]

            if valid_support:
                nearest = max(valid_support)

                # Không đặt SL sát entry hơn ATR-based SL.
                sl = min(
                    sl,
                    nearest,
                )

        risk = entry - sl

        if risk <= 0:
            return {
                "valid": False,
                "entry": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "risk_distance": None,
                "rr": {
                    "tp1": None,
                    "tp2": None,
                    "tp3": None,
                },
            }

        tp1 = entry + risk * 1.0
        tp2 = entry + risk * 2.0
        tp3 = entry + risk * 3.0

        return {
            "valid": True,
            "entry": _clean_number(entry),
            "sl": _clean_number(sl),
            "tp1": _clean_number(tp1),
            "tp2": _clean_number(tp2),
            "tp3": _clean_number(tp3),
            "risk_distance": _clean_number(risk),
            "rr": {
                "tp1": 1.0,
                "tp2": 2.0,
                "tp3": 3.0,
            },
        }

    # SELL

    sl = entry + distance

    if resistance:

        valid_resistance = [
            x for x in resistance
            if _safe(x) > entry
        ]

        if valid_resistance:
            nearest = min(valid_resistance)

            sl = max(
                sl,
                nearest,
            )

    risk = sl - entry

    if risk <= 0:
        return {
            "valid": False,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "risk_distance": None,
            "rr": {
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
        }

    tp1 = entry - risk * 1.0
    tp2 = entry - risk * 2.0
    tp3 = entry - risk * 3.0

    return {
        "valid": True,
        "entry": _clean_number(entry),
        "sl": _clean_number(sl),
        "tp1": _clean_number(tp1),
        "tp2": _clean_number(tp2),
        "tp3": _clean_number(tp3),
        "risk_distance": _clean_number(risk),
        "rr": {
            "tp1": 1.0,
            "tp2": 2.0,
            "tp3": 3.0,
        },
    }


# ============================================================
# QUALITY ENGINE
# ============================================================

def _quality_engine(
    *,
    trend_score: float,
    structure_score: float,
    momentum_score: float,
    volume_score: float,
    candle_score: float,
    regime_score: float,
    levels_score: float,
    conflict: float,
) -> float:

    quality = (
        trend_score * 0.20
        + structure_score * 0.20
        + momentum_score * 0.15
        + volume_score * 0.10
        + candle_score * 0.08
        + regime_score * 0.12
        + levels_score * 0.15
    )

    # Conflict penalty
    quality *= (
        1.0 - min(
            0.50,
            max(0.0, conflict),
        )
    )

    return max(
        0.0,
        min(
            100.0,
            quality,
        ),
    )


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_market(
    df: pd.DataFrame,
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "UNKNOWN",
    mtf_bias: str = "NEUTRAL",
) -> dict:

    data = _prepare_dataframe(df)

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    data = add_indicators(data)

    if data.empty:
        raise ValueError(
            "Không còn dữ liệu sau khi tính indicators"
        )

    latest = data.iloc[-1]

    close = _safe(
        latest.get("close"),
        0.0,
    )

    if close <= 0:
        raise ValueError(
            "Giá hiện tại không hợp lệ"
        )

    # --------------------------------------------------------
    # Technical components
    # --------------------------------------------------------

    structure = _component(
        detect_structure(data)
    )

    levels = _component(
        detect_levels(data)
    )

    candle = _component(
        detect_candle(data)
    )

    regime = _component(
        detect_regime(data)
    )

    volume = _component(
        analyze_volume(data)
    )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    ema20 = _safe(
        latest.get("EMA20"),
        0.0,
    )

    ema50 = _safe(
        latest.get("EMA50"),
        0.0,
    )

    ema200 = _safe(
        latest.get("EMA200"),
        0.0,
    )

    rsi = _safe(
        latest.get("RSI14"),
        50.0,
    )

    macd = _safe(
        latest.get("MACD"),
        0.0,
    )

    macd_signal = _safe(
        latest.get("MACD_SIGNAL"),
        0.0,
    )

    atr = _safe(
        latest.get("ATR14"),
        0.0,
    )

    # --------------------------------------------------------
    # Component signals
    # --------------------------------------------------------

    structure_signal = _signal_from_component(
        structure
    )

    regime_signal = _signal_from_component(
        regime
    )

    candle_signal = _signal_from_component(
        candle
    )

    volume_signal = _signal_from_component(
        volume
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    buy_score = 0.0
    sell_score = 0.0

    breakdown = {}

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend_buy = 0.0
    trend_sell = 0.0

    if ema20 > ema50:
        trend_buy += 25
    else:
        trend_sell += 25

    if ema50 > ema200:
        trend_buy += 25
    else:
        trend_sell += 25

    if close > ema20:
        trend_buy += 20
    else:
        trend_sell += 20

    if close > ema200:
        trend_buy += 30
    else:
        trend_sell += 30

    buy_score += trend_buy
    sell_score += trend_sell

    breakdown["trend"] = {
        "buy": round(trend_buy, 2),
        "sell": round(trend_sell, 2),
        "signal": (
            "BUY"
            if trend_buy > trend_sell
            else "SELL"
            if trend_sell > trend_buy
            else "NEUTRAL"
        ),
    }

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum_buy = 0.0
    momentum_sell = 0.0

    if rsi >= 55:
        momentum_buy += 50

    elif rsi <= 45:
        momentum_sell += 50

    if macd > macd_signal:
        momentum_buy += 50
    else:
        momentum_sell += 50

    buy_score += momentum_buy
    sell_score += momentum_sell

    breakdown["momentum"] = {
        "buy": round(momentum_buy, 2),
        "sell": round(momentum_sell, 2),
        "rsi": round(rsi, 2),
        "macd": round(macd, 8),
        "macd_signal": round(
            macd_signal,
            8,
        ),
    }

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    structure_score = _score_component(
        structure
    )

    if structure_signal == "BUY":
        buy_score += 100
    elif structure_signal == "SELL":
        sell_score += 100

    breakdown["structure"] = {
        "signal": structure_signal,
        "score": round(
            structure_score,
            2,
        ),
        "bos": structure.get("bos"),
        "bos_direction": structure.get(
            "bos_direction"
        ),
        "choch": structure.get("choch"),
        "choch_direction": structure.get(
            "choch_direction"
        ),
        "trend": structure.get("trend"),
    }

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle_score = _score_component(
        candle
    )

    if candle_signal == "BUY":
        buy_score += 60
    elif candle_signal == "SELL":
        sell_score += 60

    breakdown["candlestick"] = {
        "signal": candle_signal,
        "pattern": candle.get(
            "pattern"
        ),
        "strength": round(
            candle_score,
            2,
        ),
    }

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_score = _score_component(
        volume
    )

    if volume_signal == "BUY":
        buy_score += 50
    elif volume_signal == "SELL":
        sell_score += 50

    breakdown["volume"] = {
        "signal": volume_signal,
        "strength": round(
            volume_score,
            2,
        ),
        "volume_ratio": _clean_number(
            volume.get("volume_ratio")
        ),
        "volume_spike": bool(
            volume.get("volume_spike", False)
        ),
    }

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    regime_score = _score_component(
        regime
    )

    if regime_signal == "BUY":
        buy_score += 50
    elif regime_signal == "SELL":
        sell_score += 50

    breakdown["regime"] = {
        "signal": regime_signal,
        "trend": regime.get("trend"),
        "volatility": regime.get(
            "volatility"
        ),
        "atr_pct": _clean_number(
            regime.get("atr_pct")
        ),
        "strength": round(
            regime_score,
            2,
        ),
    }

    # --------------------------------------------------------
    # LEVELS
    # --------------------------------------------------------

    support = _to_list(
        levels.get("support")
    )

    resistance = _to_list(
        levels.get("resistance")
    )

    nearest_support = _safe(
        levels.get("nearest_support"),
        math.nan,
    )

    nearest_resistance = _safe(
        levels.get("nearest_resistance"),
        math.nan,
    )

    levels_buy = 0.0
    levels_sell = 0.0

    if math.isfinite(nearest_support):
        if close > nearest_support:
            levels_buy += 50

    if math.isfinite(nearest_resistance):
        if close < nearest_resistance:
            levels_sell += 50

    if levels_buy == 0 and levels_sell == 0:
        levels_buy = 25
        levels_sell = 25

    buy_score += levels_buy
    sell_score += levels_sell

    levels_score = max(
        levels_buy,
        levels_sell,
    )

    breakdown["levels"] = {
        "buy": round(levels_buy, 2),
        "sell": round(levels_sell, 2),
        "support": [
            _clean_number(x)
            for x in support
        ],
        "resistance": [
            _clean_number(x)
            for x in resistance
        ],
        "nearest_support":
            _clean_number(
                nearest_support
            ),
        "nearest_resistance":
            _clean_number(
                nearest_resistance
            ),
    }

    # --------------------------------------------------------
    # NORMALIZE SCORES
    # --------------------------------------------------------

    max_score = max(
        buy_score,
        sell_score,
        1.0,
    )

    buy_norm = (
        buy_score / max_score * 100
    )

    sell_norm = (
        sell_score / max_score * 100
    )

    # --------------------------------------------------------
    # RAW SIGNAL
    # --------------------------------------------------------

    total_directional = (
        buy_score + sell_score
    )

    if total_directional <= 0:
        raw_signal = "NEUTRAL"

    elif buy_score > sell_score:
        raw_signal = "BUY"

    elif sell_score > buy_score:
        raw_signal = "SELL"

    else:
        raw_signal = "NEUTRAL"

    # --------------------------------------------------------
    # CONFLICT ENGINE
    # --------------------------------------------------------

    directional_signals = [
        trend_signal
        for trend_signal in [
            breakdown["trend"]["signal"],
            structure_signal,
            candle_signal,
            volume_signal,
            regime_signal,
        ]
        if trend_signal in (
            "BUY",
            "SELL",
        )
    ]

    conflict = 0.0

    if directional_signals:

        buy_votes = directional_signals.count(
            "BUY"
        )

        sell_votes = directional_signals.count(
            "SELL"
        )

        total_votes = (
            buy_votes + sell_votes
        )

        conflict = min(
            buy_votes,
            sell_votes,
        ) / max(
            total_votes,
            1,
        )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    trend_score = max(
        trend_buy,
        trend_sell,
    )

    momentum_score = max(
        momentum_buy,
        momentum_sell,
    )

    quality = _quality_engine(
        trend_score=min(
            100,
            trend_score,
        ),
        structure_score=structure_score,
        momentum_score=min(
            100,
            momentum_score,
        ),
        volume_score=volume_score,
        candle_score=candle_score,
        regime_score=regime_score,
        levels_score=levels_score,
        conflict=conflict,
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if total_directional > 0:

        directional_confidence = (
            abs(
                buy_score - sell_score
            )
            / total_directional
            * 100
        )

    else:
        directional_confidence = 0.0

    confidence = (
        directional_confidence * 0.60
        + quality * 0.40
    )

    confidence = max(
        0.0,
        min(
            100.0,
            confidence,
        ),
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation = _validate_signal(
        raw_signal=raw_signal,
        buy_score=buy_score,
        sell_score=sell_score,
        quality=quality,
        confidence=confidence,
        conflict=conflict,
        structure_signal=structure_signal,
        regime_signal=regime_signal,
        candle_signal=candle_signal,
        volume_signal=volume_signal,
        mtf_bias=_text(
            mtf_bias,
            "NEUTRAL",
        ).upper(),
    )

    signal = validation[
        "final_signal"
    ]

    # --------------------------------------------------------
    # TRADE LEVELS
    # --------------------------------------------------------

    trade_levels = calculate_trade_levels(
        signal=signal,
        entry=close,
        atr=atr,
        support=support,
        resistance=resistance,
    )

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    reasoning = []

    if trend_buy > trend_sell:
        reasoning.append(
            "EMA đang nghiêng về xu hướng tăng."
        )

    elif trend_sell > trend_buy:
        reasoning.append(
            "EMA đang nghiêng về xu hướng giảm."
        )

    else:
        reasoning.append(
            "EMA chưa cho thấy ưu thế rõ."
        )

    if rsi >= 55:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng tăng."
        )

    elif rsi <= 45:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng giảm."
        )

    else:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng trung tính."
        )

    if macd > macd_signal:
        reasoning.append(
            "MACD bullish."
        )

    else:
        reasoning.append(
            "MACD bearish."
        )

    reasoning.append(
        f"Market Structure: {structure_signal}."
    )

    reasoning.append(
        f"Candlestick: {candle_signal}."
    )

    reasoning.append(
        f"Volume: {volume_signal}."
    )

    reasoning.append(
        f"Regime: {regime_signal}."
    )

    reasoning.append(
        f"MTF bias: {_text(mtf_bias, 'NEUTRAL').upper()}."
    )

    if validation["validation_passed"]:
        reasoning.append(
            "Signal vượt qua toàn bộ lớp validation."
        )

    else:
        for reason in validation[
            "validation_reasons"
        ]:
            reasoning.append(
                f"Validation: {reason}."
            )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "symbol": symbol,
        "timeframe": timeframe,

        "signal": signal,
        "raw_signal": raw_signal,

        "confidence": round(
            confidence,
            2,
        ),

        "signal_quality": round(
            quality,
            2,
        ),

        "directional_edge": round(
            validation[
                "directional_edge"
            ],
            4,
        ),

        "conflict": round(
            conflict,
            4,
        ),

        "buy_score": round(
            buy_norm,
            2,
        ),

        "sell_score": round(
            sell_norm,
            2,
        ),

        "price": _clean_number(
            close
        ),

        "entry": trade_levels[
            "entry"
        ],

        "sl": trade_levels[
            "sl"
        ],

        "tp1": trade_levels[
            "tp1"
        ],

        "tp2": trade_levels[
            "tp2"
        ],

        "tp3": trade_levels[
            "tp3"
        ],

        "rr": trade_levels[
            "rr"
        ],

        "trade_levels": trade_levels,

        "validation": validation,

        "breakdown": breakdown,

        "reasoning": reasoning,

        "indicators": {
            "ema20": _clean_number(
                ema20
            ),
            "ema50": _clean_number(
                ema50
            ),
            "ema200": _clean_number(
                ema200
            ),
            "rsi14": _clean_number(
                rsi
            ),
            "macd": _clean_number(
                macd
            ),
            "macd_signal":
                _clean_number(
                    macd_signal
                ),
            "atr14": _clean_number(
                atr
            ),
        },

        "modules": {
            "trend": breakdown[
                "trend"
            ],
            "momentum": breakdown[
                "momentum"
            ],
            "structure": breakdown[
                "structure"
            ],
            "candlestick": breakdown[
                "candlestick"
            ],
            "volume": breakdown[
                "volume"
            ],
            "regime": breakdown[
                "regime"
            ],
            "levels": breakdown[
                "levels"
            ],
        },

        "analysis_only": True,
    }