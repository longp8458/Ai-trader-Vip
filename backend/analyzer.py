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


# ============================================================
# NOVATRADE AI V6.3
# MARKET ANALYZER
# ============================================================

MIN_CANDLES = 220

RISK_ATR_MULTIPLIER = 1.5

SIGNAL_QUALITY_MIN = 65.0
CONFIDENCE_MIN = 55.0
MAX_CONFLICT = 0.35
MIN_DIRECTIONAL_EDGE = 0.08


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe(value: Any, default: float = 0.0) -> float:
    """
    Convert scalar numeric values safely.
    Never allow NaN / Inf into JSON.
    """

    if value is None:
        return default

    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            value = value.iloc[-1]

        if isinstance(value, pd.DataFrame):
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except Exception:
        return default


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    try:
        return str(value)
    except Exception:
        return default


def _clean_number(value: Any, digits: int = 4) -> Any:
    """
    JSON-safe number.
    """

    if value is None:
        return None

    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            value = value.iloc[-1]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating, float, int)):
            value = float(value)

            if not math.isfinite(value):
                return None

            return round(value, digits)

    except Exception:
        return None

    return value


def _json_safe(value: Any) -> Any:
    """
    Recursively remove NaN / Inf / numpy objects.
    """

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())

    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def _to_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _component(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    return {
        "signal": "NEUTRAL",
        "value": value,
    }


def _signal_from_component(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "signal",
            "direction",
            "bias",
            "trend",
            "result",
        ):
            if key in value:
                signal = _text(
                    value.get(key),
                    "NEUTRAL",
                ).upper()

                if signal in {"BUY", "SELL", "NEUTRAL"}:
                    return signal

    if isinstance(value, str):
        signal = value.upper()

        if signal in {"BUY", "SELL", "NEUTRAL"}:
            return signal

    return "NEUTRAL"


def _score_component(value: Any) -> float:
    """
    Convert a module result into a 0-100 score.
    """

    if isinstance(value, dict):
        for key in (
            "score",
            "confidence",
            "strength",
            "quality",
        ):
            if key in value:
                return max(
                    0.0,
                    min(
                        100.0,
                        _safe(value.get(key))
                    )
                )

    return 0.0


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("Không có dữ liệu thị trường.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame rỗng.")

    data = df.copy()

    # Normalize column names
    rename_map = {}

    for column in data.columns:
        name = str(column).strip().lower()

        if name == "timestamp":
            rename_map[column] = "timestamp"

        elif name == "time":
            rename_map[column] = "timestamp"

        elif name == "datetime":
            rename_map[column] = "timestamp"

        elif name == "open":
            rename_map[column] = "open"

        elif name == "high":
            rename_map[column] = "high"

        elif name == "low":
            rename_map[column] = "low"

        elif name == "close":
            rename_map[column] = "close"

        elif name == "volume":
            rename_map[column] = "volume"

    data = data.rename(columns=rename_map)

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu cột OHLC: {missing}"
        )

    if "volume" not in data.columns:
        data["volume"] = 0.0

    # Numeric conversion
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
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
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # Validate OHLC
    data = data[
        (data["high"] >= data["low"])
        & (data["high"] >= data["open"])
        & (data["high"] >= data["close"])
        & (data["low"] <= data["open"])
        & (data["low"] <= data["close"])
    ]

    # Timestamp
    if "timestamp" in data.columns:
        try:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"],
                errors="coerce",
            )
        except Exception:
            pass

        data = data.dropna(
            subset=["timestamp"]
        )

        data = data.sort_values(
            "timestamp"
        )

        data = data.drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )

    else:
        data = data.sort_index()

    data = data.reset_index(drop=True)

    if len(data) < MIN_CANDLES:
        raise ValueError(
            f"Không đủ dữ liệu: {len(data)} nến. "
            f"Cần tối thiểu {MIN_CANDLES} nến."
        )

    return data


# ============================================================
# INDICATORS
# ============================================================

def _calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    # Special cases
    rsi = rsi.where(
        ~(
            (avg_loss == 0)
            & (avg_gain > 0)
        ),
        100.0,
    )

    rsi = rsi.where(
        ~(
            (avg_gain == 0)
            & (avg_loss > 0)
        ),
        0.0,
    )

    return rsi


def _series_from_column(
    data: pd.DataFrame,
    names: list[str],
) -> pd.Series | None:

    """
    Safely get an existing DataFrame column.

    IMPORTANT:
    Never use:
        series_a or series_b

    because Pandas Series cannot be evaluated as bool.
    """

    for name in names:

        if name not in data.columns:
            continue

        value = data[name]

        if isinstance(value, pd.Series):
            return value

    return None


def _ensure_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    # --------------------------------------------------------
    # First try project's indicator engine
    # --------------------------------------------------------

    try:
        result = add_indicators(
            data.copy()
        )

        if isinstance(result, pd.DataFrame):
            data = result.copy()

    except Exception:
        # The analyzer has its own fallback indicators.
        pass

    # --------------------------------------------------------
    # Normalize possible indicator column names
    # --------------------------------------------------------

    def get_existing(
        names: list[str],
    ) -> pd.Series | None:

        return _series_from_column(
            data,
            names,
        )

    # ========================================================
    # EMA 20
    # ========================================================

    ema20 = get_existing(
        [
            "EMA20",
            "ema20",
            "EMA_20",
            "ema_20",
        ]
    )

    if ema20 is None:
        ema20 = data["close"].ewm(
            span=20,
            adjust=False,
            min_periods=20,
        ).mean()

    data["EMA20"] = pd.to_numeric(
        ema20,
        errors="coerce",
    )

    # ========================================================
    # EMA 50
    # ========================================================

    ema50 = get_existing(
        [
            "EMA50",
            "ema50",
            "EMA_50",
            "ema_50",
        ]
    )

    if ema50 is None:
        ema50 = data["close"].ewm(
            span=50,
            adjust=False,
            min_periods=50,
        ).mean()

    data["EMA50"] = pd.to_numeric(
        ema50,
        errors="coerce",
    )

    # ========================================================
    # EMA 200
    # ========================================================

    ema200 = get_existing(
        [
            "EMA200",
            "ema200",
            "EMA_200",
            "ema_200",
        ]
    )

    if ema200 is None:
        ema200 = data["close"].ewm(
            span=200,
            adjust=False,
            min_periods=200,
        ).mean()

    data["EMA200"] = pd.to_numeric(
        ema200,
        errors="coerce",
    )

    # ========================================================
    # RSI 14
    # ========================================================

    rsi14 = get_existing(
        [
            "RSI14",
            "RSI",
            "rsi14",
            "rsi",
            "RSI_14",
        ]
    )

    if rsi14 is None:
        rsi14 = _calculate_rsi(
            data["close"],
            14,
        )

    data["RSI14"] = pd.to_numeric(
        rsi14,
        errors="coerce",
    )

    # ========================================================
    # MACD
    # ========================================================

    macd = get_existing(
        [
            "MACD",
            "macd",
        ]
    )

    macd_signal = get_existing(
        [
            "MACD_SIGNAL",
            "MACDSignal",
            "macd_signal",
            "macdSignal",
        ]
    )

    if macd is None:
        ema12 = data["close"].ewm(
            span=12,
            adjust=False,
        ).mean()

        ema26 = data["close"].ewm(
            span=26,
            adjust=False,
        ).mean()

        macd = ema12 - ema26

    if macd_signal is None:
        macd_signal = pd.Series(
            macd,
            index=data.index,
        ).ewm(
            span=9,
            adjust=False,
        ).mean()

    data["MACD"] = pd.to_numeric(
        macd,
        errors="coerce",
    )

    data["MACD_SIGNAL"] = pd.to_numeric(
        macd_signal,
        errors="coerce",
    )

    # ========================================================
    # ATR 14
    # ========================================================

    atr14 = get_existing(
        [
            "ATR14",
            "ATR",
            "atr14",
            "atr",
            "ATR_14",
        ]
    )

    if atr14 is None:

        previous_close = data[
            "close"
        ].shift(1)

        tr1 = (
            data["high"]
            - data["low"]
        )

        tr2 = (
            data["high"]
            - previous_close
        ).abs()

        tr3 = (
            data["low"]
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

        atr14 = true_range.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        ).mean()

    data["ATR14"] = pd.to_numeric(
        atr14,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Fill indicator gaps
    # --------------------------------------------------------

    indicator_columns = [
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI14",
        "MACD",
        "MACD_SIGNAL",
        "ATR14",
    ]

    for column in indicator_columns:

        data[column] = (
            data[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        data[column] = (
            data[column]
            .ffill()
            .bfill()
        )

    return data


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def _validate_signal(
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

    raw_signal = (
        raw_signal
        if raw_signal in {"BUY", "SELL"}
        else "NEUTRAL"
    )

    total = buy_score + sell_score

    if total <= 0:
        directional_edge = 0.0
    else:
        directional_edge = (
            abs(buy_score - sell_score)
            / total
        )

    reasons: list[str] = []

    if quality < SIGNAL_QUALITY_MIN:
        reasons.append(
            f"Signal quality thấp "
            f"({quality:.1f} < {SIGNAL_QUALITY_MIN:.1f})"
        )

    if confidence < CONFIDENCE_MIN:
        reasons.append(
            f"Confidence thấp "
            f"({confidence:.1f} < {CONFIDENCE_MIN:.1f})"
        )

    if conflict > MAX_CONFLICT:
        reasons.append(
            f"Conflict cao "
            f"({conflict:.2f} > {MAX_CONFLICT:.2f})"
        )

    if directional_edge < MIN_DIRECTIONAL_EDGE:
        reasons.append(
            f"Directional edge thấp "
            f"({directional_edge:.2f})"
        )

    opposing = {
        "BUY": "SELL",
        "SELL": "BUY",
    }

    opposite = opposing.get(raw_signal)

    if opposite:

        components = {
            "structure": structure_signal,
            "regime": regime_signal,
            "candle": candle_signal,
            "volume": volume_signal,
        }

        for name, signal in components.items():

            if signal == opposite:
                reasons.append(
                    f"{name} đang chống lại {raw_signal}"
                )

    if (
        mtf_bias in {"BUY", "SELL"}
        and raw_signal != mtf_bias
    ):
        reasons.append(
            f"MTF bias {mtf_bias} chống lại {raw_signal}"
        )

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
        "validation_passed": (
            len(reasons) == 0
        ),
        "mtf_bias": mtf_bias,
    }


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    data: pd.DataFrame,
    signal: str,
) -> dict:

    if data.empty:
        return {
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr": None,
        }

    close = _safe(
        data["close"].iloc[-1]
    )

    atr = _safe(
        data["ATR14"].iloc[-1]
    )

    if close <= 0:
        return {
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr": None,
        }

    if atr <= 0:
        atr = close * 0.005

    risk = atr * RISK_ATR_MULTIPLIER

    if signal == "BUY":

        entry = close

        sl = entry - risk

        tp1 = entry + risk * 1.0
        tp2 = entry + risk * 2.0
        tp3 = entry + risk * 3.0

        rr = 1.0

    elif signal == "SELL":

        entry = close

        sl = entry + risk

        tp1 = entry - risk * 1.0
        tp2 = entry - risk * 2.0
        tp3 = entry - risk * 3.0

        rr = 1.0

    else:

        return {
            "entry": _clean_number(close),
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr": None,
        }

    return {
        "entry": _clean_number(entry),
        "sl": _clean_number(sl),
        "tp1": _clean_number(tp1),
        "tp2": _clean_number(tp2),
        "tp3": _clean_number(tp3),
        "rr": _clean_number(rr),
    }


# ============================================================
# QUALITY ENGINE
# ============================================================

def _quality_engine(
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

    conflict_penalty = 1.0 - min(
        0.50,
        max(0.0, conflict),
    )

    quality *= conflict_penalty

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
    symbol: str = "UNKNOWN",
    timeframe: str = "M15",
    mtf_bias: str = "NEUTRAL",
) -> dict:

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = _prepare_dataframe(
        df
    )

    data = _ensure_indicators(
        data
    )

    # --------------------------------------------------------
    # Latest values
    # --------------------------------------------------------

    latest = data.iloc[-1]

    close = _safe(
        latest["close"]
    )

    ema20 = _safe(
        latest["EMA20"]
    )

    ema50 = _safe(
        latest["EMA50"]
    )

    ema200 = _safe(
        latest["EMA200"]
    )

    rsi = _safe(
        latest["RSI14"]
    )

    macd = _safe(
        latest["MACD"]
    )

    macd_signal = _safe(
        latest["MACD_SIGNAL"]
    )

    atr = _safe(
        latest["ATR14"]
    )

    # ========================================================
    # EXTERNAL MODULES
    # ========================================================

    structure = {}

    try:
        structure = _component(
            detect_structure(data)
        )
    except Exception as exc:
        structure = {
            "signal": "NEUTRAL",
            "error": str(exc),
        }

    levels = {}

    try:
        levels = _component(
            detect_levels(data)
        )
    except Exception as exc:
        levels = {
            "signal": "NEUTRAL",
            "error": str(exc),
        }

    candle = {}

    try:
        candle = _component(
            detect_candle(data)
        )
    except Exception as exc:
        candle = {
            "signal": "NEUTRAL",
            "error": str(exc),
        }

    regime = {}

    try:
        regime = _component(
            detect_regime(data)
        )
    except Exception as exc:
        regime = {
            "signal": "NEUTRAL",
            "error": str(exc),
        }

    volume = {}

    try:
        volume = _component(
            analyze_volume(data)
        )
    except Exception as exc:
        volume = {
            "signal": "NEUTRAL",
            "error": str(exc),
        }

    # ========================================================
    # MODULE SIGNALS
    # ========================================================

    structure_signal = _signal_from_component(
        structure
    )

    candle_signal = _signal_from_component(
        candle
    )

    regime_signal = _signal_from_component(
        regime
    )

    volume_signal = _signal_from_component(
        volume
    )

    # ========================================================
    # TREND
    # ========================================================

    buy_score = 0.0
    sell_score = 0.0

    trend_buy = 0.0
    trend_sell = 0.0

    if ema20 > ema50:
        trend_buy += 25

    elif ema20 < ema50:
        trend_sell += 25

    if ema50 > ema200:
        trend_buy += 25

    elif ema50 < ema200:
        trend_sell += 25

    if close > ema20:
        trend_buy += 20

    elif close < ema20:
        trend_sell += 20

    if close > ema200:
        trend_buy += 30

    elif close < ema200:
        trend_sell += 30

    buy_score += trend_buy
    sell_score += trend_sell

    # ========================================================
    # MOMENTUM
    # ========================================================

    momentum_buy = 0.0
    momentum_sell = 0.0

    if rsi >= 55:
        momentum_buy += 50

    elif rsi <= 45:
        momentum_sell += 50

    else:
        momentum_buy += 25
        momentum_sell += 25

    if macd > macd_signal:
        momentum_buy += 50

    elif macd < macd_signal:
        momentum_sell += 50

    else:
        momentum_buy += 25
        momentum_sell += 25

    buy_score += momentum_buy
    sell_score += momentum_sell

    # ========================================================
    # STRUCTURE
    # ========================================================

    structure_score = _score_component(
        structure
    )

    if structure_signal == "BUY":
        buy_score += 100

    elif structure_signal == "SELL":
        sell_score += 100

    else:
        buy_score += 25
        sell_score += 25

    # ========================================================
    # CANDLE
    # ========================================================

    candle_score = _score_component(
        candle
    )

    if candle_signal == "BUY":
        buy_score += 60

    elif candle_signal == "SELL":
        sell_score += 60

    else:
        buy_score += 30
        sell_score += 30

    # ========================================================
    # VOLUME
    # ========================================================

    volume_score = _score_component(
        volume
    )

    if volume_signal == "BUY":
        buy_score += 50

    elif volume_signal == "SELL":
        sell_score += 50

    else:
        buy_score += 25
        sell_score += 25

    # ========================================================
    # REGIME
    # ========================================================

    regime_score = _score_component(
        regime
    )

    if regime_signal == "BUY":
        buy_score += 50

    elif regime_signal == "SELL":
        sell_score += 50

    else:
        buy_score += 25
        sell_score += 25

    # ========================================================
    # LEVELS
    # ========================================================

    levels_signal = _signal_from_component(
        levels
    )

    levels_score = _score_component(
        levels
    )

    if levels_signal == "BUY":

        buy_score += 50

    elif levels_signal == "SELL":

        sell_score += 50

    else:

        # Detect support / resistance fields
        nearest_support = None
        nearest_resistance = None

        if isinstance(levels, dict):

            for key in (
                "nearest_support",
                "support",
                "support_price",
            ):
                if key in levels:
                    nearest_support = _safe(
                        levels.get(key),
                        0.0,
                    )

                    if nearest_support > 0:
                        break

            for key in (
                "nearest_resistance",
                "resistance",
                "resistance_price",
            ):
                if key in levels:
                    nearest_resistance = _safe(
                        levels.get(key),
                        0.0,
                    )

                    if nearest_resistance > 0:
                        break

        if (
            nearest_support
            and close > nearest_support
        ):
            buy_score += 50

        elif (
            nearest_resistance
            and close < nearest_resistance
        ):
            sell_score += 50

        else:
            buy_score += 25
            sell_score += 25

    # ========================================================
    # RAW SIGNAL
    # ========================================================

    if buy_score > sell_score:
        raw_signal = "BUY"

    elif sell_score > buy_score:
        raw_signal = "SELL"

    else:
        raw_signal = "NEUTRAL"

    # ========================================================
    # CONFLICT
    # ========================================================

    module_signals = [
        structure_signal,
        candle_signal,
        volume_signal,
        regime_signal,
    ]

    buy_votes = sum(
        1
        for signal in module_signals
        if signal == "BUY"
    )

    sell_votes = sum(
        1
        for signal in module_signals
        if signal == "SELL"
    )

    directional_votes = (
        buy_votes + sell_votes
    )

    if directional_votes > 0:
        conflict = (
            min(
                buy_votes,
                sell_votes,
            )
            / directional_votes
        )
    else:
        conflict = 0.0

    # ========================================================
    # QUALITY
    # ========================================================

    trend_quality = (
        max(
            trend_buy,
            trend_sell,
        )
        / 100.0
        * 100.0
    )

    momentum_quality = (
        max(
            momentum_buy,
            momentum_sell,
        )
        / 100.0
        * 100.0
    )

    quality = _quality_engine(
        trend_score=trend_quality,
        structure_score=structure_score,
        momentum_score=momentum_quality,
        volume_score=volume_score,
        candle_score=candle_score,
        regime_score=regime_score,
        levels_score=levels_score,
        conflict=conflict,
    )

    # ========================================================
    # DIRECTIONAL EDGE
    # ========================================================

    total_score = (
        buy_score + sell_score
    )

    if total_score > 0:

        directional_edge = (
            abs(
                buy_score - sell_score
            )
            / total_score
        )

    else:
        directional_edge = 0.0

    # ========================================================
    # CONFIDENCE
    # ========================================================

    directional_confidence = (
        directional_edge * 100.0
    )

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

    # ========================================================
    # VALIDATION
    # ========================================================

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
        mtf_bias=mtf_bias,
    )

    final_signal = validation[
        "final_signal"
    ]

    # ========================================================
    # TRADE LEVELS
    # ========================================================

    trade_levels = calculate_trade_levels(
        data,
        final_signal,
    )

    # ========================================================
    # REASONING
    # ========================================================

    reasoning: list[str] = []

    if ema20 > ema50:
        reasoning.append(
            "EMA20 > EMA50: xu hướng tăng"
        )

    elif ema20 < ema50:
        reasoning.append(
            "EMA20 < EMA50: xu hướng giảm"
        )

    else:
        reasoning.append(
            "EMA20 ≈ EMA50: xu hướng chưa rõ"
        )

    if rsi >= 70:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng tăng mạnh"
        )

    elif rsi >= 55:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng tăng"
        )

    elif rsi <= 30:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng giảm mạnh"
        )

    elif rsi <= 45:
        reasoning.append(
            f"RSI {rsi:.1f}: động lượng giảm"
        )

    else:
        reasoning.append(
            f"RSI {rsi:.1f}: trung tính"
        )

    if macd > macd_signal:
        reasoning.append(
            "MACD bullish"
        )

    elif macd < macd_signal:
        reasoning.append(
            "MACD bearish"
        )

    else:
        reasoning.append(
            "MACD neutral"
        )

    reasoning.append(
        f"Market Structure: {structure_signal}"
    )

    reasoning.append(
        f"Nến: {candle_signal}"
    )

    reasoning.append(
        f"Volume: {volume_signal}"
    )

    reasoning.append(
        f"Regime: {regime_signal}"
    )

    if mtf_bias != "NEUTRAL":
        reasoning.append(
            f"MTF Bias: {mtf_bias}"
        )

    for reason in validation[
        "validation_reasons"
    ]:
        reasoning.append(
            f"Validation: {reason}"
        )

    # ========================================================
    # INDICATORS
    # ========================================================

    indicators = {
        "EMA20": _clean_number(
            ema20
        ),
        "EMA50": _clean_number(
            ema50
        ),
        "EMA200": _clean_number(
            ema200
        ),
        "RSI14": _clean_number(
            rsi
        ),
        "MACD": _clean_number(
            macd
        ),
        "MACD_SIGNAL": _clean_number(
            macd_signal
        ),
        "ATR14": _clean_number(
            atr
        ),
    }

    # ========================================================
    # BREAKDOWN
    # ========================================================

    breakdown = {
        "trend": {
            "buy": _clean_number(
                trend_buy
            ),
            "sell": _clean_number(
                trend_sell
            ),
            "signal": (
                "BUY"
                if trend_buy > trend_sell
                else "SELL"
                if trend_sell > trend_buy
                else "NEUTRAL"
            ),
        },
        "momentum": {
            "buy": _clean_number(
                momentum_buy
            ),
            "sell": _clean_number(
                momentum_sell
            ),
            "signal": (
                "BUY"
                if momentum_buy > momentum_sell
                else "SELL"
                if momentum_sell > momentum_buy
                else "NEUTRAL"
            ),
        },
        "structure": structure,
        "candlestick": candle,
        "volume": volume,
        "regime": regime,
        "levels": levels,
    }

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {
        "symbol": str(symbol),
        "timeframe": str(timeframe),

        # Main signal
        "signal": final_signal,
        "final_signal": final_signal,
        "raw_signal": raw_signal,

        # Confidence / quality
        "confidence": _clean_number(
            confidence
        ),
        "signal_quality": _clean_number(
            quality
        ),
        "directional_edge": _clean_number(
            directional_edge
        ),
        "conflict": _clean_number(
            conflict
        ),

        # Scores
        "buy_score": _clean_number(
            buy_score
        ),
        "sell_score": _clean_number(
            sell_score
        ),

        # Price
        "price": _clean_number(
            close
        ),

        # Trade levels
        "entry": trade_levels.get(
            "entry"
        ),
        "sl": trade_levels.get(
            "sl"
        ),
        "tp1": trade_levels.get(
            "tp1"
        ),
        "tp2": trade_levels.get(
            "tp2"
        ),
        "tp3": trade_levels.get(
            "tp3"
        ),
        "rr": trade_levels.get(
            "rr"
        ),
        "trade_levels": trade_levels,

        # Analysis
        "validation": validation,
        "breakdown": breakdown,
        "reasoning": reasoning,
        "indicators": indicators,

        "modules": {
            "structure": structure_signal,
            "candlestick": candle_signal,
            "volume": volume_signal,
            "regime": regime_signal,
            "levels": levels_signal,
        },

        # Metadata
        "analysis_only": True,
        "execution": False,
        "order_placement": False,
        "engine": {
            "version": "6.3",
            "type": "raw_aware_quality_weighted",
        },
    }

    return _json_safe(result)