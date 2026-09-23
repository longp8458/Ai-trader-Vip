# backend/structure.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SWING_LOOKBACK = 3
MIN_SWING_DISTANCE = 0.0


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "bullish",
            "buy",
        }

    return bool(value)


def _empty_result() -> Dict[str, Any]:
    return {
        "trend": "NEUTRAL",
        "bias": "NEUTRAL",

        "bos": "NONE",
        "choch": "NONE",

        "last_event": "NONE",
        "last_event_direction": "NONE",

        "swing_high": None,
        "swing_low": None,

        "last_swing_high": None,
        "last_swing_low": None,

        "previous_swing_high": None,
        "previous_swing_low": None,

        "structure": "NEUTRAL",

        "hh": False,
        "hl": False,
        "lh": False,
        "ll": False,

        "higher_high": False,
        "higher_low": False,
        "lower_high": False,
        "lower_low": False,

        "swing_points": [],
        "recent_swings": [],

        "bullish_structure": False,
        "bearish_structure": False,

        "analysis_only": True,
    }


# ============================================================
# DATA VALIDATION
# ============================================================

def _prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame(data)

    df = data.copy()

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["high", "low"]

    if any(column not in df.columns for column in required):
        return pd.DataFrame()

    # Convert OHLC columns to numeric
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=["high", "low"],
    )

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )
        except Exception:
            pass

    df = df.reset_index(drop=True)

    return df


# ============================================================
# SWING DETECTION
# ============================================================

def _detect_swing_points(
    df: pd.DataFrame,
    lookback: int = DEFAULT_SWING_LOOKBACK,
) -> List[Dict[str, Any]]:

    if df.empty:
        return []

    lookback = max(
        1,
        int(lookback),
    )

    if len(df) < (lookback * 2 + 1):
        return []

    swings: List[Dict[str, Any]] = []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    for i in range(
        lookback,
        len(df) - lookback,
    ):

        current_high = highs[i]
        current_low = lows[i]

        if not (
            np.isfinite(current_high)
            and np.isfinite(current_low)
        ):
            continue

        left_highs = highs[
            i - lookback:i
        ]

        right_highs = highs[
            i + 1:i + lookback + 1
        ]

        left_lows = lows[
            i - lookback:i
        ]

        right_lows = lows[
            i + 1:i + lookback + 1
        ]

        is_swing_high = (
            current_high > np.max(left_highs)
            and current_high >= np.max(right_highs)
        )

        is_swing_low = (
            current_low < np.min(left_lows)
            and current_low <= np.min(right_lows)
        )

        timestamp = None

        if "timestamp" in df.columns:
            timestamp = df.iloc[i]["timestamp"]

            if pd.isna(timestamp):
                timestamp = None

        if is_swing_high:
            swings.append(
                {
                    "index": i,
                    "type": "HIGH",
                    "price": _safe_float(current_high),
                    "timestamp": timestamp,
                }
            )

        if is_swing_low:
            swings.append(
                {
                    "index": i,
                    "type": "LOW",
                    "price": _safe_float(current_low),
                    "timestamp": timestamp,
                }
            )

    swings.sort(
        key=lambda x: x["index"]
    )

    return swings


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def _classify_swings(
    swings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not swings:
        return []

    result: List[Dict[str, Any]] = []

    previous_high: Optional[float] = None
    previous_low: Optional[float] = None

    for swing in swings:

        item = dict(swing)

        price = _safe_float(
            swing.get("price"),
            0.0,
        )

        swing_type = swing.get("type")

        label = None

        if swing_type == "HIGH":

            if previous_high is not None:

                if price > previous_high:
                    label = "HH"
                elif price < previous_high:
                    label = "LH"
                else:
                    label = "EH"

            else:
                label = "H"

            previous_high = price

        elif swing_type == "LOW":

            if previous_low is not None:

                if price > previous_low:
                    label = "HL"
                elif price < previous_low:
                    label = "LL"
                else:
                    label = "EL"

            else:
                label = "L"

            previous_low = price

        item["label"] = label

        result.append(item)

    return result


# ============================================================
# STRUCTURE TREND
# ============================================================

def _determine_trend(
    swings: List[Dict[str, Any]],
) -> str:

    if not swings:
        return "NEUTRAL"

    highs = [
        s for s in swings
        if s.get("type") == "HIGH"
        and s.get("label") in {"HH", "LH"}
    ]

    lows = [
        s for s in swings
        if s.get("type") == "LOW"
        and s.get("label") in {"HL", "LL"}
    ]

    bullish = False
    bearish = False

    if len(highs) >= 2 and len(lows) >= 2:

        latest_highs = highs[-2:]
        latest_lows = lows[-2:]

        bullish = (
            latest_highs[-1]["label"] == "HH"
            and latest_lows[-1]["label"] == "HL"
        )

        bearish = (
            latest_highs[-1]["label"] == "LH"
            and latest_lows[-1]["label"] == "LL"
        )

    if bullish:
        return "BULLISH"

    if bearish:
        return "BEARISH"

    # Fallback: inspect most recent structural labels
    recent_labels = [
        s.get("label")
        for s in swings[-6:]
    ]

    bullish_score = (
        recent_labels.count("HH")
        + recent_labels.count("HL")
    )

    bearish_score = (
        recent_labels.count("LH")
        + recent_labels.count("LL")
    )

    if bullish_score > bearish_score:
        return "BULLISH"

    if bearish_score > bullish_score:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# BOS / CHOCH
# ============================================================

def _detect_structure_events(
    df: pd.DataFrame,
    swings: List[Dict[str, Any]],
    trend: str,
) -> Dict[str, Any]:

    result = {
        "bos": "NONE",
        "choch": "NONE",
        "last_event": "NONE",
        "last_event_direction": "NONE",
    }

    if df.empty or not swings:
        return result

    close_series = pd.to_numeric(
        df["close"],
        errors="coerce",
    ) if "close" in df.columns else None

    if close_series is None:
        return result

    closes = close_series.to_numpy(
        dtype=float,
    )

    if len(closes) == 0:
        return result

    current_close = _safe_float(
        closes[-1],
        0.0,
    )

    high_swings = [
        s for s in swings
        if s.get("type") == "HIGH"
    ]

    low_swings = [
        s for s in swings
        if s.get("type") == "LOW"
    ]

    if not high_swings and not low_swings:
        return result

    latest_high = (
        high_swings[-1]
        if high_swings
        else None
    )

    latest_low = (
        low_swings[-1]
        if low_swings
        else None
    )

    # --------------------------------------------------------
    # Bullish break
    # --------------------------------------------------------

    bullish_break = False

    if latest_high is not None:

        high_price = _safe_float(
            latest_high.get("price"),
        )

        if (
            current_close > high_price
            and latest_high.get("index", -1)
            < len(df) - 1
        ):
            bullish_break = True

    # --------------------------------------------------------
    # Bearish break
    # --------------------------------------------------------

    bearish_break = False

    if latest_low is not None:

        low_price = _safe_float(
            latest_low.get("price"),
        )

        if (
            current_close < low_price
            and latest_low.get("index", -1)
            < len(df) - 1
        ):
            bearish_break = True

    # --------------------------------------------------------
    # Classify BOS / CHoCH
    # --------------------------------------------------------

    if bullish_break:

        result["last_event"] = "BULLISH_BREAK"
        result["last_event_direction"] = "BULLISH"

        if trend == "BEARISH":
            result["choch"] = "BULLISH"
        else:
            result["bos"] = "BULLISH"

    elif bearish_break:

        result["last_event"] = "BEARISH_BREAK"
        result["last_event_direction"] = "BEARISH"

        if trend == "BULLISH":
            result["choch"] = "BEARISH"
        else:
            result["bos"] = "BEARISH"

    return result


# ============================================================
# MAIN STRUCTURE ANALYZER
# ============================================================

def detect_structure(
    data: pd.DataFrame,
    lookback: int = DEFAULT_SWING_LOOKBACK,
) -> Dict[str, Any]:

    empty = _empty_result()

    try:

        df = _prepare_dataframe(data)

        if df.empty:
            return empty

        if len(df) < (lookback * 2 + 5):
            return empty

        swings = _detect_swing_points(
            df,
            lookback=lookback,
        )

        if not swings:
            return empty

        classified = _classify_swings(
            swings
        )

        trend = _determine_trend(
            classified
        )

        events = _detect_structure_events(
            df,
            classified,
            trend,
        )

        high_swings = [
            s for s in classified
            if s.get("type") == "HIGH"
        ]

        low_swings = [
            s for s in classified
            if s.get("type") == "LOW"
        ]

        last_high = (
            high_swings[-1]
            if high_swings
            else None
        )

        previous_high = (
            high_swings[-2]
            if len(high_swings) >= 2
            else None
        )

        last_low = (
            low_swings[-1]
            if low_swings
            else None
        )

        previous_low = (
            low_swings[-2]
            if len(low_swings) >= 2
            else None
        )

        labels = [
            s.get("label")
            for s in classified
        ]

        hh = "HH" in labels[-4:]
        hl = "HL" in labels[-4:]
        lh = "LH" in labels[-4:]
        ll = "LL" in labels[-4:]

        bullish_structure = (
            hh and hl
        )

        bearish_structure = (
            lh and ll
        )

        structure = trend

        # ----------------------------------------------------
        # Bias
        # ----------------------------------------------------

        bias = "NEUTRAL"

        if trend == "BULLISH":
            bias = "BULLISH"

        elif trend == "BEARISH":
            bias = "BEARISH"

        if events["choch"] == "BULLISH":
            bias = "BULLISH"

        elif events["choch"] == "BEARISH":
            bias = "BEARISH"

        # ----------------------------------------------------
        # Recent swings for frontend
        # ----------------------------------------------------

        recent_swings = []

        for swing in classified[-10:]:

            item = {
                "index": swing.get("index"),
                "type": swing.get("type"),
                "label": swing.get("label"),
                "price": _safe_float(
                    swing.get("price")
                ),
            }

            timestamp = swing.get("timestamp")

            if timestamp is not None:
                try:
                    item["timestamp"] = (
                        timestamp.isoformat()
                        if hasattr(timestamp, "isoformat")
                        else str(timestamp)
                    )
                except Exception:
                    item["timestamp"] = str(timestamp)

            recent_swings.append(item)

        # ----------------------------------------------------
        # Main result
        # ----------------------------------------------------

        result = {
            "trend": trend,
            "bias": bias,

            "bos": events["bos"],
            "choch": events["choch"],

            "last_event": events["last_event"],
            "last_event_direction": (
                events["last_event_direction"]
            ),

            "swing_high": (
                _safe_float(last_high["price"])
                if last_high
                else None
            ),

            "swing_low": (
                _safe_float(last_low["price"])
                if last_low
                else None
            ),

            "last_swing_high": (
                _safe_float(last_high["price"])
                if last_high
                else None
            ),

            "last_swing_low": (
                _safe_float(last_low["price"])
                if last_low
                else None
            ),

            "previous_swing_high": (
                _safe_float(previous_high["price"])
                if previous_high
                else None
            ),

            "previous_swing_low": (
                _safe_float(previous_low["price"])
                if previous_low
                else None
            ),

            "structure": structure,

            "hh": hh,
            "hl": hl,
            "lh": lh,
            "ll": ll,

            "higher_high": hh,
            "higher_low": hl,
            "lower_high": lh,
            "lower_low": ll,

            "bullish_structure": bullish_structure,
            "bearish_structure": bearish_structure,

            "swing_points": recent_swings,
            "recent_swings": recent_swings,

            "analysis_only": True,
        }

        return result

    except Exception:
        return empty


# ============================================================
# COMPATIBILITY HELPERS
# ============================================================

def analyze_structure(
    data: pd.DataFrame,
    lookback: int = DEFAULT_SWING_LOOKBACK,
) -> Dict[str, Any]:

    return detect_structure(
        data,
        lookback=lookback,
    )


def get_market_structure(
    data: pd.DataFrame,
    lookback: int = DEFAULT_SWING_LOOKBACK,
) -> Dict[str, Any]:

    return detect_structure(
        data,
        lookback=lookback,
    )


__all__ = [
    "detect_structure",
    "analyze_structure",
    "get_market_structure",
]