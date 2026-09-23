"""
NovaTrade AI V6 - Multi-Timeframe Consensus Engine
Analysis-only: no order placement, no broker execution.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAME_WEIGHTS: Dict[str, float] = {
    "M1": 0.05,
    "M5": 0.10,
    "M15": 0.15,
    "H1": 0.25,
    "H4": 0.10,
    "D1": 0.25,
    "W1": 0.10,
}

SUPPORTED_TIMEFRAMES = list(TIMEFRAME_WEIGHTS.keys())

# Higher timeframe influence
HTF_WEIGHTS: Dict[str, float] = {
    "H4": 0.20,
    "D1": 0.45,
    "W1": 0.35,
}

# Minimum quality required before a directional signal contributes strongly
QUALITY_FLOOR = 45.0

# Confidence below this level is treated as weak evidence
CONFIDENCE_FLOOR = 55.0

# Minimum directional edge required
MIN_DIRECTIONAL_EDGE = 0.08

# If BUY and SELL are too close, force NEUTRAL
MAX_DIRECTIONAL_CONFLICT = 0.45

# Higher timeframe conflict threshold
HTF_CONFLICT_THRESHOLD = 0.55

# Minimum final consensus confidence for BUY/SELL
MIN_FINAL_CONFIDENCE = 55.0


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)

        if result != result:
            return default

        if result in (float("inf"), float("-inf")):
            return default

        return result
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clean_signal(value: Any) -> str:
    signal = str(value or "NEUTRAL").upper().strip()

    if signal in {"BUY", "LONG"}:
        return "BUY"

    if signal in {"SELL", "SHORT"}:
        return "SELL"

    return "NEUTRAL"


def _normalize_weights(
    timeframes: Iterable[str] | None = None,
) -> Dict[str, float]:

    selected = list(timeframes or SUPPORTED_TIMEFRAMES)

    selected = [
        tf for tf in selected
        if tf in TIMEFRAME_WEIGHTS
    ]

    if not selected:
        selected = SUPPORTED_TIMEFRAMES.copy()

    total = sum(
        TIMEFRAME_WEIGHTS.get(tf, 0.0)
        for tf in selected
    )

    if total <= 0:
        equal = 1.0 / len(selected)

        return {
            tf: equal
            for tf in selected
        }

    return {
        tf: TIMEFRAME_WEIGHTS[tf] / total
        for tf in selected
    }


def _extract_analysis(result: Any) -> Dict[str, Any]:
    """
    Accept both:

        result = {
            "signal": ...,
            "confidence": ...,
            ...
        }

    and:

        result = {
            "analysis": {
                "signal": ...
            }
        }
    """

    if not isinstance(result, Mapping):
        return {}

    analysis = result.get("analysis")

    if isinstance(analysis, Mapping):
        merged = dict(analysis)

        # Preserve top-level fields if analysis doesn't contain them.
        for key in (
            "signal",
            "confidence",
            "signal_quality",
            "quality",
        ):
            if key not in merged and key in result:
                merged[key] = result[key]

        return merged

    return dict(result)


def _extract_quality(analysis: Mapping[str, Any]) -> float:
    quality = analysis.get("signal_quality")

    if quality is None:
        quality = analysis.get("quality")

    # Older analyzer versions may not expose signal_quality.
    # In that case use confidence as a compatibility fallback.
    if quality is None:
        quality = analysis.get("confidence", 50.0)

    return _clamp(
        _safe_float(quality, 50.0) / 100.0
    ) * 100.0


def _quality_factor(quality: float) -> float:
    """
    Converts quality 0..100 into an influence multiplier.

    < QUALITY_FLOOR:
        weak evidence

    >= QUALITY_FLOOR:
        increasingly strong evidence
    """

    q = _clamp(
        _safe_float(quality, 50.0) / 100.0
    )

    floor = QUALITY_FLOOR / 100.0

    if q <= floor:
        return 0.25 + (q / max(floor, 0.01)) * 0.25

    return 0.50 + (
        (q - floor)
        / max(1.0 - floor, 0.01)
    ) * 0.50


def _confidence_factor(confidence: float) -> float:
    """
    Converts confidence 0..100 into an influence multiplier.
    """

    c = _clamp(
        _safe_float(confidence, 50.0) / 100.0
    )

    floor = CONFIDENCE_FLOOR / 100.0

    if c <= floor:
        return 0.50 + (
            c / max(floor, 0.01)
        ) * 0.20

    return 0.70 + (
        (c - floor)
        / max(1.0 - floor, 0.01)
    ) * 0.30


def _signal_factor(
    signal: str,
    confidence: float,
    quality: float,
) -> float:

    if signal == "NEUTRAL":
        return 0.0

    confidence_factor = _confidence_factor(
        confidence
    )

    quality_factor = _quality_factor(
        quality
    )

    return _clamp(
        confidence_factor * quality_factor
    )


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def _build_timeframe_result(
    timeframe: str,
    raw_result: Any,
    weight: float,
) -> Dict[str, Any]:

    analysis = _extract_analysis(raw_result)

    signal = _clean_signal(
        analysis.get("signal")
    )

    confidence = _safe_float(
        analysis.get("confidence"),
        50.0,
    )

    quality = _extract_quality(
        analysis
    )

    confidence = max(
        0.0,
        min(100.0, confidence)
    )

    quality = max(
        0.0,
        min(100.0, quality)
    )

    influence = _signal_factor(
        signal,
        confidence,
        quality,
    )

    effective_weight = weight * influence

    buy_score = 0.0
    sell_score = 0.0
    neutral_score = 0.0

    if signal == "BUY":
        buy_score = effective_weight

    elif signal == "SELL":
        sell_score = effective_weight

    else:
        neutral_score = weight

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "signal_quality": round(quality, 2),
        "weight": round(weight, 6),
        "influence": round(influence, 6),
        "effective_weight": round(
            effective_weight,
            6,
        ),
        "weighted_score": round(
            effective_weight,
            6,
        ),
        "buy_score": round(
            buy_score,
            6,
        ),
        "sell_score": round(
            sell_score,
            6,
        ),
        "neutral_score": round(
            neutral_score,
            6,
        ),
    }


# ============================================================
# HIGHER TIMEFRAME FILTER
# ============================================================

def _calculate_htf_bias(
    timeframe_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:

    buy = 0.0
    sell = 0.0
    neutral = 0.0

    total = 0.0

    details: Dict[str, Any] = {}

    for timeframe, htf_weight in HTF_WEIGHTS.items():

        item = timeframe_results.get(timeframe)

        if not item:
            continue

        signal = _clean_signal(
            item.get("signal")
        )

        quality = _safe_float(
            item.get("signal_quality"),
            50.0,
        )

        confidence = _safe_float(
            item.get("confidence"),
            50.0,
        )

        q_factor = _quality_factor(
            quality
        )

        c_factor = _confidence_factor(
            confidence
        )

        influence = q_factor * c_factor

        contribution = (
            htf_weight
            * influence
        )

        if signal == "BUY":
            buy += contribution

        elif signal == "SELL":
            sell += contribution

        else:
            neutral += htf_weight

        total += htf_weight

        details[timeframe] = {
            "signal": signal,
            "confidence": round(
                confidence,
                2,
            ),
            "signal_quality": round(
                quality,
                2,
            ),
            "weight": round(
                htf_weight,
                4,
            ),
            "contribution": round(
                contribution,
                6,
            ),
        }

    if total <= 0:
        return {
            "bias": "NEUTRAL",
            "buy_score": 0.0,
            "sell_score": 0.0,
            "neutral_score": 1.0,
            "conflict": 0.0,
            "details": details,
        }

    buy_norm = buy / total
    sell_norm = sell / total
    neutral_norm = neutral / total

    directional_total = (
        buy_norm + sell_norm
    )

    if directional_total <= 0:
        bias = "NEUTRAL"
        conflict = 0.0

    else:
        edge = abs(
            buy_norm - sell_norm
        )

        conflict = 1.0 - (
            edge / directional_total
        )

        if edge < MIN_DIRECTIONAL_EDGE:
            bias = "NEUTRAL"

        elif buy_norm > sell_norm:
            bias = "BUY"

        else:
            bias = "SELL"

    return {
        "bias": bias,
        "buy_score": round(
            buy_norm,
            6,
        ),
        "sell_score": round(
            sell_norm,
            6,
        ),
        "neutral_score": round(
            neutral_norm,
            6,
        ),
        "conflict": round(
            conflict,
            6,
        ),
        "details": details,
    }


# ============================================================
# FINAL CONSENSUS
# ============================================================

def _finalize_consensus(
    buy_score: float,
    sell_score: float,
    neutral_score: float,
    htf: Mapping[str, Any],
) -> Dict[str, Any]:

    directional_total = (
        buy_score
        + sell_score
    )

    if directional_total <= 0:
        return {
            "consensus": "NEUTRAL",
            "confidence": 50.0,
            "reason": "Không có đủ bằng chứng định hướng.",
        }

    edge = abs(
        buy_score - sell_score
    )

    normalized_edge = (
        edge / directional_total
    )

    # --------------------------------------------------------
    # 1. Directional edge too small
    # --------------------------------------------------------

    if normalized_edge < MIN_DIRECTIONAL_EDGE:
        return {
            "consensus": "NEUTRAL",
            "confidence": round(
                50.0
                + normalized_edge * 50.0,
                2,
            ),
            "reason": (
                "BUY và SELL quá cân bằng; "
                "không đủ chênh lệch để xác nhận "
                "tín hiệu định hướng."
            ),
        }

    # --------------------------------------------------------
    # 2. Strong conflict
    # --------------------------------------------------------

    if normalized_edge < MAX_DIRECTIONAL_CONFLICT:
        return {
            "consensus": "NEUTRAL",
            "confidence": round(
                50.0
                + normalized_edge * 50.0,
                2,
            ),
            "reason": (
                "Các timeframe đang xung đột; "
                "ưu tiên NEUTRAL để tránh ép tín hiệu."
            ),
        }

    raw_direction = (
        "BUY"
        if buy_score > sell_score
        else "SELL"
    )

    # --------------------------------------------------------
    # 3. Higher timeframe confirmation
    # --------------------------------------------------------

    htf_bias = str(
        htf.get("bias", "NEUTRAL")
    )

    htf_conflict = _safe_float(
        htf.get("conflict"),
        0.0,
    )

    # Strong HTF conflict:
    # directional signal is suppressed.
    if (
        htf_bias != "NEUTRAL"
        and htf_bias != raw_direction
        and htf_conflict >= HTF_CONFLICT_THRESHOLD
    ):
        return {
            "consensus": "NEUTRAL",
            "confidence": round(
                50.0
                + normalized_edge * 25.0,
                2,
            ),
            "reason": (
                f"Tín hiệu {raw_direction} bị "
                f"lọc bởi xung đột timeframe lớn "
                f"({htf_bias})."
            ),
        }

    # --------------------------------------------------------
    # 4. Calculate confidence
    # --------------------------------------------------------

    base_confidence = (
        50.0
        + normalized_edge * 50.0
    )

    # HTF agreement bonus
    if (
        htf_bias == raw_direction
        and htf_conflict < HTF_CONFLICT_THRESHOLD
    ):
        base_confidence += 7.5

    # HTF neutral -> no bonus
    elif htf_bias == "NEUTRAL":
        base_confidence -= 2.5

    final_confidence = max(
        0.0,
        min(100.0, base_confidence)
    )

    # --------------------------------------------------------
    # 5. Final confidence filter
    # --------------------------------------------------------

    if final_confidence < MIN_FINAL_CONFIDENCE:
        return {
            "consensus": "NEUTRAL",
            "confidence": round(
                final_confidence,
                2,
            ),
            "reason": (
                "Độ tin cậy tổng hợp chưa đạt "
                "ngưỡng xác nhận tín hiệu."
            ),
        }

    return {
        "consensus": raw_direction,
        "confidence": round(
            final_confidence,
            2,
        ),
        "reason": (
            f"Consensus {raw_direction} được xác nhận "
            "bởi trọng số đa khung và chất lượng tín hiệu."
        ),
    }


# ============================================================
# MAIN MTF FUNCTION
# ============================================================

def calculate_mtf_consensus(
    results: Mapping[str, Any],
    timeframes: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Calculate multi-timeframe consensus.

    Parameters
    ----------
    results:
        Dictionary:

        {
            "M1": analysis_result,
            "M5": analysis_result,
            ...
        }

    timeframes:
        Optional timeframe list.

    Returns
    -------
    Dict
        Analysis-only MTF consensus.
    """

    selected = list(
        timeframes
        or SUPPORTED_TIMEFRAMES
    )

    selected = [
        tf for tf in selected
        if tf in TIMEFRAME_WEIGHTS
    ]

    if not selected:
        selected = SUPPORTED_TIMEFRAMES.copy()

    weights = _normalize_weights(
        selected
    )

    timeframe_results: Dict[str, Dict[str, Any]] = {}

    buy_score = 0.0
    sell_score = 0.0
    neutral_score = 0.0

    counts = {
        "BUY": 0,
        "SELL": 0,
        "NEUTRAL": 0,
    }

    errors: Dict[str, str] = {}

    # --------------------------------------------------------
    # Analyze every timeframe
    # --------------------------------------------------------

    for timeframe in selected:

        raw_result = (
            results.get(timeframe)
            if isinstance(results, Mapping)
            else None
        )

        if raw_result is None:
            errors[timeframe] = (
                "Không có dữ liệu phân tích."
            )

            timeframe_results[timeframe] = {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "signal_quality": 0.0,
                "weight": round(
                    weights.get(
                        timeframe,
                        0.0,
                    ),
                    6,
                ),
                "influence": 0.0,
                "effective_weight": 0.0,
                "weighted_score": 0.0,
                "buy_score": 0.0,
                "sell_score": 0.0,
                "neutral_score": round(
                    weights.get(
                        timeframe,
                        0.0,
                    ),
                    6,
                ),
            }

            continue

        try:
            item = _build_timeframe_result(
                timeframe,
                raw_result,
                weights.get(
                    timeframe,
                    0.0,
                ),
            )

            timeframe_results[timeframe] = item

            signal = item["signal"]

            counts[signal] += 1

            buy_score += item["buy_score"]
            sell_score += item["sell_score"]
            neutral_score += item["neutral_score"]

        except Exception as exc:
            errors[timeframe] = str(exc)

            timeframe_results[timeframe] = {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "signal_quality": 0.0,
                "weight": round(
                    weights.get(
                        timeframe,
                        0.0,
                    ),
                    6,
                ),
                "influence": 0.0,
                "effective_weight": 0.0,
                "weighted_score": 0.0,
                "buy_score": 0.0,
                "sell_score": 0.0,
                "neutral_score": round(
                    weights.get(
                        timeframe,
                        0.0,
                    ),
                    6,
                ),
            }

    # --------------------------------------------------------
    # Higher timeframe analysis
    # --------------------------------------------------------

    htf = _calculate_htf_bias(
        timeframe_results
    )

    # --------------------------------------------------------
    # Final consensus
    # --------------------------------------------------------

    final = _finalize_consensus(
        buy_score,
        sell_score,
        neutral_score,
        htf,
    )

    consensus = final["consensus"]
    confidence = final["confidence"]

    total_score = (
        buy_score
        + sell_score
        + neutral_score
    )

    if total_score > 0:
        buy_normalized = (
            buy_score / total_score
        )

        sell_normalized = (
            sell_score / total_score
        )

        neutral_normalized = (
            neutral_score / total_score
        )

    else:
        buy_normalized = 0.0
        sell_normalized = 0.0
        neutral_normalized = 1.0

    # --------------------------------------------------------
    # Conflict ratio
    # --------------------------------------------------------

    directional_total = (
        buy_score
        + sell_score
    )

    if directional_total > 0:
        conflict_ratio = (
            1.0
            - abs(
                buy_score
                - sell_score
            )
            / directional_total
        )
    else:
        conflict_ratio = 0.0

    conflict_ratio = _clamp(
        conflict_ratio
    )

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    reasoning = []

    reasoning.append(
        f"MTF Consensus: {consensus}"
    )

    reasoning.append(
        f"BUY score: {buy_normalized * 100:.1f}%"
    )

    reasoning.append(
        f"SELL score: {sell_normalized * 100:.1f}%"
    )

    reasoning.append(
        f"NEUTRAL score: {neutral_normalized * 100:.1f}%"
    )

    reasoning.append(
        f"Conflict: {conflict_ratio * 100:.1f}%"
    )

    htf_bias = htf.get(
        "bias",
        "NEUTRAL",
    )

    reasoning.append(
        f"HTF bias: {htf_bias}"
    )

    if final.get("reason"):
        reasoning.append(
            str(final["reason"])
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "consensus": consensus,

        "confidence": round(
            confidence,
            2,
        ),

        "buy_score": round(
            buy_normalized,
            6,
        ),

        "sell_score": round(
            sell_normalized,
            6,
        ),

        "neutral_score": round(
            neutral_normalized,
            6,
        ),

        "agreement": round(
            1.0 - conflict_ratio,
            6,
        ),

        "conflict_ratio": round(
            conflict_ratio,
            6,
        ),

        "counts": counts,

        "timeframes": timeframe_results,

        "higher_timeframe": {
            "bias": htf.get(
                "bias",
                "NEUTRAL",
            ),
            "buy_score": htf.get(
                "buy_score",
                0.0,
            ),
            "sell_score": htf.get(
                "sell_score",
                0.0,
            ),
            "neutral_score": htf.get(
                "neutral_score",
                0.0,
            ),
            "conflict": htf.get(
                "conflict",
                0.0,
            ),
            "details": htf.get(
                "details",
                {},
            ),
        },

        "reasoning": reasoning,

        "analysis_only": True,

        "engine": {
            "version": "6.2",
            "type": "quality_weighted_mtf_consensus",
            "execution": False,
            "order_placement": False,
        },

        "errors": errors,
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def analyze_mtf(
    results: Mapping[str, Any],
    timeframes: Iterable[str] | None = None,
) -> Dict[str, Any]:

    return calculate_mtf_consensus(
        results,
        timeframes,
    )


def get_timeframe_weights() -> Dict[str, float]:

    return dict(TIMEFRAME_WEIGHTS)


def get_supported_timeframes() -> list[str]:

    return list(SUPPORTED_TIMEFRAMES)