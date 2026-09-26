from __future__ import annotations

"""
TraderAI V6 - Multi-Timeframe Consensus Engine
Version: 6.5

Mục tiêu:
- Phân tích đa khung thời gian.
- BUY / SELL / NEUTRAL phản ánh chặt chẽ hơn dữ liệu.
- Không để raw signal yếu tự biến thành consensus.
- Ưu tiên các timeframe lớn H4 / D1 / W1.
- Không coi timeframe lỗi là NEUTRAL thật.
- Tách raw_consensus / validated_consensus / consensus.
- Không tự đặt lệnh.
"""

from typing import Any, Mapping
import math


# ============================================================
# VERSION
# ============================================================

ENGINE_VERSION = "6.5"


# ============================================================
# TIMEFRAME WEIGHTS
# ============================================================

TIMEFRAME_WEIGHTS: dict[str, float] = {
    "M1": 0.05,
    "M5": 0.10,
    "M15": 0.15,
    "H1": 0.25,
    "H4": 0.10,
    "D1": 0.25,
    "W1": 0.10,
}

SUPPORTED_TIMEFRAMES = list(TIMEFRAME_WEIGHTS.keys())


# ============================================================
# HIGHER TIMEFRAME WEIGHTS
# ============================================================

HTF_WEIGHTS: dict[str, float] = {
    "H4": 0.20,
    "D1": 0.45,
    "W1": 0.35,
}


# ============================================================
# CONSENSUS THRESHOLDS - V6.5
# ============================================================

# Chất lượng tối thiểu của một timeframe
QUALITY_FLOOR = 55.0

# Confidence tối thiểu
CONFIDENCE_FLOOR = 60.0

# Directional edge tối thiểu
MIN_DIRECTIONAL_EDGE = 0.12

# Conflict tối đa của một timeframe
MAX_DIRECTIONAL_CONFLICT = 0.35

# HTF conflict tối đa
HTF_CONFLICT_THRESHOLD = 0.40

# Confidence tối thiểu cho signal validated
MIN_FINAL_CONFIDENCE = 60.0

# Quality tối thiểu cho signal validated
MIN_FINAL_QUALITY = 55.0

# Phải có ít nhất 2 timeframe xác nhận
MIN_VALIDATED_DIRECTIONAL = 2

# Raw consensus phải có hướng đủ mạnh
MIN_RAW_DIRECTIONAL_SCORE = 0.58

# BUY / SELL phải thắng hướng đối diện ít nhất 12%
MIN_DIRECTIONAL_MARGIN = 0.12

# HTF phải ủng hộ ít nhất 55%
MIN_HTF_SUPPORT = 0.55

# Ít nhất 65% tổng trọng số timeframe phải có dữ liệu
MIN_EVIDENCE_WEIGHT = 0.65

# Raw-only bị giới hạn confidence
RAW_ONLY_MIN_CONFIDENCE = 60.0
RAW_ONLY_MAX_CONFIDENCE = 60.0


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Chuyển giá trị thành float an toàn.
    NaN / inf => default.
    """

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _clamp(
    value: Any,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    value = _safe_float(value, low)

    if value < low:
        return low

    if value > high:
        return high

    return value


def _safe_signal(value: Any) -> str:
    """
    Chuẩn hóa signal về:
    BUY / SELL / NEUTRAL
    """

    if value is None:
        return "NEUTRAL"

    signal = str(value).strip().upper()

    if signal in {
        "BUY",
        "LONG",
        "BULLISH",
    }:
        return "BUY"

    if signal in {
        "SELL",
        "SHORT",
        "BEARISH",
    }:
        return "SELL"

    return "NEUTRAL"


def _direction_from_result(
    result: Mapping[str, Any],
) -> str:
    """
    Ưu tiên raw_signal.

    Nếu raw_signal không có hướng,
    dùng signal cuối cùng.
    """

    raw_signal = _safe_signal(
        result.get("raw_signal")
    )

    if raw_signal in {
        "BUY",
        "SELL",
    }:
        return raw_signal

    return _safe_signal(
        result.get("signal")
    )


def _validated_direction(
    result: Mapping[str, Any],
) -> str:
    """
    Chỉ lấy signal cuối cùng sau validation.
    """

    signal = _safe_signal(
        result.get("signal")
    )

    if signal in {
        "BUY",
        "SELL",
    }:
        return signal

    return "NEUTRAL"


def _is_validated(
    result: Mapping[str, Any],
) -> bool:
    """
    Kiểm tra một timeframe có thực sự
    vượt qua validation hay không.
    """

    signal = _validated_direction(
        result
    )

    if signal not in {
        "BUY",
        "SELL",
    }:
        return False

    explicit_validated = result.get(
        "validated"
    )

    if explicit_validated is not None:
        return bool(
            explicit_validated
        )

    quality = _safe_float(
        result.get(
            "signal_quality"
        ),
        0.0,
    )

    confidence = _safe_float(
        result.get(
            "confidence"
        ),
        0.0,
    )

    edge = abs(
        _safe_float(
            result.get(
                "directional_edge"
            ),
            0.0,
        )
    )

    conflict = _safe_float(
        result.get(
            "conflict"
        ),
        1.0,
    )

    return (
        quality
        >= MIN_FINAL_QUALITY
        and confidence
        >= MIN_FINAL_CONFIDENCE
        and edge
        >= MIN_DIRECTIONAL_EDGE
        and conflict
        <= MAX_DIRECTIONAL_CONFLICT
    )


def _has_usable_result(
    result: Any,
) -> bool:
    """
    Kiểm tra timeframe có dữ liệu sử dụng được không.

    Quan trọng:
    timeframe lỗi / timeout / unavailable
    KHÔNG được tính là NEUTRAL thật.
    """

    if not isinstance(
        result,
        Mapping,
    ):
        return False

    if result.get("error"):
        return False

    if result.get(
        "available"
    ) is False:
        return False

    if result.get(
        "status"
    ) in {
        "error",
        "failed",
        "unavailable",
    }:
        return False

    return True


# ============================================================
# INFLUENCE
# ============================================================

def _calculate_influence(
    result: Mapping[str, Any],
) -> float:
    """
    Tính độ tin cậy của một timeframe.

    Dựa trên:
    - confidence
    - signal quality
    - directional edge
    - conflict
    """

    confidence = _clamp(
        _safe_float(
            result.get(
                "confidence"
            ),
            0.0,
        ) / 100.0
    )

    quality = _clamp(
        _safe_float(
            result.get(
                "signal_quality"
            ),
            0.0,
        ) / 100.0
    )

    edge = _clamp(
        abs(
            _safe_float(
                result.get(
                    "directional_edge"
                ),
                0.0,
            )
        )
    )

    conflict = _clamp(
        _safe_float(
            result.get(
                "conflict"
            ),
            0.0,
        )
    )

    conflict_factor = (
        1.0 - conflict
    )

    base = (
        confidence * 0.40
        + quality * 0.35
        + edge * 0.25
    )

    influence = (
        base
        * conflict_factor
    )

    return _clamp(
        influence
    )


def _calculate_effective_weight(
    timeframe: str,
    result: Mapping[str, Any],
) -> float:
    """
    Weight thực tế =
    timeframe weight × influence.
    """

    configured_weight = _safe_float(
        TIMEFRAME_WEIGHTS.get(
            timeframe,
            0.0,
        ),
        0.0,
    )

    influence = _calculate_influence(
        result
    )

    return max(
        0.0,
        configured_weight
        * influence,
    )


# ============================================================
# HIGHER TIMEFRAME ANALYSIS
# ============================================================

def _calculate_htf(
    results: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """
    Phân tích H4 / D1 / W1.

    Nếu một timeframe HTF lỗi,
    trọng số được renormalize trên dữ liệu
    thực sự có.
    """

    buy_score = 0.0
    sell_score = 0.0
    neutral_score = 0.0

    available_weight = 0.0

    details: dict[str, Any] = {}

    for (
        timeframe,
        base_weight,
    ) in HTF_WEIGHTS.items():

        result = results.get(
            timeframe
        )

        if (
            not isinstance(
                result,
                Mapping,
            )
            or not _has_usable_result(
                result
            )
        ):

            details[timeframe] = {
                "available": False,
                "signal": "UNAVAILABLE",
                "weight": base_weight,
                "effective_weight": 0.0,
            }

            continue

        available_weight += (
            base_weight
        )

        raw_signal = (
            _direction_from_result(
                result
            )
        )

        final_signal = (
            _validated_direction(
                result
            )
        )

        influence = (
            _calculate_influence(
                result
            )
        )

        effective_weight = (
            base_weight
            * influence
        )

        details[timeframe] = {
            "available": True,
            "raw_signal": raw_signal,
            "signal": final_signal,
            "validated": _is_validated(
                result
            ),
            "confidence": _safe_float(
                result.get(
                    "confidence"
                ),
                0.0,
            ),
            "quality": _safe_float(
                result.get(
                    "signal_quality"
                ),
                0.0,
            ),
            "directional_edge": _safe_float(
                result.get(
                    "directional_edge"
                ),
                0.0,
            ),
            "conflict": _safe_float(
                result.get(
                    "conflict"
                ),
                0.0,
            ),
            "weight": base_weight,
            "influence": influence,
            "effective_weight": effective_weight,
        }

        if raw_signal == "BUY":

            buy_score += (
                effective_weight
            )

        elif raw_signal == "SELL":

            sell_score += (
                effective_weight
            )

        else:

            neutral_score += (
                effective_weight
            )

    total = (
        buy_score
        + sell_score
        + neutral_score
    )

    if total > 0:

        buy_norm = (
            buy_score / total
        )

        sell_norm = (
            sell_score / total
        )

        neutral_norm = (
            neutral_score / total
        )

    else:

        buy_norm = 0.0
        sell_norm = 0.0
        neutral_norm = 1.0

    directional_total = (
        buy_norm
        + sell_norm
    )

    if directional_total <= 0:

        bias = "NEUTRAL"

    elif buy_norm > sell_norm:

        margin = (
            buy_norm
            - sell_norm
        )

        if (
            buy_norm
            >= MIN_HTF_SUPPORT
            and margin
            >= MIN_DIRECTIONAL_MARGIN
        ):
            bias = "BUY"

        else:
            bias = "NEUTRAL"

    elif sell_norm > buy_norm:

        margin = (
            sell_norm
            - buy_norm
        )

        if (
            sell_norm
            >= MIN_HTF_SUPPORT
            and margin
            >= MIN_DIRECTIONAL_MARGIN
        ):
            bias = "SELL"

        else:
            bias = "NEUTRAL"

    else:

        bias = "NEUTRAL"

    if directional_total > 0:

        conflict = (
            min(
                buy_norm,
                sell_norm,
            )
            / directional_total
        )

    else:

        conflict = 0.0

    if (
        conflict
        > HTF_CONFLICT_THRESHOLD
    ):
        bias = "NEUTRAL"

    return {
        "bias": bias,

        "buy_score": _safe_float(
            buy_norm
        ),

        "sell_score": _safe_float(
            sell_norm
        ),

        "neutral_score": _safe_float(
            neutral_norm
        ),

        "conflict": _safe_float(
            conflict
        ),

        "available_weight": _safe_float(
            available_weight
        ),

        "details": details,
    }


# ============================================================
# PROCESS ONE TIMEFRAME
# ============================================================

def _process_timeframe(
    timeframe: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:

    configured_weight = _safe_float(
        TIMEFRAME_WEIGHTS.get(
            timeframe,
            0.0,
        ),
        0.0,
    )

    raw_signal = (
        _direction_from_result(
            result
        )
    )

    final_signal = (
        _validated_direction(
            result
        )
    )

    validated = _is_validated(
        result
    )

    confidence = _safe_float(
        result.get(
            "confidence"
        ),
        0.0,
    )

    quality = _safe_float(
        result.get(
            "signal_quality"
        ),
        0.0,
    )

    edge = abs(
        _safe_float(
            result.get(
                "directional_edge"
            ),
            0.0,
        )
    )

    conflict = _clamp(
        _safe_float(
            result.get(
                "conflict"
            ),
            0.0,
        )
    )

    influence = (
        _calculate_influence(
            result
        )
    )

    effective_weight = (
        configured_weight
        * influence
    )

    raw_buy = 0.0
    raw_sell = 0.0
    raw_neutral = 0.0

    if raw_signal == "BUY":

        raw_buy = (
            effective_weight
        )

    elif raw_signal == "SELL":

        raw_sell = (
            effective_weight
        )

    else:

        raw_neutral = (
            effective_weight
        )

    validated_buy = 0.0
    validated_sell = 0.0
    validated_neutral = 0.0

    if (
        validated
        and final_signal == "BUY"
    ):

        validated_buy = (
            configured_weight
        )

    elif (
        validated
        and final_signal == "SELL"
    ):

        validated_sell = (
            configured_weight
        )

    else:

        validated_neutral = (
            configured_weight
        )

    raw_only = (
        raw_signal
        in {"BUY", "SELL"}
        and not validated
    )

    if validated:

        reasoning = (
            f"Validated {final_signal}"
        )

    elif raw_only:

        reasoning = (
            f"Raw {raw_signal} "
            "nhưng chưa đủ điều kiện validation"
        )

    else:

        reasoning = (
            "Neutral / chưa có hướng rõ"
        )

    return {
        "timeframe": timeframe,
        "available": True,

        "raw_signal": raw_signal,
        "signal": final_signal,

        "direction": (
            final_signal
            if final_signal
            in {"BUY", "SELL"}
            else raw_signal
        ),

        "validated": validated,
        "raw_only": raw_only,

        "confidence": _safe_float(
            confidence
        ),

        "signal_quality": _safe_float(
            quality
        ),

        "directional_edge": _safe_float(
            edge
        ),

        "conflict": _safe_float(
            conflict
        ),

        "weight": _safe_float(
            configured_weight
        ),

        "influence": _safe_float(
            influence
        ),

        "effective_weight": _safe_float(
            effective_weight
        ),

        "weighted_score": _safe_float(
            effective_weight
            if raw_signal
            in {"BUY", "SELL"}
            else 0.0
        ),

        "buy_score": _safe_float(
            raw_buy
        ),

        "sell_score": _safe_float(
            raw_sell
        ),

        "neutral_score": _safe_float(
            raw_neutral
        ),

        "validated_buy_score": _safe_float(
            validated_buy
        ),

        "validated_sell_score": _safe_float(
            validated_sell
        ),

        "validated_neutral_score": _safe_float(
            validated_neutral
        ),

        "reasoning": reasoning,
    }


# ============================================================
# CONSENSUS DECISION
# ============================================================

def _decide_consensus(
    *,
    raw_buy: float,
    raw_sell: float,
    raw_neutral: float,

    validated_buy: float,
    validated_sell: float,
    validated_neutral: float,

    evidence_weight: float,

    validated_buy_count: int,
    validated_sell_count: int,

    htf: Mapping[str, Any],
) -> dict[str, Any]:

    # --------------------------------------------------------
    # RAW NORMALIZATION
    # --------------------------------------------------------

    total_raw = (
        raw_buy
        + raw_sell
        + raw_neutral
    )

    if total_raw > 0:

        raw_buy_norm = (
            raw_buy / total_raw
        )

        raw_sell_norm = (
            raw_sell / total_raw
        )

        raw_neutral_norm = (
            raw_neutral / total_raw
        )

    else:

        raw_buy_norm = 0.0
        raw_sell_norm = 0.0
        raw_neutral_norm = 1.0

    raw_margin = abs(
        raw_buy_norm
        - raw_sell_norm
    )

    # --------------------------------------------------------
    # RAW CONSENSUS
    # --------------------------------------------------------

    if (
        raw_buy_norm
        >= MIN_RAW_DIRECTIONAL_SCORE
        and raw_margin
        >= MIN_DIRECTIONAL_MARGIN
    ):

        raw_consensus = "BUY"

    elif (
        raw_sell_norm
        >= MIN_RAW_DIRECTIONAL_SCORE
        and raw_margin
        >= MIN_DIRECTIONAL_MARGIN
    ):

        raw_consensus = "SELL"

    else:

        raw_consensus = "NEUTRAL"

    # --------------------------------------------------------
    # VALIDATED NORMALIZATION
    # --------------------------------------------------------

    total_validated = (
        validated_buy
        + validated_sell
        + validated_neutral
    )

    if total_validated > 0:

        validated_buy_norm = (
            validated_buy
            / total_validated
        )

        validated_sell_norm = (
            validated_sell
            / total_validated
        )

        validated_neutral_norm = (
            validated_neutral
            / total_validated
        )

    else:

        validated_buy_norm = 0.0
        validated_sell_norm = 0.0
        validated_neutral_norm = 1.0

    # --------------------------------------------------------
    # VALIDATED CONSENSUS
    # --------------------------------------------------------

    if (
        validated_buy_count
        >= MIN_VALIDATED_DIRECTIONAL
        and validated_buy_norm
        >= MIN_RAW_DIRECTIONAL_SCORE
        and (
            validated_buy_norm
            - validated_sell_norm
        )
        >= MIN_DIRECTIONAL_MARGIN
    ):

        validated_consensus = "BUY"

    elif (
        validated_sell_count
        >= MIN_VALIDATED_DIRECTIONAL
        and validated_sell_norm
        >= MIN_RAW_DIRECTIONAL_SCORE
        and (
            validated_sell_norm
            - validated_buy_norm
        )
        >= MIN_DIRECTIONAL_MARGIN
    ):

        validated_consensus = "SELL"

    else:

        validated_consensus = "NEUTRAL"

    # --------------------------------------------------------
    # HTF SUPPORT
    # --------------------------------------------------------

    htf_bias = str(
        htf.get(
            "bias",
            "NEUTRAL",
        )
    ).upper()

    htf_buy = _safe_float(
        htf.get(
            "buy_score"
        ),
        0.0,
    )

    htf_sell = _safe_float(
        htf.get(
            "sell_score"
        ),
        0.0,
    )

    htf_conflict = _safe_float(
        htf.get(
            "conflict"
        ),
        0.0,
    )

    buy_htf_supported = (
        htf_bias == "BUY"
        and htf_buy
        >= MIN_HTF_SUPPORT
        and htf_conflict
        <= HTF_CONFLICT_THRESHOLD
    )

    sell_htf_supported = (
        htf_bias == "SELL"
        and htf_sell
        >= MIN_HTF_SUPPORT
        and htf_conflict
        <= HTF_CONFLICT_THRESHOLD
    )

    # --------------------------------------------------------
    # FINAL CONSENSUS
    # --------------------------------------------------------

    consensus = "NEUTRAL"

    consensus_mode = (
        "NO_CLEAR_CONSENSUS"
    )

    # --------------------------------------------------------
    # CASE 1
    # Validated consensus
    # --------------------------------------------------------

    if (
        validated_consensus
        == "BUY"
    ):

        consensus = "BUY"

        consensus_mode = (
            "VALIDATED"
        )

    elif (
        validated_consensus
        == "SELL"
    ):

        consensus = "SELL"

        consensus_mode = (
            "VALIDATED"
        )

    # --------------------------------------------------------
    # CASE 2
    # Raw + HTF confirmation
    #
    # Không cho raw BUY/SELL một mình
    # biến thành consensus.
    # --------------------------------------------------------

    elif (
        raw_consensus == "BUY"
        and buy_htf_supported
        and evidence_weight
        >= MIN_EVIDENCE_WEIGHT
    ):

        consensus = "BUY"

        consensus_mode = (
            "RAW_WITH_HTF_CONFIRMATION"
        )

    elif (
        raw_consensus == "SELL"
        and sell_htf_supported
        and evidence_weight
        >= MIN_EVIDENCE_WEIGHT
    ):

        consensus = "SELL"

        consensus_mode = (
            "RAW_WITH_HTF_CONFIRMATION"
        )

    else:

        consensus = "NEUTRAL"

        consensus_mode = (
            "INSUFFICIENT_VALIDATION"
        )

    return {
        "raw_consensus": raw_consensus,

        "validated_consensus": (
            validated_consensus
        ),

        "consensus": consensus,

        "consensus_mode": (
            consensus_mode
        ),

        "raw_buy_score": _safe_float(
            raw_buy_norm
        ),

        "raw_sell_score": _safe_float(
            raw_sell_norm
        ),

        "raw_neutral_score": _safe_float(
            raw_neutral_norm
        ),

        "validated_buy_score": (
            _safe_float(
                validated_buy_norm
            )
        ),

        "validated_sell_score": (
            _safe_float(
                validated_sell_norm
            )
        ),

        "validated_neutral_score": (
            _safe_float(
                validated_neutral_norm
            )
        ),

        "raw_margin": _safe_float(
            raw_margin
        ),

        "htf_bias": htf_bias,

        "htf_buy_supported": (
            buy_htf_supported
        ),

        "htf_sell_supported": (
            sell_htf_supported
        ),

        "evidence_weight": _safe_float(
            evidence_weight
        ),
    }


# ============================================================
# CONFIDENCE
# ============================================================

def _calculate_confidence(
    *,
    consensus: str,
    consensus_mode: str,

    raw_buy: float,
    raw_sell: float,
    raw_neutral: float,

    validated_buy: float,
    validated_sell: float,
    validated_neutral: float,

    validated_count: int,

    htf: Mapping[str, Any],

    evidence_weight: float,
) -> float:

    raw_total = (
        raw_buy
        + raw_sell
        + raw_neutral
    )

    if raw_total > 0:

        raw_buy_norm = (
            raw_buy / raw_total
        )

        raw_sell_norm = (
            raw_sell / raw_total
        )

        raw_neutral_norm = (
            raw_neutral / raw_total
        )

    else:

        raw_buy_norm = 0.0
        raw_sell_norm = 0.0
        raw_neutral_norm = 1.0

    validated_total = (
        validated_buy
        + validated_sell
        + validated_neutral
    )

    if validated_total > 0:

        validated_buy_norm = (
            validated_buy
            / validated_total
        )

        validated_sell_norm = (
            validated_sell
            / validated_total
        )

        validated_neutral_norm = (
            validated_neutral
            / validated_total
        )

    else:

        validated_buy_norm = 0.0
        validated_sell_norm = 0.0
        validated_neutral_norm = 1.0

    htf_bias = str(
        htf.get(
            "bias",
            "NEUTRAL",
        )
    ).upper()

    htf_buy = _safe_float(
        htf.get(
            "buy_score"
        ),
        0.0,
    )

    htf_sell = _safe_float(
        htf.get(
            "sell_score"
        ),
        0.0,
    )

    htf_conflict = _safe_float(
        htf.get(
            "conflict"
        ),
        0.0,
    )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    if consensus in {
        "BUY",
        "SELL",
    }:

        if consensus == "BUY":

            raw_direction = (
                raw_buy_norm
            )

            validated_direction = (
                validated_buy_norm
            )

            htf_support = (
                htf_buy
                if htf_bias == "BUY"
                else 0.0
            )

        else:

            raw_direction = (
                raw_sell_norm
            )

            validated_direction = (
                validated_sell_norm
            )

            htf_support = (
                htf_sell
                if htf_bias == "SELL"
                else 0.0
            )

        if (
            consensus_mode
            == "VALIDATED"
        ):

            confidence = (
                validated_direction
                * 55.0
                + raw_direction
                * 20.0
                + htf_support
                * 15.0
                + _clamp(
                    evidence_weight
                )
                * 10.0
            )

        elif (
            consensus_mode
            == "RAW_WITH_HTF_CONFIRMATION"
        ):

            confidence = (
                raw_direction
                * 45.0
                + htf_support
                * 30.0
                + _clamp(
                    evidence_weight
                )
                * 15.0
                + (
                    1.0
                    - htf_conflict
                )
                * 10.0
            )

            # Raw-only không được confidence quá cao.
            confidence = min(
                confidence,
                RAW_ONLY_MAX_CONFIDENCE,
            )

        else:

            confidence = 40.0

        return round(
            max(
                0.0,
                min(
                    100.0,
                    confidence,
                ),
            ),
            1,
        )

    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------

    directional = max(
        raw_buy_norm,
        raw_sell_norm,
        validated_buy_norm,
        validated_sell_norm,
    )

    disagreement = min(
        raw_buy_norm,
        raw_sell_norm,
    )

    neutral_strength = (
        raw_neutral_norm
    )

    confidence = (
        35.0
        + neutral_strength
        * 30.0
        + disagreement
        * 20.0
        + (
            1.0
            - directional
        )
        * 15.0
    )

    if validated_count >= 2:
        confidence += 5.0

    return round(
        max(
            0.0,
            min(
                100.0,
                confidence,
            ),
        ),
        1,
    )


# ============================================================
# REASONING
# ============================================================

def _build_reasoning(
    *,
    consensus: str,
    raw_consensus: str,
    validated_consensus: str,
    consensus_mode: str,

    raw_buy: float,
    raw_sell: float,
    raw_neutral: float,

    validated_buy_count: int,
    validated_sell_count: int,

    available_timeframes: list[str],
    missing_timeframes: list[str],

    htf: Mapping[str, Any],
) -> list[str]:

    reasons: list[str] = []

    reasons.append(
        f"MTF Consensus: {consensus}"
    )

    reasons.append(
        f"Raw Consensus: {raw_consensus}"
    )

    reasons.append(
        "Validated Consensus: "
        + validated_consensus
    )

    reasons.append(
        "Consensus Mode: "
        + consensus_mode
    )

    reasons.append(
        f"Raw BUY: {raw_buy:.1%}"
    )

    reasons.append(
        f"Raw SELL: {raw_sell:.1%}"
    )

    reasons.append(
        f"Raw NEUTRAL: {raw_neutral:.1%}"
    )

    reasons.append(
        "Validated BUY: "
        + str(
            validated_buy_count
        )
    )

    reasons.append(
        "Validated SELL: "
        + str(
            validated_sell_count
        )
    )

    htf_bias = str(
        htf.get(
            "bias",
            "NEUTRAL",
        )
    ).upper()

    htf_conflict = _safe_float(
        htf.get(
            "conflict"
        ),
        0.0,
    )

    reasons.append(
        f"HTF Bias: {htf_bias}"
    )

    reasons.append(
        f"HTF Conflict: {htf_conflict:.1%}"
    )

    if missing_timeframes:

        reasons.append(
            "Timeframe không khả dụng: "
            + ", ".join(
                missing_timeframes
            )
        )

    reasons.append(
        "Timeframe khả dụng: "
        + ", ".join(
            available_timeframes
        )
    )

    if consensus == "NEUTRAL":

        if raw_consensus in {
            "BUY",
            "SELL",
        }:

            reasons.append(
                "Raw evidence có hướng "
                "nhưng chưa đủ validation "
                "hoặc HTF confirmation."
            )

        else:

            reasons.append(
                "Các timeframe chưa tạo được "
                "ưu thế đủ mạnh cho BUY hoặc SELL."
            )

    elif (
        consensus_mode
        == "VALIDATED"
    ):

        reasons.append(
            "Consensus được xác nhận bởi "
            "nhiều timeframe đã vượt validation."
        )

    elif (
        consensus_mode
        == "RAW_WITH_HTF_CONFIRMATION"
    ):

        reasons.append(
            "Raw evidence mạnh và được "
            "higher timeframe xác nhận."
        )

    return reasons


# ============================================================
# JSON SAFE
# ============================================================

def _json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            _json_safe(
                item
            )
            for item in value
        ]

    try:

        if hasattr(
            value,
            "item",
        ):

            return _json_safe(
                value.item()
            )

    except Exception:
        pass

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        )


# ============================================================
# MAIN MTF FUNCTION
# ============================================================

def calculate_mtf_consensus(
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Tính consensus đa khung thời gian.

    Hỗ trợ:

    calculate_mtf_consensus({
        "M1": {...},
        "M5": {...},
        ...
    })

    hoặc:

    calculate_mtf_consensus({
        "results": {
            "M1": {...},
            ...
        }
    })
    """

    # --------------------------------------------------------
    # Detect nested results
    # --------------------------------------------------------

    if isinstance(
        results,
        Mapping,
    ):

        nested = results.get(
            "results"
        )

        if isinstance(
            nested,
            Mapping,
        ):

            source_results = (
                nested
            )

        else:

            source_results = (
                results
            )

    else:

        source_results = {}

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    timeframe_output: dict[
        str,
        Any,
    ] = {}

    available_timeframes: list[
        str
    ] = []

    missing_timeframes: list[
        str
    ] = []

    raw_buy = 0.0
    raw_sell = 0.0
    raw_neutral = 0.0

    validated_buy = 0.0
    validated_sell = 0.0
    validated_neutral = 0.0

    validated_buy_count = 0
    validated_sell_count = 0

    evidence_weight = 0.0

    raw_counts = {
        "BUY": 0,
        "SELL": 0,
        "NEUTRAL": 0,
    }

    final_counts = {
        "BUY": 0,
        "SELL": 0,
        "NEUTRAL": 0,
    }

    # --------------------------------------------------------
    # Process each timeframe
    # --------------------------------------------------------

    for timeframe in (
        SUPPORTED_TIMEFRAMES
    ):

        raw_result = (
            source_results.get(
                timeframe
            )
        )

        # ----------------------------------------------------
        # Missing / failed
        # ----------------------------------------------------

        if (
            not isinstance(
                raw_result,
                Mapping,
            )
            or not _has_usable_result(
                raw_result
            )
        ):

            missing_timeframes.append(
                timeframe
            )

            timeframe_output[
                timeframe
            ] = {

                "timeframe": (
                    timeframe
                ),

                "available": False,

                "raw_signal": (
                    "UNAVAILABLE"
                ),

                "signal": (
                    "UNAVAILABLE"
                ),

                "direction": (
                    "UNAVAILABLE"
                ),

                "validated": False,

                "raw_only": False,

                "weight": _safe_float(
                    TIMEFRAME_WEIGHTS.get(
                        timeframe,
                        0.0,
                    )
                ),

                "effective_weight": 0.0,

                "weighted_score": 0.0,

                "buy_score": 0.0,

                "sell_score": 0.0,

                "neutral_score": 0.0,

                "validated_buy_score": 0.0,

                "validated_sell_score": 0.0,

                "validated_neutral_score": 0.0,

                "reasoning": (
                    "Timeframe "
                    "không khả dụng."
                ),
            }

            continue

        available_timeframes.append(
            timeframe
        )

        processed = (
            _process_timeframe(
                timeframe,
                raw_result,
            )
        )

        timeframe_output[
            timeframe
        ] = processed

        # ----------------------------------------------------
        # Raw scores
        # ----------------------------------------------------

        raw_buy += _safe_float(
            processed.get(
                "buy_score"
            ),
            0.0,
        )

        raw_sell += _safe_float(
            processed.get(
                "sell_score"
            ),
            0.0,
        )

        raw_neutral += _safe_float(
            processed.get(
                "neutral_score"
            ),
            0.0,
        )

        raw_signal = _safe_signal(
            processed.get(
                "raw_signal"
            )
        )

        if raw_signal in raw_counts:

            raw_counts[
                raw_signal
            ] += 1

        # ----------------------------------------------------
        # Validated scores
        # ----------------------------------------------------

        validated_buy += _safe_float(
            processed.get(
                "validated_buy_score"
            ),
            0.0,
        )

        validated_sell += _safe_float(
            processed.get(
                "validated_sell_score"
            ),
            0.0,
        )

        validated_neutral += _safe_float(
            processed.get(
                "validated_neutral_score"
            ),
            0.0,
        )

        final_signal = _safe_signal(
            processed.get(
                "signal"
            )
        )

        if final_signal in final_counts:

            final_counts[
                final_signal
            ] += 1

        if processed.get(
            "validated"
        ):

            if final_signal == "BUY":

                validated_buy_count += 1

            elif final_signal == "SELL":

                validated_sell_count += 1

        # ----------------------------------------------------
        # Effective evidence
        # ----------------------------------------------------

        evidence_weight += (
            _safe_float(
                processed.get(
                    "effective_weight"
                ),
                0.0,
            )
        )

    # --------------------------------------------------------
    # Evidence coverage
    # --------------------------------------------------------

    evidence_weight = _clamp(
        evidence_weight,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # HTF
    # --------------------------------------------------------

    result_map: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for timeframe in (
        SUPPORTED_TIMEFRAMES
    ):

        result = (
            source_results.get(
                timeframe
            )
        )

        if (
            isinstance(
                result,
                Mapping,
            )
            and _has_usable_result(
                result
            )
        ):

            result_map[
                timeframe
            ] = result

    htf = _calculate_htf(
        result_map
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = _decide_consensus(

        raw_buy=raw_buy,

        raw_sell=raw_sell,

        raw_neutral=raw_neutral,

        validated_buy=validated_buy,

        validated_sell=validated_sell,

        validated_neutral=(
            validated_neutral
        ),

        evidence_weight=(
            evidence_weight
        ),

        validated_buy_count=(
            validated_buy_count
        ),

        validated_sell_count=(
            validated_sell_count
        ),

        htf=htf,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        _calculate_confidence(

            consensus=(
                decision[
                    "consensus"
                ]
            ),

            consensus_mode=(
                decision[
                    "consensus_mode"
                ]
            ),

            raw_buy=raw_buy,

            raw_sell=raw_sell,

            raw_neutral=raw_neutral,

            validated_buy=(
                validated_buy
            ),

            validated_sell=(
                validated_sell
            ),

            validated_neutral=(
                validated_neutral
            ),

            validated_count=(
                validated_buy_count
                + validated_sell_count
            ),

            htf=htf,

            evidence_weight=(
                evidence_weight
            ),
        )
    )

    # --------------------------------------------------------
    # Raw normalized scores
    # --------------------------------------------------------

    total_raw = (
        raw_buy
        + raw_sell
        + raw_neutral
    )

    if total_raw > 0:

        raw_buy_norm = (
            raw_buy
            / total_raw
        )

        raw_sell_norm = (
            raw_sell
            / total_raw
        )

        raw_neutral_norm = (
            raw_neutral
            / total_raw
        )

    else:

        raw_buy_norm = 0.0
        raw_sell_norm = 0.0
        raw_neutral_norm = 1.0

    # --------------------------------------------------------
    # Agreement / conflict
    # --------------------------------------------------------

    directional_total = (
        raw_buy_norm
        + raw_sell_norm
    )

    if directional_total > 0:

        agreement = max(
            raw_buy_norm,
            raw_sell_norm,
        )

        conflict_ratio = (
            min(
                raw_buy_norm,
                raw_sell_norm,
            )
            / directional_total
        )

    else:

        agreement = 0.0
        conflict_ratio = 0.0

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    available_count = len(
        available_timeframes
    )

    missing_count = len(
        missing_timeframes
    )

    total_timeframes = len(
        SUPPORTED_TIMEFRAMES
    )

    coverage = (
        available_count
        / total_timeframes
        if total_timeframes
        else 0.0
    )

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    reasoning = _build_reasoning(

        consensus=(
            decision[
                "consensus"
            ]
        ),

        raw_consensus=(
            decision[
                "raw_consensus"
            ]
        ),

        validated_consensus=(
            decision[
                "validated_consensus"
            ]
        ),

        consensus_mode=(
            decision[
                "consensus_mode"
            ]
        ),

        raw_buy=raw_buy_norm,

        raw_sell=raw_sell_norm,

        raw_neutral=(
            raw_neutral_norm
        ),

        validated_buy_count=(
            validated_buy_count
        ),

        validated_sell_count=(
            validated_sell_count
        ),

        available_timeframes=(
            available_timeframes
        ),

        missing_timeframes=(
            missing_timeframes
        ),

        htf=htf,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = {

        "engine": (
            "raw_aware_quality_weighted_mtf"
        ),

        "engine_version": (
            ENGINE_VERSION
        ),

        "analysis_only": True,

        "order_placement": False,

        "execution_enabled": False,

        # ----------------------------------------------------
        # FINAL CONSENSUS
        # ----------------------------------------------------

        "consensus": (
            decision[
                "consensus"
            ]
        ),

        "confidence": (
            confidence
        ),

        # ----------------------------------------------------
        # CONSENSUS DETAILS
        # ----------------------------------------------------

        "raw_consensus": (
            decision[
                "raw_consensus"
            ]
        ),

        "validated_consensus": (
            decision[
                "validated_consensus"
            ]
        ),

        "consensus_mode": (
            decision[
                "consensus_mode"
            ]
        ),

        # ----------------------------------------------------
        # RAW SCORES
        # ----------------------------------------------------

        "buy_score": _safe_float(
            decision[
                "raw_buy_score"
            ]
        ),

        "sell_score": _safe_float(
            decision[
                "raw_sell_score"
            ]
        ),

        "neutral_score": _safe_float(
            decision[
                "raw_neutral_score"
            ]
        ),

        "raw_buy_score": _safe_float(
            decision[
                "raw_buy_score"
            ]
        ),

        "raw_sell_score": _safe_float(
            decision[
                "raw_sell_score"
            ]
        ),

        "raw_neutral_score": _safe_float(
            decision[
                "raw_neutral_score"
            ]
        ),

        # ----------------------------------------------------
        # VALIDATED SCORES
        # ----------------------------------------------------

        "validated_buy_score": _safe_float(
            decision[
                "validated_buy_score"
            ]
        ),

        "validated_sell_score": _safe_float(
            decision[
                "validated_sell_score"
            ]
        ),

        "validated_neutral_score": _safe_float(
            decision[
                "validated_neutral_score"
            ]
        ),

        # ----------------------------------------------------
        # AGREEMENT
        # ----------------------------------------------------

        "agreement": _safe_float(
            agreement
        ),

        "conflict_ratio": _safe_float(
            conflict_ratio
        ),

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        "counts": final_counts,

        "raw_counts": raw_counts,

        "validated_buy_count": (
            validated_buy_count
        ),

        "validated_sell_count": (
            validated_sell_count
        ),

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        "available_timeframes": (
            available_timeframes
        ),

        "missing_timeframes": (
            missing_timeframes
        ),

        "available_count": (
            available_count
        ),

        "missing_count": (
            missing_count
        ),

        "evidence_weight": (
            _safe_float(
                evidence_weight
            )
        ),

        "data_quality": {

            "available_count": (
                available_count
            ),

            "missing_count": (
                missing_count
            ),

            "coverage": _safe_float(
                coverage
            ),

            "evidence_weight": (
                _safe_float(
                    evidence_weight
                )
            ),

            "missing_timeframes": (
                missing_timeframes
            ),
        },

        # ----------------------------------------------------
        # HIGHER TIMEFRAME
        # ----------------------------------------------------

        "higher_timeframe": htf,

        "htf": htf,

        # ----------------------------------------------------
        # TIMEFRAME DETAILS
        # ----------------------------------------------------

        "timeframes": (
            timeframe_output
        ),

        "details": (
            timeframe_output
        ),

        # ----------------------------------------------------
        # REASONING
        # ----------------------------------------------------

        "reasoning": reasoning,

        # ----------------------------------------------------
        # CONFIG
        # ----------------------------------------------------

        "timeframe_weights": dict(
            TIMEFRAME_WEIGHTS
        ),

        "htf_weights": dict(
            HTF_WEIGHTS
        ),

        "thresholds": {

            "quality_floor": (
                QUALITY_FLOOR
            ),

            "confidence_floor": (
                CONFIDENCE_FLOOR
            ),

            "min_directional_edge": (
                MIN_DIRECTIONAL_EDGE
            ),

            "max_directional_conflict": (
                MAX_DIRECTIONAL_CONFLICT
            ),

            "htf_conflict_threshold": (
                HTF_CONFLICT_THRESHOLD
            ),

            "min_final_confidence": (
                MIN_FINAL_CONFIDENCE
            ),

            "min_final_quality": (
                MIN_FINAL_QUALITY
            ),

            "min_validated_directional": (
                MIN_VALIDATED_DIRECTIONAL
            ),

            "min_raw_directional_score": (
                MIN_RAW_DIRECTIONAL_SCORE
            ),

            "min_directional_margin": (
                MIN_DIRECTIONAL_MARGIN
            ),

            "min_htf_support": (
                MIN_HTF_SUPPORT
            ),

            "min_evidence_weight": (
                MIN_EVIDENCE_WEIGHT
            ),
        },
    }

    return _json_safe(
        output
    )


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def calculate_consensus(
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Alias tương thích.
    """

    return calculate_mtf_consensus(
        results
    )


def analyze_mtf(
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Alias tương thích.
    """

    return calculate_mtf_consensus(
        results
    )