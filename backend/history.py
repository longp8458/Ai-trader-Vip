from __future__ import annotations

from pathlib import Path
from threading import RLock
from datetime import datetime, timezone
from typing import Any
import json
import math
import time
import uuid

import pandas as pd

from backend.data_adapter import fetch_market_dataframe

HISTORY_LIMIT = 100
OUTCOME_TIMEFRAME = "M15"
VALID_TIMEFRAMES = {"M1", "M5", "M15", "H1", "H4", "D1", "W1"}
VALID_SIGNALS = {"BUY", "SELL"}

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HISTORY_FILE = DATA_DIR / "signal_history.json"
_LOCK = RLock()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def _now_epoch() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch_to_iso(value: Any) -> str | None:
    try:
        ts = float(value)
        if not math.isfinite(ts):
            return None
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def _load_locked() -> list[dict[str, Any]]:
    _ensure_storage()
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    except Exception:
        return []


def _save_locked(rows: list[dict[str, Any]]) -> None:
    _ensure_storage()
    safe = _json_safe(rows[:HISTORY_LIMIT])
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(HISTORY_FILE)


def _sort_trim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda x: _safe_float(x.get("created_at"), 0) or 0, reverse=True)
    return rows[:HISTORY_LIMIT]


def _same_price(a: Any, b: Any) -> bool:
    fa, fb = _safe_float(a), _safe_float(b)
    if fa is None or fb is None:
        return fa == fb
    tolerance = max(abs(fa), abs(fb), 1.0) * 0.000001
    return abs(fa - fb) <= tolerance


def _normalise_signal(value: Any) -> str:
    value = str(value or "").upper().strip()
    return value if value in VALID_SIGNALS else "NEUTRAL"


def _normalise_timeframe(value: Any) -> str:
    value = str(value or "").upper().strip()
    return value if value in VALID_TIMEFRAMES or value == "MTF" else "MTF"


def get_history(limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), HISTORY_LIMIT))
    with _LOCK:
        rows = _sort_trim(_load_locked())
    return rows[:limit]


def get_stats(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = get_history(HISTORY_LIMIT) if rows is None else rows
    wins = sum(str(x.get("result", "")).upper() == "WIN" for x in rows)
    losses = sum(str(x.get("result", "")).upper() == "LOSS" for x in rows)
    pending = len(rows) - wins - losses
    completed = wins + losses
    win_rate = wins / completed * 100 if completed else 0.0
    buy = sum(str(x.get("signal", "")).upper() == "BUY" for x in rows)
    sell = sum(str(x.get("signal", "")).upper() == "SELL" for x in rows)
    return {"total": len(rows), "completed": completed, "wins": wins, "losses": losses, "pending": pending, "win_rate": round(win_rate, 2), "buy": buy, "sell": sell}


def record_signal(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol", "")).upper().strip()
    signal = _normalise_signal(payload.get("signal"))
    if not symbol:
        raise ValueError("symbol is required")
    if signal not in VALID_SIGNALS:
        raise ValueError("Only BUY or SELL signals can be recorded")

    timeframe = _normalise_timeframe(payload.get("timeframe"))
    entry = _safe_float(payload.get("entry"))
    sl = _safe_float(payload.get("sl", payload.get("stop_loss")))
    tp1 = _safe_float(payload.get("tp1", payload.get("take_profit")))
    confidence = _safe_float(payload.get("confidence"))
    quality = _safe_float(payload.get("signal_quality", payload.get("quality")))
    now, now_iso = _now_epoch(), _now_iso()

    with _LOCK:
        rows = _load_locked()
        for row in rows:
            if str(row.get("symbol", "")).upper() == symbol and str(row.get("signal", "")).upper() == signal and str(row.get("result", "")).upper() == "PENDING":
                if _same_price(row.get("entry"), entry) and _same_price(row.get("sl"), sl) and _same_price(row.get("tp1"), tp1):
                    row.update({"entry": entry, "sl": sl, "tp1": tp1, "confidence": confidence, "signal_quality": quality, "updated_at": now, "updated_at_iso": now_iso})
                    _save_locked(_sort_trim(rows))
                    return {"recorded": False, "created": False, "duplicate": True, "record": row}

        record = {
            "id": uuid.uuid4().hex,
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "confidence": confidence,
            "signal_quality": quality,
            "current": entry,
            "result": "PENDING",
            "result_time": None,
            "result_time_iso": None,
            "created_at": now,
            "created_at_iso": now_iso,
            "updated_at": now,
            "updated_at_iso": now_iso,
            "analysis_only": True,
        }
        rows.insert(0, record)
        _save_locked(_sort_trim(rows))
        return {"recorded": True, "created": True, "duplicate": False, "record": record}


def _timestamp_to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number / 1000.0 if number > 10_000_000_000 else number
    try:
        ts = pd.to_datetime(value, utc=True)
        return None if pd.isna(ts) else float(ts.timestamp())
    except Exception:
        return None


def _candle_time(df: pd.DataFrame, position: int, row: pd.Series) -> float | None:
    for key in ("timestamp", "time", "datetime", "date", "open_time"):
        if key in df.columns:
            value = _timestamp_to_epoch(row.get(key))
            if value is not None:
                return value
    try:
        return _timestamp_to_epoch(df.index[position])
    except Exception:
        return None


def _value(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row.index:
            value = _safe_float(row.get(name))
            if value is not None:
                return value
    return None


def _refresh_one(row: dict[str, Any]) -> bool:
    if str(row.get("result", "PENDING")).upper() in {"WIN", "LOSS"}:
        return False
    symbol = str(row.get("symbol", "")).upper().strip()
    signal = _normalise_signal(row.get("signal"))
    sl, tp1 = _safe_float(row.get("sl")), _safe_float(row.get("tp1"))
    created_at = _safe_float(row.get("created_at"))
    if not symbol or signal not in VALID_SIGNALS or sl is None or tp1 is None or created_at is None:
        return False

    timeframe = _normalise_timeframe(row.get("timeframe"))
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = OUTCOME_TIMEFRAME
    try:
        df = fetch_market_dataframe(symbol, timeframe, limit=300)
    except Exception:
        return False
    if df is None or df.empty:
        return False
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    changed = False
    latest_close = None

    for position, (_, candle) in enumerate(df.iterrows()):
        candle_time = _candle_time(df, position, candle)
        if candle_time is None or candle_time < created_at:
            continue
        high, low, close = _value(candle, "high", "h"), _value(candle, "low", "l"), _value(candle, "close", "c")
        if close is not None:
            latest_close = close
        if high is None or low is None:
            continue

        if signal == "BUY":
            hit_sl, hit_tp = low <= sl, high >= tp1
            if hit_sl and hit_tp:
                continue
            if hit_tp:
                row.update({"result": "WIN", "result_time": candle_time, "result_time_iso": _epoch_to_iso(candle_time), "current": tp1})
                changed = True
                break
            if hit_sl:
                row.update({"result": "LOSS", "result_time": candle_time, "result_time_iso": _epoch_to_iso(candle_time), "current": sl})
                changed = True
                break
        else:
            hit_sl, hit_tp = high >= sl, low <= tp1
            if hit_sl and hit_tp:
                continue
            if hit_tp:
                row.update({"result": "WIN", "result_time": candle_time, "result_time_iso": _epoch_to_iso(candle_time), "current": tp1})
                changed = True
                break
            if hit_sl:
                row.update({"result": "LOSS", "result_time": candle_time, "result_time_iso": _epoch_to_iso(candle_time), "current": sl})
                changed = True
                break

    if latest_close is not None and str(row.get("result", "PENDING")).upper() == "PENDING":
        row["current"] = latest_close
        changed = True
    if changed:
        row["updated_at"] = _now_epoch()
        row["updated_at_iso"] = _now_iso()
    return changed


def refresh_history(limit: int = HISTORY_LIMIT) -> dict[str, Any]:
    limit = max(1, min(int(limit), HISTORY_LIMIT))
    with _LOCK:
        rows = _load_locked()
        for row in rows:
            try:
                _refresh_one(row)
            except Exception:
                continue
        rows = _sort_trim(rows)
        _save_locked(rows)
        selected = rows[:limit]
    return {"history": selected, "stats": get_stats(selected), "limit": limit}


def clear_history() -> dict[str, Any]:
    with _LOCK:
        _save_locked([])
    return {"history": [], "stats": get_stats([]), "limit": HISTORY_LIMIT}
