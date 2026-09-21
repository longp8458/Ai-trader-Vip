from datetime import datetime, timezone
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]


def normalize_candles(candles):
    """
    Chuẩn hóa dữ liệu OHLCV thành DataFrame.

    Đây chỉ là Data Engine phân tích,
    không thực hiện giao dịch.
    """

    if not candles:
        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    df = pd.DataFrame(candles)

    # Kiểm tra các cột bắt buộc
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột dữ liệu: "
            + ", ".join(missing)
        )

    # Chỉ giữ các cột cần thiết
    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # Chuẩn hóa timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Chuẩn hóa số
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Xóa dữ liệu lỗi
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Giá phải > 0
    df = df[
        (df["open"] > 0)
        &
        (df["high"] > 0)
        &
        (df["low"] > 0)
        &
        (df["close"] > 0)
        &
        (df["volume"] >= 0)
    ]

    # High phải >= Low
    df = df[
        df["high"] >= df["low"]
    ]

    # Loại timestamp trùng
    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def validate_candles(df):
    """
    Kiểm tra chất lượng dữ liệu OHLCV.
    """

    if df is None or len(df) == 0:

        return {
            "valid": False,
            "rows": 0,
            "errors": [
                "Không có dữ liệu."
            ]
        }

    errors = []

    # Kiểm tra cột
    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            errors.append(
                f"Thiếu cột {column}"
            )

    if errors:

        return {
            "valid": False,
            "rows": len(df),
            "errors": errors
        }

    # Kiểm tra NaN
    if df[
        REQUIRED_COLUMNS
    ].isna().any().any():

        errors.append(
            "Có dữ liệu rỗng hoặc NaN."
        )

    # Kiểm tra giá
    if (
        df["high"]
        <
        df["low"]
    ).any():

        errors.append(
            "Có nến High < Low."
        )

    # Kiểm tra volume
    if (
        df["volume"] < 0
    ).any():

        errors.append(
            "Có volume âm."
        )

    return {
        "valid": len(errors) == 0,
        "rows": len(df),
        "errors": errors
    }


def get_data_metadata(
    df,
    symbol,
    timeframe,
    source="unknown"
):
    """
    Metadata của dữ liệu.
    """

    if df is None or len(df) == 0:

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "rows": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

    return {

        "symbol": symbol,

        "timeframe": timeframe,

        "source": source,

        "rows": len(df),

        "first_timestamp":
            df["timestamp"]
            .iloc[0]
            .isoformat(),

        "last_timestamp":
            df["timestamp"]
            .iloc[-1]
            .isoformat(),

        "retrieved_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }