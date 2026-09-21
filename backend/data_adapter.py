import requests
import pandas as pd


# =====================================================
# BINANCE PUBLIC MARKET DATA
# =====================================================

BINANCE_URL = (
    "https://data-api.binance.vision"
    "/api/v3/klines"
)


# =====================================================
# TIMEFRAME
# =====================================================

TIMEFRAME_MAP = {

    "M1": "1m",

    "M5": "5m",

    "M15": "15m",

    "H1": "1h",

    "D1": "1d",

    "W1": "1w"

}


# =====================================================
# CRYPTO SYMBOL
# =====================================================

CRYPTO_SYMBOLS = {

    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",

}


# =====================================================
# FETCH BINANCE KLINES
# =====================================================

def fetch_binance_klines(
    symbol,
    timeframe,
    limit=300
):

    symbol = symbol.upper().strip()


    if timeframe not in TIMEFRAME_MAP:

        raise ValueError(
            f"Timeframe không được hỗ trợ: "
            f"{timeframe}"
        )


    interval = TIMEFRAME_MAP[
        timeframe
    ]


    limit = max(
        1,
        min(
            int(limit),
            1000
        )
    )


    params = {

        "symbol":
            symbol,

        "interval":
            interval,

        "limit":
            limit

    }


    try:

        response = requests.get(

            BINANCE_URL,

            params=params,

            timeout=15

        )

        response.raise_for_status()


    except requests.RequestException as error:

        raise RuntimeError(
            "Không thể kết nối Binance: "
            + str(error)
        )


    try:

        raw_data = response.json()

    except ValueError:

        raise RuntimeError(
            "Binance trả về JSON không hợp lệ."
        )


    if not isinstance(
        raw_data,
        list
    ):

        raise RuntimeError(
            f"Binance API trả về: "
            f"{raw_data}"
        )


    candles = []


    for candle in raw_data:

        if len(candle) < 6:
            continue


        candles.append({

            "timestamp":
                pd.to_datetime(
                    int(candle[0]),
                    unit="ms",
                    utc=True
                ).isoformat(),

            "open":
                float(candle[1]),

            "high":
                float(candle[2]),

            "low":
                float(candle[3]),

            "close":
                float(candle[4]),

            "volume":
                float(candle[5])

        })


    if not candles:

        raise RuntimeError(
            f"Không có dữ liệu cho "
            f"{symbol} {timeframe}."
        )


    return candles


# =====================================================
# DATAFRAME
# =====================================================

def fetch_binance_dataframe(
    symbol,
    timeframe,
    limit=300
):

    candles = fetch_binance_klines(

        symbol,

        timeframe,

        limit

    )


    return pd.DataFrame(
        candles
    )


# =====================================================
# CHECK SYMBOL
# =====================================================

def check_binance_symbol(
    symbol
):

    symbol = symbol.upper().strip()


    url = (
        "https://data-api.binance.vision"
        "/api/v3/exchangeInfo"
    )


    try:

        response = requests.get(

            url,

            params={
                "symbol":
                    symbol
            },

            timeout=15

        )

        response.raise_for_status()

        data = response.json()


    except requests.RequestException as error:

        raise RuntimeError(
            "Không thể kiểm tra symbol: "
            + str(error)
        )


    symbols = data.get(
        "symbols",
        []
    )


    if not symbols:

        return {

            "valid":
                False,

            "symbol":
                symbol,

            "status":
                "NOT_FOUND"

        }


    item = symbols[0]


    return {

        "valid":
            True,

        "symbol":
            item.get("symbol"),

        "status":
            item.get("status"),

        "base_asset":
            item.get("baseAsset"),

        "quote_asset":
            item.get("quoteAsset")

    }