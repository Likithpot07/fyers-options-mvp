import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing from .env")

UNDERLYING = "NSE_INDEX|Nifty 50"

LOOKBACK_DAYS = 30
STRIKES_EACH_SIDE = 5

OUTPUT_PATH = Path("data/raw/candles_1m.parquet")

API_BASE = "https://api.upstox.com"

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


# --------------------------------------------------
# UPSTOX REST HELPER
# --------------------------------------------------

def upstox_get(path, params=None, timeout=30):

    response = requests.get(
        f"{API_BASE}{path}",
        headers=HEADERS,
        params=params,
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox API failed [{response.status_code}] "
            f"{path}: {response.text}"
        )

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Upstox API error {path}: {payload}"
        )

    return payload


# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------

today = datetime.now(IST)

range_to = today.date()
range_from = (
    today - timedelta(days=LOOKBACK_DAYS)
).date()


# --------------------------------------------------
# GET NIFTY SPOT
# --------------------------------------------------

spot_payload = upstox_get(
    "/v3/market-quote/ltp",
    params={
        "instrument_key": UNDERLYING
    }
)

spot_data = spot_payload.get("data", {})

if not spot_data:
    raise RuntimeError(
        f"No NIFTY LTP returned: {spot_payload}"
    )

spot_row = next(
    iter(
        spot_data.values()
    )
)

spot = float(
    spot_row["last_price"]
)

print("NIFTY:", spot)


# --------------------------------------------------
# GET OPTION CONTRACTS
# --------------------------------------------------

contracts_payload = upstox_get(
    "/v2/option/contract",
    params={
        "instrument_key": UNDERLYING
    }
)

contracts = contracts_payload.get(
    "data",
    []
)

if not contracts:
    raise RuntimeError(
        "Upstox returned no NIFTY option contracts."
    )


# --------------------------------------------------
# CURRENT / NEAREST EXPIRY
# --------------------------------------------------

today_date = today.date()

future_contracts = []

for contract in contracts:

    expiry_text = contract.get(
        "expiry"
    )

    if not expiry_text:
        continue

    expiry_date = datetime.strptime(
        expiry_text,
        "%Y-%m-%d"
    ).date()

    if expiry_date >= today_date:

        future_contracts.append(
            contract
        )


if not future_contracts:
    raise RuntimeError(
        "No non-expired NIFTY option contracts found."
    )


nearest_expiry = min(
    datetime.strptime(
        contract["expiry"],
        "%Y-%m-%d"
    ).date()
    for contract in future_contracts
)


expiry_contracts = [
    contract
    for contract in future_contracts
    if datetime.strptime(
        contract["expiry"],
        "%Y-%m-%d"
    ).date() == nearest_expiry
]


print(
    "Current expiry:",
    nearest_expiry
)


# Store expiry as actual market expiry time:
# 15:30 IST on expiry day.
expiry_ist = datetime(
    year=nearest_expiry.year,
    month=nearest_expiry.month,
    day=nearest_expiry.day,
    hour=15,
    minute=30,
    second=0,
    tzinfo=IST,
)

EXPIRY = pd.Timestamp(
    expiry_ist.astimezone(
        timezone.utc
    )
)


# --------------------------------------------------
# FIND AVAILABLE STRIKES
# --------------------------------------------------

by_strike = {}

for contract in expiry_contracts:

    option_type = contract.get(
        "instrument_type"
    )

    strike = contract.get(
        "strike_price"
    )

    if (
        option_type not in ("CE", "PE")
        or strike is None
    ):
        continue

    strike = float(
        strike
    )

    by_strike.setdefault(
        strike,
        {}
    )[option_type] = contract


# Only use strikes where both CE and PE exist.
strikes = sorted(
    strike
    for strike, pair in by_strike.items()
    if "CE" in pair and "PE" in pair
)


if not strikes:
    raise RuntimeError(
        "Could not find matching CE/PE strikes."
    )


atm_strike = min(
    strikes,
    key=lambda strike: abs(
        strike - spot
    )
)

print(
    "ATM:",
    atm_strike
)


atm_index = strikes.index(
    atm_strike
)

start = max(
    0,
    atm_index - STRIKES_EACH_SIDE
)

end = (
    atm_index
    + STRIKES_EACH_SIDE
    + 1
)

selected_strikes = strikes[
    start:end
]

print(
    "Selected strikes:",
    selected_strikes
)


# --------------------------------------------------
# SYMBOLS WE WANT
# --------------------------------------------------

symbols = [
    {
        "symbol": UNDERLYING,
        "type": "INDEX",
        "strike": None,
        "expiry": pd.NaT,
    }
]


for strike in selected_strikes:

    pair = by_strike[
        strike
    ]

    for option_type in (
        "CE",
        "PE"
    ):

        contract = pair[
            option_type
        ]

        symbols.append(
            {
                "symbol": contract[
                    "instrument_key"
                ],
                "trading_symbol": contract.get(
                    "trading_symbol",
                    contract["instrument_key"],
                ),
                "type": option_type,
                "strike": strike,
                "expiry": EXPIRY,
            }
        )


print(
    f"\nFetching {len(symbols)} instruments...\n"
)


# --------------------------------------------------
# FETCH HISTORY
# --------------------------------------------------

all_frames = []


for instrument in symbols:

    symbol = instrument[
        "symbol"
    ]

    trading_symbol = instrument.get(
        "trading_symbol",
        symbol
    )

    print(
        "Fetching:",
        trading_symbol
    )

    encoded_symbol = quote(
        symbol,
        safe=""
    )

    path = (
        f"/v3/historical-candle/"
        f"{encoded_symbol}/minutes/1/"
        f"{range_to.strftime('%Y-%m-%d')}/"
        f"{range_from.strftime('%Y-%m-%d')}"
    )

    try:

        payload = upstox_get(
            path
        )

    except Exception as exc:

        print(
            "  SKIPPED:",
            exc
        )

        continue


    candles = (
        payload
        .get(
            "data",
            {}
        )
        .get(
            "candles",
            []
        )
    )


    if not candles:

        print(
            "  No candles"
        )

        continue


    # Upstox V3 candle format:
    # [
    #   timestamp,
    #   open,
    #   high,
    #   low,
    #   close,
    #   volume,
    #   open_interest
    # ]
    normalized_rows = [
        row[:7]
        for row in candles
        if len(row) >= 7
    ]


    df = pd.DataFrame(
        normalized_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    )


    # Keep canonical timestamp in UTC.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )


    df = df.dropna(
        subset=[
            "timestamp"
        ]
    )


    # Preserve the old timestamp_epoch column
    # so downstream code/schema remains compatible.
    df["timestamp_epoch"] = (
        df["timestamp"]
        .astype("int64")
        // 1_000_000_000
    )


    df["symbol"] = symbol

    df["instrument_type"] = instrument[
        "type"
    ]

    df["strike"] = instrument[
        "strike"
    ]

    df["expiry"] = instrument[
        "expiry"
    ]


    all_frames.append(
        df
    )


    print(
        f"  {len(df)} candles"
    )


    # Small delay so we don't hammer Upstox.
    time.sleep(
        0.2
    )


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

if not all_frames:

    print(
        "No data fetched."
    )

    raise SystemExit


final_df = pd.concat(
    all_frames,
    ignore_index=True
)


# Ensure expiry remains timezone-aware UTC.
final_df["expiry"] = pd.to_datetime(
    final_df["expiry"],
    utc=True,
    errors="coerce",
)


final_df = final_df[
    [
        "timestamp",
        "timestamp_epoch",
        "symbol",
        "instrument_type",
        "strike",
        "expiry",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
]


final_df = (
    final_df
    .drop_duplicates(
        subset=[
            "timestamp",
            "symbol",
        ],
        keep="last",
    )
    .sort_values(
        [
            "timestamp",
            "symbol",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# SAVE PARQUET
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


final_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


print(
    "\n----------------------------"
)

print(
    "DATA FETCH COMPLETE"
)

print(
    "----------------------------"
)

print(
    "Rows:",
    len(final_df)
)

print(
    "Symbols:",
    final_df["symbol"].nunique()
)

print(
    "From:",
    final_df["timestamp"].min()
)

print(
    "To:",
    final_df["timestamp"].max()
)

print(
    "Option expiry:",
    EXPIRY
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)
