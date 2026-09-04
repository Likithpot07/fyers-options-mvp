from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import pandas as pd
from fyers_apiv3 import fyersModel

from config import CLIENT_ID, ACCESS_TOKEN


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))

UNDERLYING = "NSE:NIFTY50-INDEX"

LOOKBACK_DAYS = 30

STRIKES_EACH_SIDE = 5

OUTPUT_PATH = Path("data/raw/candles_1m.parquet")


# --------------------------------------------------
# FYERS CLIENT
# --------------------------------------------------

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    log_path=""
)


# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------

today = datetime.now(IST)

range_to = today.strftime("%Y-%m-%d")

range_from = (
    today - timedelta(days=LOOKBACK_DAYS)
).strftime("%Y-%m-%d")


# --------------------------------------------------
# GET CURRENT OPTION CHAIN
# --------------------------------------------------

chain_request = {
    "symbol": UNDERLYING,
    "strikecount": 10,
    "timestamp": ""
}

chain = fyers.optionchain(data=chain_request)

if chain.get("s") != "ok":
    print("OPTION CHAIN ERROR:", chain)
    raise SystemExit


options = chain["data"]["optionsChain"]

expiry_data = chain["data"].get("expiryData", [])

if expiry_data:
    print("Current expiry:", expiry_data[0].get("date"))


# --------------------------------------------------
# GET NIFTY SPOT
# --------------------------------------------------

spot_row = next(
    x
    for x in options
    if x.get("symbol") == UNDERLYING
)

spot = spot_row["ltp"]

print("NIFTY:", spot)


# --------------------------------------------------
# FIND AVAILABLE STRIKES
# --------------------------------------------------

option_rows = [
    x
    for x in options
    if x.get("option_type") in ("CE", "PE")
    and x.get("strike_price") is not None
]

strikes = sorted(
    set(x["strike_price"] for x in option_rows)
)

atm_strike = min(
    strikes,
    key=lambda strike: abs(strike - spot)
)

print("ATM:", atm_strike)


atm_index = strikes.index(atm_strike)

start = max(
    0,
    atm_index - STRIKES_EACH_SIDE
)

end = atm_index + STRIKES_EACH_SIDE + 1

selected_strikes = strikes[start:end]

print("Selected strikes:", selected_strikes)


# --------------------------------------------------
# SYMBOLS WE WANT
# --------------------------------------------------

selected_options = [
    x
    for x in option_rows
    if x["strike_price"] in selected_strikes
]

symbols = [
    {
        "symbol": UNDERLYING,
        "type": "INDEX",
        "strike": None
    }
]

for option in selected_options:

    symbols.append(
        {
            "symbol": option["symbol"],
            "type": option["option_type"],
            "strike": option["strike_price"]
        }
    )


print(f"\nFetching {len(symbols)} instruments...\n")


# --------------------------------------------------
# FETCH HISTORY
# --------------------------------------------------

all_frames = []


for instrument in symbols:

    symbol = instrument["symbol"]

    print("Fetching:", symbol)

    request = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }

    response = fyers.history(data=request)

    if response.get("s") != "ok":

        print(
            "  SKIPPED:",
            response.get("message")
        )

        continue

    candles = response.get("candles", [])

    if not candles:
        print("  No candles")
        continue


    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp_epoch",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )


    # Keep canonical timestamp in UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp_epoch"],
        unit="s",
        utc=True
    )


    df["symbol"] = symbol

    df["instrument_type"] = instrument["type"]

    df["strike"] = instrument["strike"]


    all_frames.append(df)

    print(f"  {len(df)} candles")

    # Small delay so we don't hammer FYERS
    time.sleep(0.2)


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

if not all_frames:

    print("No data fetched.")
    raise SystemExit


final_df = pd.concat(
    all_frames,
    ignore_index=True
)


final_df = final_df[
    [
        "timestamp",
        "timestamp_epoch",
        "symbol",
        "instrument_type",
        "strike",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
]


final_df = final_df.sort_values(
    ["timestamp", "symbol"]
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


print("\n----------------------------")
print("DATA FETCH COMPLETE")
print("----------------------------")

print("Rows:", len(final_df))
print("Symbols:", final_df["symbol"].nunique())
print("From:", final_df["timestamp"].min())
print("To:", final_df["timestamp"].max())

print("\nSaved:")
print(OUTPUT_PATH)