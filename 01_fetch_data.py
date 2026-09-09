import os
import time
import json
import math
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing from .env")

API_BASE = "https://api.upstox.com"

UNDERLYING = "NSE_INDEX|Nifty 50"

START_DATE = "2026-08-01"
END_DATE = "2026-08-15"

STRIKES_EACH_SIDE = 5

RAW_DIR = Path("data/raw")
CACHE_DIR = RAW_DIR / "historical_cache"

NIFTY_CACHE = CACHE_DIR / "nifty.parquet"
CONTRACT_CACHE = CACHE_DIR / "contracts.json"
PROGRESS_FILE = CACHE_DIR / "progress.json"
OPTIONS_CACHE = CACHE_DIR / "options.parquet"

TEMP_OUTPUT = RAW_DIR / "candles_1m_historical.parquet"
FINAL_OUTPUT = RAW_DIR / "candles_1m.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RATE LIMITING
# ============================================================

# Stay comfortably below the documented limits.
REQUEST_DELAY = 0.45

# Maximum requests we intentionally make in one rolling window.
WINDOW_REQUEST_LIMIT = 1800

WINDOW_SECONDS = 30 * 60

request_times = []


def wait_for_rate_limit():
    global request_times

    now = time.time()

    # Remove requests outside the rolling window.
    request_times = [
        t for t in request_times
        if now - t < WINDOW_SECONDS
    ]

    if len(request_times) >= WINDOW_REQUEST_LIMIT:
        oldest = min(request_times)
        wait_seconds = WINDOW_SECONDS - (now - oldest) + 5

        print(
            f"\nRate-limit safety pause: waiting "
            f"{wait_seconds / 60:.1f} minutes..."
        )

        time.sleep(max(wait_seconds, 1))

        now = time.time()

        request_times = [
            t for t in request_times
            if now - t < WINDOW_SECONDS
        ]

    if request_times:
        elapsed = now - request_times[-1]

        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

    request_times.append(time.time())


# ============================================================
# HTTP
# ============================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json",
})


def upstox_get(path, max_retries=8):
    """
    GET with proper 429 handling.

    429:
      wait progressively, then retry.

    5xx:
      retry.

    4xx other than 429:
      fail immediately.
    """

    url = API_BASE + path

    for attempt in range(max_retries):
        wait_for_rate_limit()

        response = session.get(url, timeout=60)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = 30
            else:
                wait = min(60 * (attempt + 1), 300)

            print(
                f"  HTTP 429; rate limit reached. "
                f"Waiting {wait:.0f}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )

            time.sleep(wait)
            continue

        if 500 <= response.status_code < 600:
            wait = min(5 * (2 ** attempt), 120)

            print(
                f"  HTTP {response.status_code}; "
                f"retrying in {wait}s..."
            )

            time.sleep(wait)
            continue

        raise RuntimeError(
            f"Upstox API HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    raise RuntimeError(
        f"Upstox API failed after {max_retries} retries: {path}"
    )


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)

    tmp.replace(path)


def parse_candles(payload):
    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    rows = []

    for candle in candles:
        if len(candle) < 7:
            continue

        rows.append({
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
            "open_interest": candle[6],
        })

    return rows


# ============================================================
# HISTORICAL EXPIRIES
# ============================================================

print("=" * 70)
print("HISTORICAL NIFTY OPTIONS DATA REBUILD")
print("=" * 70)

print(f"Date range: {START_DATE} -> {END_DATE}")
print(f"Underlying: {UNDERLYING}")
print()


print("Fetching historical expiry dates...")

expiry_path = (
    "/v2/expired-instruments/expiries"
    f"?instrument_key={UNDERLYING}"
)

expiry_payload = upstox_get(expiry_path)

expiry_dates = expiry_payload.get("data", [])

if not expiry_dates:
    raise RuntimeError("No historical expiry dates returned.")

expiry_dates = sorted(
    str(x)[:10]
    for x in expiry_dates
)

expiry_dates = [
    x for x in expiry_dates
    if START_DATE <= x <= END_DATE
]

if not expiry_dates:
    raise RuntimeError("No expiry dates inside requested range.")

print(f"Historical expiries: {len(expiry_dates)}")
print(f"First expiry: {expiry_dates[0]}")
print(f"Last expiry:  {expiry_dates[-1]}")
print()


# ============================================================
# CONTRACT METADATA
# ============================================================

contracts_cache = load_json(CONTRACT_CACHE)

if contracts_cache is None:
    contracts_cache = {}

print("Loading historical option contracts...")

for expiry in expiry_dates:

    if expiry in contracts_cache:
        continue

    print(f"  Contracts: {expiry}")

    path = (
        "/v2/expired-instruments/option/contract"
        f"?instrument_key={UNDERLYING}"
        f"&expiry_date={expiry}"
    )

    payload = upstox_get(path)

    data = payload.get("data", [])

    contracts_cache[expiry] = data

    save_json(CONTRACT_CACHE, contracts_cache)

    print(f"    {len(data)} contracts")


print()
print("Contract metadata ready.")


# ============================================================
# NIFTY HISTORICAL DATA
# ============================================================

def fetch_nifty_day(date_str):

    path = (
        f"/v3/historical-candle/"
        f"{UNDERLYING}/minutes/1/"
        f"{date_str}/{date_str}"
    )

    payload = upstox_get(path)

    rows = parse_candles(payload)

    for row in rows:
        row["timestamp"] = pd.to_datetime(
            row["timestamp"],
            utc=True
        )

    return rows


if NIFTY_CACHE.exists():

    print("Loading cached NIFTY candles...")

    nifty_df = pd.read_parquet(NIFTY_CACHE)

else:

    print()
    print("Downloading historical NIFTY 1-minute candles...")
    print("This creates the trading-day calendar used by the option rebuild.")

    current = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    nifty_rows = []

    while current <= end:

        date_str = current.strftime("%Y-%m-%d")

        # Weekends can be skipped without making an API request.
        if current.weekday() < 5:

            try:
                rows = fetch_nifty_day(date_str)

                if rows:
                    nifty_rows.extend(rows)

                    print(
                        f"  NIFTY {date_str}: "
                        f"{len(rows)} candles"
                    )

            except Exception as exc:
                print(
                    f"  NIFTY {date_str} FAILED: {exc}"
                )

        current += pd.Timedelta(days=1)

    nifty_df = pd.DataFrame(nifty_rows)

    if nifty_df.empty:
        raise RuntimeError("No NIFTY historical candles downloaded.")

    nifty_df["timestamp"] = pd.to_datetime(
        nifty_df["timestamp"],
        utc=True
    )

    nifty_df = (
        nifty_df
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    nifty_df.to_parquet(NIFTY_CACHE, index=False)

print()
print(f"NIFTY candles: {len(nifty_df):,}")


# ============================================================
# TRADING DAYS
# ============================================================

nifty_df["date"] = (
    nifty_df["timestamp"]
    .dt.tz_convert("Asia/Kolkata")
    .dt.strftime("%Y-%m-%d")
)

daily_nifty = (
    nifty_df
    .sort_values("timestamp")
    .groupby("date", as_index=False)
    .first()
)

trading_days = sorted(daily_nifty["date"].tolist())

print(f"Trading days: {len(trading_days)}")


# ============================================================
# EXPIRY SELECTION
# ============================================================

def nearest_expiry_for_day(date_str):

    valid = [
        expiry
        for expiry in expiry_dates
        if expiry >= date_str
    ]

    if not valid:
        return None

    return min(valid)


# ============================================================
# OPTION UNIVERSE
# ============================================================

def build_daily_universe(date_str, nifty_open):

    expiry = nearest_expiry_for_day(date_str)

    if expiry is None:
        return []

    contracts = contracts_cache.get(expiry, [])

    if not contracts:
        return []

    usable = []

    for contract in contracts:

        instrument_key = contract.get("instrument_key")
        strike = contract.get("strike_price")
        option_type = contract.get("instrument_type")

        if (
            instrument_key is None
            or strike is None
            or option_type not in ("CE", "PE")
        ):
            continue

        usable.append({
            "instrument_key": instrument_key,
            "strike": float(strike),
            "option_type": option_type,
            "expiry": expiry,
            "trading_symbol": contract.get(
                "trading_symbol",
                ""
            ),
        })

    if not usable:
        return []

    strikes = sorted({
        x["strike"]
        for x in usable
    })

    # Find the closest available strike to the first
    # NIFTY candle OPEN. This avoids using the 09:15 close.
    atm_strike = min(
        strikes,
        key=lambda s: abs(s - float(nifty_open))
    )

    selected_strikes = sorted(
        strikes,
        key=lambda s: abs(s - atm_strike)
    )[:(STRIKES_EACH_SIDE * 2 + 1)]

    selected_strikes = set(selected_strikes)

    universe = [
        x for x in usable
        if x["strike"] in selected_strikes
    ]

    # Keep only strikes for which BOTH CE and PE exist.
    strike_types = {}

    for x in universe:
        strike_types.setdefault(
            x["strike"],
            set()
        ).add(x["option_type"])

    valid_strikes = {
        strike
        for strike, types in strike_types.items()
        if {"CE", "PE"}.issubset(types)
    }

    universe = [
        x for x in universe
        if x["strike"] in valid_strikes
    ]

    return universe


# ============================================================
# BUILD REQUEST QUEUE
# ============================================================

print()
print("Building option download queue...")

requests_queue = []

for _, row in daily_nifty.iterrows():

    date_str = row["date"]

    # IMPORTANT:
    # first candle OPEN is used for ATM.
    nifty_open = row["open"]

    universe = build_daily_universe(
        date_str,
        nifty_open
    )

    for contract in universe:

        requests_queue.append({
            "date": date_str,
            "expiry": contract["expiry"],
            "instrument_key": contract["instrument_key"],
            "strike": contract["strike"],
            "option_type": contract["option_type"],
            "trading_symbol": contract["trading_symbol"],
        })


print(f"Option requests: {len(requests_queue):,}")


# ============================================================
# RESUME STATE
# ============================================================

progress = load_json(PROGRESS_FILE)

if progress is None:
    progress = {
        "completed": {},
        "failed": {},
    }

completed = progress.setdefault("completed", {})
failed = progress.setdefault("failed", {})

print(f"Previously completed: {len(completed):,}")
print(f"Previously failed:    {len(failed):,}")

remaining = [
    x for x in requests_queue
    if (
        f"{x['date']}|{x['instrument_key']}"
        not in completed
    )
]

print(f"Remaining:             {len(remaining):,}")
print()


# ============================================================
# EXISTING OPTION CACHE
# ============================================================

if OPTIONS_CACHE.exists():
    print("Loading existing option cache...")
    option_cache_df = pd.read_parquet(OPTIONS_CACHE)
else:
    option_cache_df = pd.DataFrame()


# ============================================================
# DOWNLOAD OPTIONS
# ============================================================

new_rows = []

total = len(requests_queue)

completed_count = len(completed)

print("=" * 70)
print("DOWNLOADING OPTION CANDLES")
print("=" * 70)

for item in remaining:

    completed_count += 1

    date_str = item["date"]
    instrument_key = item["instrument_key"]

    cache_key = f"{date_str}|{instrument_key}"

    print(
        f"[{completed_count}/{total}] "
        f"{item['trading_symbol']} | {date_str}"
    )

    encoded_key = requests.utils.quote(
        instrument_key,
        safe=""
    )

    path = (
        f"/v2/expired-instruments/"
        f"historical-candle/"
        f"{encoded_key}/1minute/"
        f"{date_str}/{date_str}"
    )

    try:

        payload = upstox_get(path)

        rows = parse_candles(payload)

        for row in rows:

            row.update({
                "timestamp": pd.to_datetime(
                    row["timestamp"],
                    utc=True
                ),
                "symbol": instrument_key,
                "instrument_type": item["option_type"],
                "strike": item["strike"],
                "expiry": item["expiry"],
            })

            new_rows.append(row)

        completed[cache_key] = {
            "rows": len(rows),
            "date": date_str,
            "instrument_key": instrument_key,
        }

        failed.pop(cache_key, None)

    except Exception as exc:

        print(f"  FAILED: {exc}")

        failed[cache_key] = str(exc)

    # Checkpoint frequently.
    if (
        len(new_rows) >= 25
        or completed_count % 25 == 0
    ):

        if new_rows:

            chunk = pd.DataFrame(new_rows)

            if option_cache_df.empty:
                option_cache_df = chunk
            else:
                option_cache_df = pd.concat(
                    [option_cache_df, chunk],
                    ignore_index=True
                )

            new_rows = []

            option_cache_df = (
                option_cache_df
                .drop_duplicates(
                    subset=[
                        "timestamp",
                        "symbol",
                    ]
                )
                .sort_values(
                    ["timestamp", "symbol"]
                )
                .reset_index(drop=True)
            )

            option_cache_df.to_parquet(
                OPTIONS_CACHE,
                index=False
            )

        save_json(
            PROGRESS_FILE,
            progress
        )

        print(
            f"  CHECKPOINT: "
            f"{len(completed):,} completed | "
            f"{len(failed):,} failed"
        )


# Final checkpoint.

if new_rows:

    chunk = pd.DataFrame(new_rows)

    if option_cache_df.empty:
        option_cache_df = chunk
    else:
        option_cache_df = pd.concat(
            [option_cache_df, chunk],
            ignore_index=True
        )

    option_cache_df = (
        option_cache_df
        .drop_duplicates(
            subset=[
                "timestamp",
                "symbol",
            ]
        )
        .sort_values(
            ["timestamp", "symbol"]
        )
        .reset_index(drop=True)
    )

    option_cache_df.to_parquet(
        OPTIONS_CACHE,
        index=False
    )

save_json(
    PROGRESS_FILE,
    progress
)


# ============================================================
# BUILD FINAL RAW DATASET
# ============================================================

print()
print("=" * 70)
print("BUILDING FINAL RAW DATASET")
print("=" * 70)

if option_cache_df.empty:
    raise RuntimeError(
        "Option cache is empty. No historical option data was downloaded."
    )


# NIFTY INDEX rows.

index_df = nifty_df.copy()

index_df["symbol"] = UNDERLYING
index_df["instrument_type"] = "INDEX"
index_df["strike"] = math.nan
index_df["expiry"] = None


# Normalize option schema.

option_df = option_cache_df.copy()

option_df["instrument_type"] = (
    option_df["instrument_type"]
    .astype(str)
    .str.upper()
)

# Keep only actual option rows.
option_df = option_df[
    option_df["instrument_type"].isin(["CE", "PE"])
].copy()


columns = [
    "timestamp",
    "symbol",
    "instrument_type",
    "strike",
    "expiry",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]


index_df = index_df[columns]
option_df = option_df[columns]


final_df = pd.concat(
    [index_df, option_df],
    ignore_index=True
)

final_df["timestamp"] = pd.to_datetime(
    final_df["timestamp"],
    utc=True
)

final_df = (
    final_df
    .drop_duplicates(
        subset=[
            "timestamp",
            "symbol",
        ]
    )
    .sort_values(
        ["timestamp", "instrument_type", "symbol"]
    )
    .reset_index(drop=True)
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("VALIDATION")
print("-" * 70)

print(
    f"Total rows:       {len(final_df):,}"
)

print(
    f"INDEX rows:       "
    f"{(final_df['instrument_type'] == 'INDEX').sum():,}"
)

print(
    f"CE rows:          "
    f"{(final_df['instrument_type'] == 'CE').sum():,}"
)

print(
    f"PE rows:          "
    f"{(final_df['instrument_type'] == 'PE').sum():,}"
)

print(
    f"Unique symbols:   "
    f"{final_df['symbol'].nunique():,}"
)

print(
    f"Date range:       "
    f"{final_df['timestamp'].min()} "
    f"-> "
    f"{final_df['timestamp'].max()}"
)

print(
    f"Completed requests: {len(completed):,}"
)

print(
    f"Failed requests:    {len(failed):,}"
)


if failed:
    print()
    print(
        "WARNING: Some requests failed. "
        "The cache is preserved and the script can be rerun."
    )


# Write temporary output first.
final_df.to_parquet(
    TEMP_OUTPUT,
    index=False
)

print()
print(f"Temporary output written:")
print(f"  {TEMP_OUTPUT}")
print(
    f"  Size: {TEMP_OUTPUT.stat().st_size / (1024 * 1024):.1f} MB"
)


# ============================================================
# ONLY REPLACE FINAL FILE WHEN DOWNLOAD IS COMPLETE
# ============================================================

if not failed:

    final_df.to_parquet(
        FINAL_OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("HISTORICAL REBUILD COMPLETE")
    print("=" * 70)
    print(f"Final file: {FINAL_OUTPUT}")

else:

    print()
    print("=" * 70)
    print("DOWNLOAD INCOMPLETE — ORIGINAL RAW FILE PRESERVED")
    print("=" * 70)
    print()
    print(
        "Run this script again later. "
        "It will resume from the cache."
    )