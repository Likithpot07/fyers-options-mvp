import os
import csv
import json
import time
import atexit
from pathlib import Path
from collections import deque
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import pandas as pd
import requests
import upstox_client
from dotenv import load_dotenv
from xgboost import XGBClassifier

from technical_features import build_features


# ==================================================
# CONFIG
# ==================================================

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing from .env")

INDEX_SYMBOL = "NSE_INDEX|Nifty 50"

CE_MODEL_PATH = Path("models/xgboost_ce_10pct.json")
PE_MODEL_PATH = Path("models/xgboost_pe_10pct.json")

CONFIDENCE_THRESHOLD = 0.85
SETUP_SCORE_THRESHOLD = 6

TARGET_PCT = 0.10
STOP_PCT = 0.10

# Historical candles required for RSI / EMA / VWAP / Pivot / MACD / Fidelity
WARMUP_DAYS = 7

API_BASE = "https://api.upstox.com"

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}

# Persistent live-data storage
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SAVE_TICKS = True
SAVE_CANDLES = True
SAVE_PREDICTIONS = True

FEATURES = [
    "return_1m",
    "return_3m",
    "return_5m",
    "range_pct",
    "body_pct",
    "candle_strength",
    "rsi_14",
    "atr_pct",
    "ema_spread_pct",
    "trend_up",
    "volume_ratio_5",
    "volume_ratio_20",
    "distance_from_high_5",
    "distance_from_low_5",
    "distance_from_vwap_pct",
    "breakout_up_5",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "lvg_score",
    "lvg_detected",
    "time_to_expiry_minutes",
    "time_to_expiry_days",
    "theta",
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_location_in_range",
    "range_vs_prev",
    "range_compression_3",
    "inside_bar",
    "nr4",
    "higher_highs_3",
    "lower_lows_3",
    "distance_to_resistance_15",
    "distance_to_support_15",
    "breakout_strength_up_5",
    "breakout_strength_down_5",
    "breakout_confirm_up_5",
    "breakout_confirm_down_5",
    "false_breakout_up_5",
    "false_breakout_down_5",
    "fidelity_score",
    "fidelity_confirmed",
    "nifty_return_1m",
    "nifty_return_3m",
    "nifty_return_5m",
    "nifty_range_pct",
    "nifty_rsi_14",
    "nifty_atr_pct",
    "nifty_ema_spread_pct",
    "nifty_trend_up",
    "nifty_trend_down",
    "nifty_breakout_up_15",
    "nifty_breakout_down_15",
    "nifty_body_to_range_ratio",
    "nifty_upper_wick_ratio",
    "nifty_lower_wick_ratio",
    "nifty_close_location_in_range",
    "nifty_range_vs_prev",
    "nifty_range_compression_3",
    "nifty_inside_bar",
    "nifty_nr4",
    "nifty_higher_highs_3",
    "nifty_lower_lows_3",
    "nifty_distance_to_resistance_15",
    "nifty_distance_to_support_15",
    "nifty_breakout_strength_up_5",
    "nifty_breakout_strength_down_5",
    "nifty_breakout_confirm_up_5",
    "nifty_breakout_confirm_down_5",
    "nifty_false_breakout_up_5",
    "nifty_false_breakout_down_5",
    "distance_from_pivot_pct",
    "distance_from_strike_pct",
    "premium_pct_of_spot",
    "ce_pe_ratio",
    "ce_return_3m",
    "pe_return_3m",
    "ce_minus_pe_return_3m",
    "setup_score",
]


# ==================================================
# GLOBAL STATE
# ==================================================

candles = {}
latest = {}
market_df = pd.DataFrame()

last_prediction_minute = None
latest_predictions = {"CE": None, "PE": None}
latest_signal = None
last_display = 0

CE_SYMBOL = None
PE_SYMBOL = None
CE_TRADING_SYMBOL = None
PE_TRADING_SYMBOL = None
atm_strike = None
selected_expiry = None

# Ignore first partial minute after startup
startup_now = datetime.now(IST)
first_full_minute = startup_now.replace(second=0, microsecond=0) + timedelta(minutes=1)


# ==================================================
# LATENCY / FEED HEALTH
# ==================================================

feed_ages = deque(maxlen=1000)
tick_gaps = deque(maxlen=1000)

current_feed_age_ms = None
current_tick_gap_ms = None

last_receive_ns = None
last_provider_time = None
last_receive_time = None


# ==================================================
# PERSISTENT CSV STORAGE
# ==================================================

class DailyCSVLogger:
    def __init__(self, subdir, suffix, fieldnames):
        self.directory = DATA_DIR / subdir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.fieldnames = fieldnames
        self.current_date = None
        self.file_handle = None
        self.writer = None

    def _ensure_open(self):
        date_text = datetime.now(IST).strftime("%Y-%m-%d")

        if self.file_handle is not None and self.current_date == date_text:
            return

        self.close()

        path = self.directory / f"{date_text}_{self.suffix}.csv"
        has_data = path.exists() and path.stat().st_size > 0

        # Line-buffered append: each CSV row is flushed promptly without
        # reopening the file for every market tick.
        self.file_handle = path.open(
            "a",
            newline="",
            encoding="utf-8",
            buffering=1,
        )
        self.writer = csv.DictWriter(
            self.file_handle,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )

        if not has_data:
            self.writer.writeheader()

        self.current_date = date_text

    def write(self, row):
        self._ensure_open()
        self.writer.writerow(row)

    def close(self):
        if self.file_handle is not None:
            try:
                self.file_handle.flush()
                self.file_handle.close()
            finally:
                self.file_handle = None
                self.writer = None
                self.current_date = None


TICK_FIELDS = [
    "receive_timestamp_ist",
    "provider_timestamp_ist",
    "event_timestamp_ist",
    "last_trade_timestamp_ist",
    "instrument_key",
    "trading_symbol",
    "instrument_type",
    "strike",
    "expiry",
    "ltp",
    "ltq",
    "close_price",
    "atp",
    "cumulative_volume",
    "open_interest",
    "total_buy_qty",
    "total_sell_qty",
    "iv",
    "delta",
    "theta",
    "gamma",
    "vega",
    "rho",
    "feed_age_ms",
    "tick_gap_ms",
]

for level in range(1, 6):
    TICK_FIELDS.extend(
        [
            f"bid_price_{level}",
            f"bid_qty_{level}",
            f"ask_price_{level}",
            f"ask_qty_{level}",
        ]
    )

CANDLE_FIELDS = [
    "timestamp_utc",
    "timestamp_ist",
    "instrument_key",
    "trading_symbol",
    "instrument_type",
    "strike",
    "expiry",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

PREDICTION_FIELDS = [
    "minute_utc",
    "minute_ist",
    "option_type",
    "instrument_key",
    "trading_symbol",
    "strike",
    "expiry",
    "probability",
    "setup_score",
    "fidelity_score",
    "fidelity_confirmed",
    "lvg_score",
    "lvg_detected",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "theta",
    "time_to_expiry_minutes",
    "time_to_expiry_days",
    "approved",
    "selected",
    "signal",
    "price",
    "target",
    "stop",
    "confidence_threshold",
    "setup_score_threshold",
]

tick_logger = DailyCSVLogger("ticks", "ticks", TICK_FIELDS)
candle_logger = DailyCSVLogger("candles", "candles", CANDLE_FIELDS)
prediction_logger = DailyCSVLogger(
    "predictions",
    "predictions",
    PREDICTION_FIELDS,
)


def close_loggers():
    tick_logger.close()
    candle_logger.close()
    prediction_logger.close()


atexit.register(close_loggers)


def instrument_metadata(symbol):
    if symbol == INDEX_SYMBOL:
        return "NIFTY 50", "INDEX", 0, ""
    if symbol == CE_SYMBOL:
        return CE_TRADING_SYMBOL, "CE", atm_strike, selected_expiry
    if symbol == PE_SYMBOL:
        return PE_TRADING_SYMBOL, "PE", atm_strike, selected_expiry
    return symbol, "UNKNOWN", "", ""


def save_tick(tick):
    if not SAVE_TICKS:
        return

    trading_symbol, instrument_type, strike, expiry = instrument_metadata(
        tick["symbol"]
    )

    row = {
        "receive_timestamp_ist": (
            last_receive_time.isoformat() if last_receive_time else ""
        ),
        "provider_timestamp_ist": (
            last_provider_time.isoformat() if last_provider_time else ""
        ),
        "event_timestamp_ist": tick["event_time"].isoformat(),
        "last_trade_timestamp_ist": (
            tick["last_trade_time"].isoformat()
            if tick.get("last_trade_time")
            else ""
        ),
        "instrument_key": tick["symbol"],
        "trading_symbol": trading_symbol,
        "instrument_type": instrument_type,
        "strike": strike,
        "expiry": expiry,
        "ltp": tick.get("ltp"),
        "ltq": tick.get("ltq"),
        "close_price": tick.get("close_price"),
        "atp": tick.get("atp"),
        "cumulative_volume": tick.get("vol_traded_today"),
        "open_interest": tick.get("open_interest"),
        "total_buy_qty": tick.get("total_buy_qty"),
        "total_sell_qty": tick.get("total_sell_qty"),
        "iv": tick.get("iv"),
        "delta": tick.get("delta"),
        "theta": tick.get("theta"),
        "gamma": tick.get("gamma"),
        "vega": tick.get("vega"),
        "rho": tick.get("rho"),
        "feed_age_ms": (
            round(current_feed_age_ms, 3)
            if current_feed_age_ms is not None
            else ""
        ),
        "tick_gap_ms": (
            round(current_tick_gap_ms, 3)
            if current_tick_gap_ms is not None
            else ""
        ),
    }

    for level in range(1, 6):
        depth = tick.get("depth", {}).get(level, {})
        row[f"bid_price_{level}"] = depth.get("bid_price", "")
        row[f"bid_qty_{level}"] = depth.get("bid_qty", "")
        row[f"ask_price_{level}"] = depth.get("ask_price", "")
        row[f"ask_qty_{level}"] = depth.get("ask_qty", "")

    tick_logger.write(row)


def save_closed_candle(symbol, candle, instrument_type, strike):
    if not SAVE_CANDLES:
        return

    trading_symbol, _, _, expiry = instrument_metadata(symbol)
    timestamp_ist = candle["minute"]
    timestamp_utc = pd.Timestamp(timestamp_ist).tz_convert("UTC")

    candle_logger.write(
        {
            "timestamp_utc": timestamp_utc.isoformat(),
            "timestamp_ist": timestamp_ist.isoformat(),
            "instrument_key": symbol,
            "trading_symbol": trading_symbol,
            "instrument_type": instrument_type,
            "strike": strike,
            "expiry": expiry,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        }
    )


def save_prediction(minute, prediction):
    if not SAVE_PREDICTIONS:
        return

    minute_utc = pd.Timestamp(minute).tz_convert("UTC")
    option_type = prediction["option_type"]

    if option_type == "CE":
        instrument_key = CE_SYMBOL
        trading_symbol = CE_TRADING_SYMBOL
    else:
        instrument_key = PE_SYMBOL
        trading_symbol = PE_TRADING_SYMBOL

    prediction_logger.write(
        {
            "minute_utc": minute_utc.isoformat(),
            "minute_ist": minute.isoformat(),
            "option_type": option_type,
            "instrument_key": instrument_key,
            "trading_symbol": trading_symbol,
            "strike": atm_strike,
            "expiry": selected_expiry,
            "probability": prediction["probability"],
            "setup_score": prediction["setup_score"],
            "fidelity_score": prediction.get("fidelity_score", ""),
            "fidelity_confirmed": prediction.get("fidelity_confirmed", ""),
            "lvg_score": prediction.get("lvg_score", ""),
            "lvg_detected": prediction.get("lvg_detected", ""),
            "macd_line": prediction.get("macd_line", ""),
            "macd_signal": prediction.get("macd_signal", ""),
            "macd_histogram": prediction.get("macd_histogram", ""),
            "theta": prediction.get("theta", ""),
            "time_to_expiry_minutes": prediction.get("time_to_expiry_minutes", ""),
            "time_to_expiry_days": prediction.get("time_to_expiry_days", ""),
            "approved": prediction["approved"],
            "selected": prediction.get("selected", False),
            "signal": (
                f"BUY_{option_type}"
                if prediction.get("selected", False)
                else "NO_TRADE"
            ),
            "price": prediction["price"],
            "target": prediction["target"],
            "stop": prediction["stop"],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "setup_score_threshold": SETUP_SCORE_THRESHOLD,
        }
    )



def update_latency(message):
    """Measure local receive time vs Upstox currentTs.

    currentTs is in milliseconds and represents the timestamp attached to the
    current Upstox feed update. This is a transport/processing estimate, not a
    pure exchange-to-client latency measurement.
    """
    global current_feed_age_ms
    global current_tick_gap_ms
    global last_receive_ns
    global last_provider_time
    global last_receive_time

    receive_ns = time.time_ns()
    receive_ms = receive_ns / 1_000_000

    provider_ms = message.get("currentTs")

    if provider_ms:
        try:
            provider_ms = float(provider_ms)
            current_feed_age_ms = receive_ms - provider_ms

            if current_feed_age_ms >= 0:
                feed_ages.append(current_feed_age_ms)

            last_provider_time = datetime.fromtimestamp(provider_ms / 1000, IST)
            last_receive_time = datetime.fromtimestamp(receive_ms / 1000, IST)
        except (TypeError, ValueError):
            pass

    if last_receive_ns is not None:
        current_tick_gap_ms = (receive_ns - last_receive_ns) / 1_000_000
        tick_gaps.append(current_tick_gap_ms)

    last_receive_ns = receive_ns


# ==================================================
# LOAD XGBOOST MODELS
# ==================================================

for model_path in [CE_MODEL_PATH, PE_MODEL_PATH]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

ce_model = XGBClassifier()
ce_model.load_model(CE_MODEL_PATH)

pe_model = XGBClassifier()
pe_model.load_model(PE_MODEL_PATH)

print("CE XGBoost model loaded.")
print("PE XGBoost model loaded.")


# ==================================================
# UPSTOX REST HELPERS
# ==================================================

def upstox_get(path, params=None, timeout=20):
    response = requests.get(
        f"{API_BASE}{path}",
        headers=HEADERS,
        params=params,
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox API failed [{response.status_code}] {path}: {response.text}"
        )

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"Upstox API error {path}: {payload}")

    return payload


def get_spot_ltp():
    payload = upstox_get(
        "/v3/market-quote/ltp",
        params={"instrument_key": INDEX_SYMBOL},
    )

    data = payload.get("data", {})
    if not data:
        raise RuntimeError(f"No NIFTY LTP returned: {payload}")

    first_quote = next(iter(data.values()))
    return float(first_quote["last_price"])


# ==================================================
# FIND CURRENT EXPIRY + ATM CE / PE
# ==================================================

def find_atm_contracts(spot):
    payload = upstox_get(
        "/v2/option/contract",
        params={"instrument_key": INDEX_SYMBOL},
    )

    contracts = payload.get("data", [])
    if not contracts:
        raise RuntimeError("Upstox returned no NIFTY option contracts.")

    today = datetime.now(IST).date()

    future_contracts = []
    for contract in contracts:
        expiry_text = contract.get("expiry")
        if not expiry_text:
            continue

        expiry = datetime.strptime(expiry_text, "%Y-%m-%d").date()
        if expiry >= today:
            future_contracts.append(contract)

    if not future_contracts:
        raise RuntimeError("No non-expired NIFTY option contracts found.")

    nearest_expiry = min(
        datetime.strptime(c["expiry"], "%Y-%m-%d").date()
        for c in future_contracts
    )

    expiry_contracts = [
        c
        for c in future_contracts
        if datetime.strptime(c["expiry"], "%Y-%m-%d").date() == nearest_expiry
    ]

    by_strike = {}
    for contract in expiry_contracts:
        strike = float(contract["strike_price"])
        option_type = contract.get("instrument_type")

        if option_type not in ("CE", "PE"):
            continue

        by_strike.setdefault(strike, {})[option_type] = contract

    valid_strikes = [
        strike
        for strike, pair in by_strike.items()
        if "CE" in pair and "PE" in pair
    ]

    if not valid_strikes:
        raise RuntimeError("Could not find matching CE/PE strikes for nearest expiry.")

    selected_strike = min(valid_strikes, key=lambda strike: abs(strike - spot))
    pair = by_strike[selected_strike]

    return nearest_expiry, selected_strike, pair["CE"], pair["PE"]


spot = get_spot_ltp()
selected_expiry, atm_strike, ce_contract, pe_contract = find_atm_contracts(spot)

CE_SYMBOL = ce_contract["instrument_key"]
PE_SYMBOL = pe_contract["instrument_key"]
CE_TRADING_SYMBOL = ce_contract.get("trading_symbol", CE_SYMBOL)
PE_TRADING_SYMBOL = pe_contract.get("trading_symbol", PE_SYMBOL)

print()
print("NIFTY  :", spot)
print("EXPIRY :", selected_expiry)
print("ATM    :", atm_strike)
print("CE     :", CE_TRADING_SYMBOL)
print("PE     :", PE_TRADING_SYMBOL)
print()

# Save one small metadata file per process start so each CSV session can be
# traced back to the exact contracts/model configuration used.
session_dir = DATA_DIR / "sessions"
session_dir.mkdir(parents=True, exist_ok=True)
session_start = datetime.now(IST)
session_path = session_dir / (
    f"{session_start.strftime('%Y-%m-%d_%H%M%S')}_session.json"
)

session_path.write_text(
    json.dumps(
        {
            "started_at_ist": session_start.isoformat(),
            "index_instrument_key": INDEX_SYMBOL,
            "spot_at_start": spot,
            "expiry": str(selected_expiry),
            "atm_strike": atm_strike,
            "ce_instrument_key": CE_SYMBOL,
            "ce_trading_symbol": CE_TRADING_SYMBOL,
            "pe_instrument_key": PE_SYMBOL,
            "pe_trading_symbol": PE_TRADING_SYMBOL,
            "ce_model_path": str(CE_MODEL_PATH),
            "pe_model_path": str(PE_MODEL_PATH),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "setup_score_threshold": SETUP_SCORE_THRESHOLD,
            "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

print("Saving live data to:", DATA_DIR.resolve())
print()


# ==================================================
# FETCH HISTORICAL WARMUP DATA
# ==================================================

def fetch_history(symbol, instrument_type, strike=0, expiry=None):
    today = datetime.now(IST).date()
    start_date = today - timedelta(days=WARMUP_DAYS)

    encoded_symbol = quote(symbol, safe="")

    path = (
        f"/v3/historical-candle/{encoded_symbol}/minutes/1/"
        f"{today.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"
    )

    try:
        payload = upstox_get(path)
    except Exception as exc:
        print(f"History failed for {symbol}: {exc}")
        return pd.DataFrame()

    rows = payload.get("data", {}).get("candles", [])

    if not rows:
        return pd.DataFrame()

    # Upstox V3 candle format:
    # [timestamp, open, high, low, close, volume, open_interest]
    normalized_rows = [row[:7] for row in rows if len(row) >= 6]

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
        ],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = symbol
    df["instrument_type"] = instrument_type
    df["strike"] = strike
    df["expiry"] = pd.to_datetime(expiry, utc=True, errors="coerce") if expiry else pd.NaT

    return df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "instrument_type",
            "strike",
            "expiry",
        ]
    ]


print("Loading historical warmup candles...")

nifty_history = fetch_history(INDEX_SYMBOL, "INDEX")
ce_history = fetch_history(CE_SYMBOL, "CE", atm_strike, selected_expiry)
pe_history = fetch_history(PE_SYMBOL, "PE", atm_strike, selected_expiry)

history_frames = [df for df in [nifty_history, ce_history, pe_history] if not df.empty]

if not history_frames:
    raise RuntimeError("No historical warmup candles could be loaded from Upstox.")

market_df = pd.concat(history_frames, ignore_index=True)

market_df = (
    market_df
    .sort_values("timestamp")
    .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    .reset_index(drop=True)
)

print("Warmup candles loaded:", len(market_df))
print()


# ==================================================
# ADD CLOSED LIVE CANDLE
# ==================================================

def store_closed_candle(symbol, candle):
    global market_df

    if symbol == INDEX_SYMBOL:
        instrument_type = "INDEX"
        strike = 0
    elif symbol == CE_SYMBOL:
        instrument_type = "CE"
        strike = atm_strike
    elif symbol == PE_SYMBOL:
        instrument_type = "PE"
        strike = atm_strike
    else:
        return

    row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(candle["minute"]).tz_convert("UTC"),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
                "symbol": symbol,
                "instrument_type": instrument_type,
                "strike": strike,
                "expiry": (
                    pd.Timestamp(selected_expiry).tz_localize(IST).tz_convert("UTC")
                    if instrument_type in ("CE", "PE")
                    else pd.NaT
                ),
            }
        ]
    )

    market_df = pd.concat([market_df, row], ignore_index=True)

    market_df = (
        market_df
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    save_closed_candle(
        symbol,
        candle,
        instrument_type,
        strike,
    )


# ==================================================
# LIVE ML PREDICTION
# ==================================================

def try_prediction(minute):
    global last_prediction_minute
    global latest_predictions
    global latest_signal

    if minute < first_full_minute:
        return

    minute_utc = pd.Timestamp(minute).tz_convert("UTC")

    if last_prediction_minute == minute_utc:
        return

    minute_rows = market_df[
        market_df["timestamp"] == minute_utc
    ]

    symbols_available = set(
        minute_rows["symbol"]
    )

    required = {
        INDEX_SYMBOL,
        CE_SYMBOL,
        PE_SYMBOL
    }

    if not required.issubset(symbols_available):
        return

    feature_df = build_features(
        market_df.copy()
    )

    predictions_this_minute = []

    for option_type, symbol, trading_symbol, model in [
        ("CE", CE_SYMBOL, CE_TRADING_SYMBOL, ce_model),
        ("PE", PE_SYMBOL, PE_TRADING_SYMBOL, pe_model),
    ]:

        latest_row = feature_df[
            (feature_df["symbol"] == symbol)
            & (feature_df["timestamp"] == minute_utc)
        ]

        if latest_row.empty:
            continue

        row = latest_row.iloc[-1]

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in row.index
        ]

        if missing_features:
            raise ValueError(
                "build_features() is missing live model features:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing_features
                )
            )

        if row[FEATURES].isna().any():
            continue

        X = pd.DataFrame(
            [row[FEATURES]]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        setup_score = int(
            row["setup_score"]
        )

        fidelity_score = int(
            row.get("fidelity_score", 0)
        )

        fidelity_confirmed = int(
            row.get("fidelity_confirmed", 0)
        )

        signal_price = float(
            row["close"]
        )

        target_price = (
            signal_price
            * (1 + TARGET_PCT)
        )

        stop_price = (
            signal_price
            * (1 - STOP_PCT)
        )

        approved = (
            probability >= CONFIDENCE_THRESHOLD
            and setup_score >= SETUP_SCORE_THRESHOLD
            and fidelity_confirmed == 1
        )

        prediction = {
            "minute": minute,
            "option_type": option_type,
            "symbol": symbol,
            "trading_symbol": trading_symbol,
            "probability": probability,
            "setup_score": setup_score,
            "fidelity_score": fidelity_score,
            "fidelity_confirmed": fidelity_confirmed,
            "lvg_score": row.get("lvg_score", ""),
            "lvg_detected": row.get("lvg_detected", ""),
            "macd_line": row.get("macd_line", ""),
            "macd_signal": row.get("macd_signal", ""),
            "macd_histogram": row.get("macd_histogram", ""),
            "theta": row.get("theta", ""),
            "time_to_expiry_minutes": row.get("time_to_expiry_minutes", ""),
            "time_to_expiry_days": row.get("time_to_expiry_days", ""),
            "price": signal_price,
            "target": target_price,
            "stop": stop_price,
            "approved": approved,
            "selected": False,
        }

        latest_predictions[option_type] = prediction
        predictions_this_minute.append(prediction)

    if not predictions_this_minute:
        return

    approved_predictions = [
        prediction
        for prediction in predictions_this_minute
        if prediction["approved"]
    ]

    latest_signal = None

    if approved_predictions:
        latest_signal = max(
            approved_predictions,
            key=lambda prediction: prediction["probability"]
        )
        latest_signal["selected"] = True

    # Save both CE and PE evaluations for every completed minute.
    for prediction in predictions_this_minute:
        save_prediction(minute, prediction)

    last_prediction_minute = minute_utc

    if latest_signal:
        print()
        print("======================================")
        print(f"{latest_signal['option_type']} TRADE SIGNAL")
        print("======================================")
        print("Time:", minute.strftime("%H:%M"))
        print("Symbol:", latest_signal["trading_symbol"])
        print(
            f"{latest_signal['option_type']} Price: "
            f"Rs {latest_signal['price']:.2f}"
        )
        print(
            f"ML Probability: "
            f"{latest_signal['probability'] * 100:.2f}%"
        )
        print(
            f"Setup Score: "
            f"{latest_signal['setup_score']}/10"
        )
        print(
            f"Fidelity Score: "
            f"{latest_signal['fidelity_score']}"
        )
        print(
            f"Target: Rs {latest_signal['target']:.2f}"
        )
        print(
            f"Stop: Rs {latest_signal['stop']:.2f}"
        )
        print("======================================")
        print()


# ==================================================
# TERMINAL DASHBOARD
# ==================================================

def show_dashboard():
    global last_display

    now = time.time()
    if now - last_display < 0.25:
        return

    last_display = now

    print("\033[2J\033[H", end="")

    current_time = datetime.now(IST).strftime("%H:%M:%S")

    nifty = latest.get(INDEX_SYMBOL, {})
    ce_data = latest.get(CE_SYMBOL, {})
    pe_data = latest.get(PE_SYMBOL, {})

    print("==============================================")
    print(f"       NIFTY LIVE PREDICTOR   {current_time}")
    print("==============================================")
    print()
    print(f"NIFTY      {nifty.get('ltp', '-')}")
    print(f"EXPIRY     {selected_expiry}")
    print()

    print(
        f"{int(atm_strike)} CE   "
        f"LTP {ce_data.get('ltp', '-'):>8}   "
        f"BID {ce_data.get('bid_price', '-'):>8}   "
        f"ASK {ce_data.get('ask_price', '-'):>8}"
    )

    print(
        f"{int(atm_strike)} PE   "
        f"LTP {pe_data.get('ltp', '-'):>8}   "
        f"BID {pe_data.get('bid_price', '-'):>8}   "
        f"ASK {pe_data.get('ask_price', '-'):>8}"
    )

    print()
    print("----------------------------------------------")
    print("LIVE FEED HEALTH")

    if feed_ages and last_provider_time and last_receive_time:
        print(
            f"Upstox tick time      : "
            f"{last_provider_time.strftime('%H:%M:%S.%f')[:-3]}"
        )
        print(
            f"Received time         : "
            f"{last_receive_time.strftime('%H:%M:%S.%f')[:-3]}"
        )
        print(f"Feed age estimate*    : {current_feed_age_ms:.0f} ms")
        print(f"Feed age avg*         : {sum(feed_ages) / len(feed_ages):.0f} ms")
        print(f"Feed age min/max*     : {min(feed_ages):.0f} / {max(feed_ages):.0f} ms")
    else:
        print("Feed age              : Waiting for timestamp...")

    if tick_gaps:
        print(f"WebSocket tick gap    : {current_tick_gap_ms:.1f} ms")
        print(f"Average tick gap      : {sum(tick_gaps) / len(tick_gaps):.1f} ms")
        print(f"Tick gap min/max      : {min(tick_gaps):.1f} / {max(tick_gaps):.1f} ms")

    print("* Based on Upstox currentTs; transport/processing estimate.")
    print("----------------------------------------------")

    ce_prediction = latest_predictions.get("CE")
    pe_prediction = latest_predictions.get("PE")

    if ce_prediction or pe_prediction:

        if ce_prediction:
            print(
                f"CE ML Probability      : "
                f"{ce_prediction['probability'] * 100:.2f}%"
            )
            print(
                f"CE Setup / Fidelity    : "
                f"{ce_prediction['setup_score']}/10 / "
                f"{ce_prediction['fidelity_confirmed']}"
            )

        if pe_prediction:
            print(
                f"PE ML Probability      : "
                f"{pe_prediction['probability'] * 100:.2f}%"
            )
            print(
                f"PE Setup / Fidelity    : "
                f"{pe_prediction['setup_score']}/10 / "
                f"{pe_prediction['fidelity_confirmed']}"
            )

        if latest_signal:
            print(
                f"Signal                : "
                f"BUY {latest_signal['option_type']}"
            )
            print(
                f"Reference Price       : "
                f"Rs {latest_signal['price']:.2f}"
            )
            print(
                f"Target                : "
                f"Rs {latest_signal['target']:.2f}"
            )
            print(
                f"Stop                  : "
                f"Rs {latest_signal['stop']:.2f}"
            )
        else:
            print("Signal                : NO TRADE")
    else:
        print("ML                    : Waiting for next full 1M candle...")

    print("----------------------------------------------")
    print(f"Saving data           : {DATA_DIR}")
    print()
    print("Ctrl+C to stop")


# ==================================================
# NORMALIZE UPSTOX FEED
# ==================================================

def normalize_feed(instrument_key, feed, provider_ms=None):
    full_feed = feed.get("fullFeed", {})

    # Index uses indexFF; options/equities use marketFF.
    data = full_feed.get("indexFF") or full_feed.get("marketFF")
    if not data:
        return None

    ltpc = data.get("ltpc", {})
    ltp = ltpc.get("ltp")

    if ltp is None:
        return None

    # Use Upstox currentTs for candle bucketing. ltpc.ltt is the last-trade
    # timestamp and can stay stale when only quotes/depth are changing.
    event_ms = provider_ms or ltpc.get("ltt")
    if event_ms:
        try:
            event_time = datetime.fromtimestamp(float(event_ms) / 1000, IST)
        except (TypeError, ValueError):
            event_time = datetime.now(IST)
    else:
        event_time = datetime.now(IST)

    bid_price = "-"
    ask_price = "-"

    quotes = data.get("marketLevel", {}).get("bidAskQuote", []) or []
    depth = {}

    for level, quote_data in enumerate(quotes[:5], start=1):
        depth[level] = {
            "bid_price": quote_data.get("bidP", ""),
            "bid_qty": quote_data.get("bidQ", ""),
            "ask_price": quote_data.get("askP", ""),
            "ask_qty": quote_data.get("askQ", ""),
        }

    if quotes:
        bid_price = quotes[0].get("bidP", "-")
        ask_price = quotes[0].get("askP", "-")

    cumulative_volume = data.get("vtt", 0)
    try:
        cumulative_volume = int(cumulative_volume or 0)
    except (TypeError, ValueError):
        cumulative_volume = 0

    last_trade_time = None
    ltt = ltpc.get("ltt")
    if ltt:
        try:
            last_trade_time = datetime.fromtimestamp(float(ltt) / 1000, IST)
        except (TypeError, ValueError):
            pass

    greeks = data.get("optionGreeks", {}) or {}

    return {
        "symbol": instrument_key,
        "ltp": float(ltp),
        "event_time": event_time,
        "last_trade_time": last_trade_time,
        "ltq": ltpc.get("ltq", ""),
        "close_price": ltpc.get("cp", ""),
        "bid_price": bid_price,
        "ask_price": ask_price,
        "depth": depth,
        "atp": data.get("atp", ""),
        "vol_traded_today": cumulative_volume,
        "open_interest": data.get("oi", ""),
        "total_buy_qty": data.get("tbq", ""),
        "total_sell_qty": data.get("tsq", ""),
        "iv": greeks.get("iv", ""),
        "delta": greeks.get("delta", ""),
        "theta": greeks.get("theta", ""),
        "gamma": greeks.get("gamma", ""),
        "vega": greeks.get("vega", ""),
        "rho": greeks.get("rho", ""),
    }


# ==================================================
# PROCESS ONE NORMALIZED TICK
# ==================================================

def process_tick(tick):
    symbol = tick["symbol"]
    price = tick["ltp"]
    dt = tick["event_time"]

    save_tick(tick)
    latest[symbol] = tick

    minute = dt.replace(second=0, microsecond=0)
    cumulative_volume = tick.get("vol_traded_today", 0)

    if symbol not in candles:
        candles[symbol] = {
            "minute": minute,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,
            "last_cumulative_volume": cumulative_volume,
        }
        return

    candle = candles[symbol]

    previous_cumulative = candle.get(
        "last_cumulative_volume",
        cumulative_volume,
    )

    volume_delta = max(0, cumulative_volume - previous_cumulative)

    if candle["minute"] == minute:
        candle["high"] = max(candle["high"], price)
        candle["low"] = min(candle["low"], price)
        candle["close"] = price
        candle["volume"] += volume_delta
        candle["last_cumulative_volume"] = cumulative_volume
    else:
        completed_minute = candle["minute"]

        store_closed_candle(symbol, candle)

        candles[symbol] = {
            "minute": minute,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume_delta,
            "last_cumulative_volume": cumulative_volume,
        }

        try_prediction(completed_minute)


# ==================================================
# WEBSOCKET CALLBACKS
# ==================================================

def on_message(message):
    if not isinstance(message, dict):
        return

    # market_info messages do not contain instrument feeds.
    feeds = message.get("feeds", {})
    if not feeds:
        return

    update_latency(message)

    provider_ms = message.get("currentTs")

    for instrument_key, feed in feeds.items():
        if instrument_key not in {INDEX_SYMBOL, CE_SYMBOL, PE_SYMBOL}:
            continue

        tick = normalize_feed(instrument_key, feed, provider_ms)
        if tick:
            process_tick(tick)

    show_dashboard()


def on_error(error):
    print("UPSTOX ERROR:", error)


def on_close(*args):
    print("UPSTOX CONNECTION CLOSED", *args)


def on_open():
    symbols = [INDEX_SYMBOL, CE_SYMBOL, PE_SYMBOL]
    print("Connected to Upstox. Subscribing to:")
    print("  NIFTY:", INDEX_SYMBOL)
    print("  CE   :", CE_TRADING_SYMBOL)
    print("  PE   :", PE_TRADING_SYMBOL)

    upstox_streamer.subscribe(symbols, "full")


# ==================================================
# WEBSOCKET
# ==================================================

configuration = upstox_client.Configuration()
configuration.access_token = ACCESS_TOKEN

api_client = upstox_client.ApiClient(configuration)
upstox_streamer = upstox_client.MarketDataStreamerV3(api_client)

upstox_streamer.on("open", on_open)
upstox_streamer.on("message", on_message)
upstox_streamer.on("error", on_error)
upstox_streamer.on("close", on_close)

# Reconnect every 5 seconds, up to 20 retries.
upstox_streamer.auto_reconnect(True, 5, 20)

upstox_streamer.connect()
