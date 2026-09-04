import time
from pathlib import Path
from collections import deque
from datetime import datetime, timezone, timedelta

import pandas as pd
from xgboost import XGBClassifier

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from config import CLIENT_ID, ACCESS_TOKEN
from technical_features import build_features


# ==================================================
# CONFIG
# ==================================================

IST = timezone(timedelta(hours=5, minutes=30))

INDEX_SYMBOL = "NSE:NIFTY50-INDEX"

MODEL_PATH = Path("models/xgboost_pe_10pct.json")

CONFIDENCE_THRESHOLD = 0.85
SETUP_SCORE_THRESHOLD = 5

TARGET_PCT = 0.10
STOP_PCT = 0.05

# Historical candles required for RSI / EMA / VWAP / Pivot
WARMUP_DAYS = 7


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

    "distance_from_pivot_pct",

    "distance_from_strike_pct",
    "premium_pct_of_spot",

    "ce_pe_ratio",

    "ce_return_3m",
    "pe_return_3m",
    "ce_minus_pe_return_3m",

    "setup_score"
]


# ==================================================
# GLOBAL STATE
# ==================================================

candles = {}
latest = {}

market_df = pd.DataFrame()

last_prediction_minute = None
latest_prediction = None

last_display = 0

socket_token = f"{CLIENT_ID}:{ACCESS_TOKEN}"


# Ignore first partial minute after startup
startup_now = datetime.now(IST)

first_full_minute = (
    startup_now.replace(
        second=0,
        microsecond=0
    )
    + timedelta(minutes=1)
)


# ==========================================
# LATENCY / FEED HEALTH
# ==========================================

feed_ages = deque(maxlen=1000)
tick_gaps = deque(maxlen=1000)

current_feed_age_ms = None
current_tick_gap_ms = None

last_receive_ns = None
last_exchange_time = None
last_receive_time = None




def update_latency(message):
    global current_feed_age_ms
    global current_tick_gap_ms
    global last_receive_ns
    global last_exchange_time
    global last_receive_time

    receive_ns = time.time_ns()
    receive_seconds = receive_ns / 1_000_000_000

    exchange_seconds = message.get("exch_feed_time")

    # ------------------------------------------
    # Exchange timestamp -> local receive age
    # ------------------------------------------

    if exchange_seconds:
        current_feed_age_ms = (
            receive_seconds - float(exchange_seconds)
        ) * 1000

        # Clock differences can occasionally make it negative
        if current_feed_age_ms >= 0:
            feed_ages.append(current_feed_age_ms)

        last_exchange_time = datetime.fromtimestamp(
            exchange_seconds,
            IST
        )

        last_receive_time = datetime.fromtimestamp(
            receive_seconds,
            IST
        )

    # ------------------------------------------
    # Time between WebSocket packets
    # ------------------------------------------

    if last_receive_ns is not None:

        current_tick_gap_ms = (
            receive_ns - last_receive_ns
        ) / 1_000_000

        tick_gaps.append(current_tick_gap_ms)

    last_receive_ns = receive_ns
# ==================================================
# LOAD XGBOOST MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = XGBClassifier()
model.load_model(MODEL_PATH)

print("PE XGBoost model loaded.")


# ==================================================
# FYERS REST CLIENT
# ==================================================

fyers_rest = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    log_path=""
)


# ==================================================
# FIND CURRENT ATM CE + PE
# ==================================================

chain = fyers_rest.optionchain(
    data={
        "symbol": INDEX_SYMBOL,
        "strikecount": 5,
        "timestamp": ""
    }
)

if "data" not in chain:
    raise Exception(
        f"FYERS option chain failed: {chain}"
    )


options = chain["data"]["optionsChain"]


spot = next(
    x["ltp"]
    for x in options
    if x.get("symbol") == INDEX_SYMBOL
)


atm_strike = round(spot / 50) * 50


ce = next(
    x for x in options
    if x.get("strike_price") == atm_strike
    and x.get("option_type") == "CE"
)


pe = next(
    x for x in options
    if x.get("strike_price") == atm_strike
    and x.get("option_type") == "PE"
)


CE_SYMBOL = ce["symbol"]
PE_SYMBOL = pe["symbol"]


print()
print("NIFTY :", spot)
print("ATM   :", atm_strike)
print("CE    :", CE_SYMBOL)
print("PE    :", PE_SYMBOL)
print()


# ==================================================
# FETCH HISTORICAL WARMUP DATA
# ==================================================

def fetch_history(
    symbol,
    instrument_type,
    strike=0
):

    today = datetime.now(IST).date()

    start_date = (
        today
        - timedelta(days=WARMUP_DAYS)
    )


    payload = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": start_date.strftime("%Y-%m-%d"),
        "range_to": today.strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }


    response = fyers_rest.history(
        data=payload
    )


    if response.get("s") != "ok":

        print(
            f"History failed for {symbol}:",
            response
        )

        return pd.DataFrame()


    rows = response.get(
        "candles",
        []
    )


    if not rows:
        return pd.DataFrame()


    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )


    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )


    df["symbol"] = symbol
    df["instrument_type"] = instrument_type
    df["strike"] = strike


    return df


print("Loading historical warmup candles...")


nifty_history = fetch_history(
    INDEX_SYMBOL,
    "INDEX"
)

ce_history = fetch_history(
    CE_SYMBOL,
    "CE",
    atm_strike
)

pe_history = fetch_history(
    PE_SYMBOL,
    "PE",
    atm_strike
)


market_df = pd.concat(
    [
        nifty_history,
        ce_history,
        pe_history
    ],
    ignore_index=True
)


market_df = (
    market_df
    .sort_values("timestamp")
    .drop_duplicates(
        subset=[
            "symbol",
            "timestamp"
        ],
        keep="last"
    )
    .reset_index(drop=True)
)


print(
    "Warmup candles loaded:",
    len(market_df)
)

print()


# ==================================================
# ADD CLOSED LIVE CANDLE
# ==================================================

def store_closed_candle(
    symbol,
    candle
):

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
                "timestamp":
                    pd.Timestamp(
                        candle["minute"]
                    ).tz_convert("UTC"),

                "open":
                    candle["open"],

                "high":
                    candle["high"],

                "low":
                    candle["low"],

                "close":
                    candle["close"],

                "volume":
                    candle["volume"],

                "symbol":
                    symbol,

                "instrument_type":
                    instrument_type,

                "strike":
                    strike
            }
        ]
    )


    market_df = pd.concat(
        [
            market_df,
            row
        ],
        ignore_index=True
    )


    market_df = (
        market_df
        .drop_duplicates(
            subset=[
                "symbol",
                "timestamp"
            ],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ==================================================
# LIVE ML PREDICTION
# ==================================================

def try_prediction(minute):

    global last_prediction_minute
    global latest_prediction


    # First candle after startup may be incomplete
    if minute < first_full_minute:
        return


    minute_utc = pd.Timestamp(
        minute
    ).tz_convert("UTC")


    # Already predicted this candle
    if (
        last_prediction_minute
        == minute_utc
    ):
        return


    # Make sure NIFTY + CE + PE all have
    # this completed minute
    minute_rows = market_df[
        market_df["timestamp"]
        == minute_utc
    ]


    symbols_available = set(
        minute_rows["symbol"]
    )


    required = {
        INDEX_SYMBOL,
        CE_SYMBOL,
        PE_SYMBOL
    }


    if not required.issubset(
        symbols_available
    ):
        return


    # Calculate exactly the same features
    # used during training
    feature_df = build_features(
        market_df.copy()
    )


    latest_pe = feature_df[
        (
            feature_df["symbol"]
            == PE_SYMBOL
        )
        &
        (
            feature_df["timestamp"]
            == minute_utc
        )
    ]


    if latest_pe.empty:
        return


    row = latest_pe.iloc[-1]


    # Don't predict with incomplete indicators
    if row[FEATURES].isna().any():
        return


    X = pd.DataFrame(
        [row[FEATURES]]
    )


    probability = model.predict_proba(
        X
    )[0][1]


    setup_score = int(
        row["setup_score"]
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
        probability
        >= CONFIDENCE_THRESHOLD

        and

        setup_score
        >= SETUP_SCORE_THRESHOLD
    )


    latest_prediction = {
        "minute":
            minute,

        "probability":
            probability,

        "setup_score":
            setup_score,

        "price":
            signal_price,

        "target":
            target_price,

        "stop":
            stop_price,

        "approved":
            approved
    }


    last_prediction_minute = (
        minute_utc
    )


    # Immediate signal
    if approved:

        print()
        print("======================================")
        print("🚨 PE TRADE SIGNAL")
        print("======================================")

        print(
            "Time:",
            minute.strftime("%H:%M")
        )

        print(
            "Symbol:",
            PE_SYMBOL
        )

        print(
            f"PE Price: ₹{signal_price:.2f}"
        )

        print(
            f"ML Probability: "
            f"{probability * 100:.2f}%"
        )

        print(
            f"Setup Score: "
            f"{setup_score}/8"
        )

        print(
            f"Target: ₹{target_price:.2f}"
        )

        print(
            f"Stop: ₹{stop_price:.2f}"
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


    print(
        "\033[2J\033[H",
        end=""
    )


    current_time = (
        datetime.now(IST)
        .strftime("%H:%M:%S")
    )


    nifty = latest.get(
        INDEX_SYMBOL,
        {}
    )

    ce_data = latest.get(
        CE_SYMBOL,
        {}
    )

    pe_data = latest.get(
        PE_SYMBOL,
        {}
    )


    print(
        "=============================================="
    )

    print(
        f"       NIFTY LIVE PREDICTOR   {current_time}"
    )

    print(
        "=============================================="
    )

    print()


    print(
        f"NIFTY      "
        f"{nifty.get('ltp', '-')}"
    )


    print()


    print(
        f"{atm_strike} CE   "
        f"LTP {ce_data.get('ltp', '-'):>8}   "
        f"BID {ce_data.get('bid_price', '-'):>8}   "
        f"ASK {ce_data.get('ask_price', '-'):>8}"
    )


    print(
        f"{atm_strike} PE   "
        f"LTP {pe_data.get('ltp', '-'):>8}   "
        f"BID {pe_data.get('bid_price', '-'):>8}   "
        f"ASK {pe_data.get('ask_price', '-'):>8}"
    )


    # ----------------------------------------------
    # LIVE FEED HEALTH
    # ----------------------------------------------

    print()
    print(
        "----------------------------------------------"
    )
    print("LIVE FEED HEALTH")

    if feed_ages:

        print(
            f"Exchange time         : "
            f"{last_exchange_time.strftime('%H:%M:%S')}"
        )

        print(
            f"Received time         : "
            f"{last_receive_time.strftime('%H:%M:%S.%f')[:-3]}"
        )

        print(
            f"Feed age estimate*    : "
            f"{current_feed_age_ms:.0f} ms"
        )

        print(
            f"Feed age avg*         : "
            f"{sum(feed_ages) / len(feed_ages):.0f} ms"
        )

        print(
            f"Feed age min/max*     : "
            f"{min(feed_ages):.0f} / "
            f"{max(feed_ages):.0f} ms"
        )

    else:

        print("Feed age              : Waiting for timestamp...")

    if tick_gaps:

        print(
            f"WebSocket tick gap    : "
            f"{current_tick_gap_ms:.1f} ms"
        )

        print(
            f"Average tick gap      : "
            f"{sum(tick_gaps) / len(tick_gaps):.1f} ms"
        )

        print(
            f"Tick gap min/max      : "
            f"{min(tick_gaps):.1f} / "
            f"{max(tick_gaps):.1f} ms"
        )

    print(
        "* FYERS exch_feed_time has 1-second precision; "
        "feed age is an estimate."
    )

    print(
        "----------------------------------------------"
    )


    # ----------------------------------------------
    # ML STATUS
    # ----------------------------------------------

    if latest_prediction:

        probability = (
            latest_prediction[
                "probability"
            ]
            * 100
        )


        print(
            f"Latest ML Probability : "
            f"{probability:.2f}%"
        )


        print(
            f"Setup Score           : "
            f"{latest_prediction['setup_score']}/8"
        )


        if latest_prediction[
            "approved"
        ]:

            print(
                "Signal                : 🚨 BUY PE"
            )

            print(
                f"Reference Price       : "
                f"₹{latest_prediction['price']:.2f}"
            )

            print(
                f"Target                : "
                f"₹{latest_prediction['target']:.2f}"
            )

            print(
                f"Stop                  : "
                f"₹{latest_prediction['stop']:.2f}"
            )

        else:

            print(
                "Signal                : NO TRADE"
            )

    else:

        print(
            "ML                    : Waiting for next full 1M candle..."
        )


    print(
        "----------------------------------------------"
    )

    print()
    print("Ctrl+C to stop")


# ==================================================
# WEBSOCKET MESSAGE
# ==================================================

def on_message(message):

    if (
        "symbol" not in message
        or
        "ltp" not in message
    ):
        return


    # Capture receive time as early as possible for feed-health stats.
    update_latency(message)


    symbol = message["symbol"]
    price = message["ltp"]


    latest[symbol] = message


    timestamp = message.get(
        "exch_feed_time"
    )


    if timestamp:

        dt = datetime.fromtimestamp(
            timestamp,
            IST
        )

    else:

        dt = datetime.now(IST)


    minute = dt.replace(
        second=0,
        microsecond=0
    )


    cumulative_volume = message.get(
        "vol_traded_today",
        0
    )


    # ----------------------------------------------
    # FIRST TICK
    # ----------------------------------------------

    if symbol not in candles:

        candles[symbol] = {

            "minute":
                minute,

            "open":
                price,

            "high":
                price,

            "low":
                price,

            "close":
                price,

            "volume":
                0,

            "last_cumulative_volume":
                cumulative_volume
        }


        show_dashboard()
        return


    candle = candles[symbol]


    previous_cumulative = candle.get(
        "last_cumulative_volume",
        cumulative_volume
    )


    volume_delta = max(
        0,
        cumulative_volume
        - previous_cumulative
    )


    # ----------------------------------------------
    # SAME MINUTE
    # ----------------------------------------------

    if candle["minute"] == minute:

        candle["high"] = max(
            candle["high"],
            price
        )

        candle["low"] = min(
            candle["low"],
            price
        )

        candle["close"] = price

        candle["volume"] += (
            volume_delta
        )

        candle[
            "last_cumulative_volume"
        ] = cumulative_volume


    # ----------------------------------------------
    # NEW MINUTE
    # ----------------------------------------------

    else:

        completed_minute = (
            candle["minute"]
        )


        # Store finished candle
        store_closed_candle(
            symbol,
            candle
        )


        # Start new candle
        candles[symbol] = {

            "minute":
                minute,

            "open":
                price,

            "high":
                price,

            "low":
                price,

            "close":
                price,

            "volume":
                volume_delta,

            "last_cumulative_volume":
                cumulative_volume
        }


        # Prediction happens only after
        # NIFTY + CE + PE have all closed
        # the same minute.
        try_prediction(
            completed_minute
        )


    show_dashboard()


# ==================================================
# CALLBACKS
# ==================================================

def on_error(message):
    print(
        "ERROR:",
        message
    )


def on_close(message):
    print(
        "CLOSED:",
        message
    )


def on_open():

    symbols = [
        INDEX_SYMBOL,
        CE_SYMBOL,
        PE_SYMBOL
    ]


    fyers_socket.subscribe(
        symbols=symbols,
        data_type="SymbolUpdate"
    )


# ==================================================
# WEBSOCKET
# ==================================================

fyers_socket = data_ws.FyersDataSocket(

    access_token=
        socket_token,

    log_path="",

    litemode=False,

    write_to_file=False,

    reconnect=True,

    on_connect=
        on_open,

    on_close=
        on_close,

    on_error=
        on_error,

    on_message=
        on_message
)


fyers_socket.connect()