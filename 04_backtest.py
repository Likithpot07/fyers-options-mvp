from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PREDICTIONS_PATH = Path(
    "data/processed/test_predictions.parquet"
)

RAW_PATH = Path(
    "data/raw/candles_1m.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/backtest_results.parquet"
)

CONFIDENCE_THRESHOLD = 0.80

TARGET_PCT = 0.10
STOP_PCT = 0.05
MAX_HOLD_MINUTES = 30


# --------------------------------------------------
# LOAD
# --------------------------------------------------

predictions = pd.read_parquet(PREDICTIONS_PATH)
raw = pd.read_parquet(RAW_PATH)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"],
    utc=True
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    utc=True
)

raw = raw[
    raw["instrument_type"].isin(["CE", "PE"])
].copy()

raw = raw.sort_values(
    ["symbol", "timestamp"]
)


# --------------------------------------------------
# HIGH CONFIDENCE ONLY
# --------------------------------------------------

signals = predictions[
    (predictions["instrument_type"] == "PE")
    &
    (predictions["prediction_probability"] >= 0.80)
    &
    (predictions["candidate_setup"] == 1)
].copy()

print("Raw 80%+ candidate signals:", len(signals))


# --------------------------------------------------
# PICK ONLY BEST OPTION AT EACH MINUTE
# --------------------------------------------------

signals = (
    signals
    .sort_values(
        [
            "timestamp",
            "prediction_probability"
        ],
        ascending=[True, False]
    )
    .groupby(
        "timestamp",
        as_index=False
    )
    .first()
)

print(
    "Best-candidate signals:",
    len(signals)
)


# --------------------------------------------------
# RAW DATA BY SYMBOL
# --------------------------------------------------

raw_by_symbol = {
    symbol: group.reset_index(drop=True)
    for symbol, group in raw.groupby("symbol")
}


# --------------------------------------------------
# BACKTEST
# --------------------------------------------------

trades = []

# Only ONE active trade at a time
last_global_exit = None


for _, signal in signals.iterrows():

    symbol = signal["symbol"]
    signal_time = signal["timestamp"]

    # Ignore new signals while previous trade is open
    if (
        last_global_exit is not None
        and signal_time <= last_global_exit
    ):
        continue

    if symbol not in raw_by_symbol:
        continue

    candles = raw_by_symbol[symbol]


    # --------------------------------------------------
    # ENTRY = NEXT CANDLE OPEN
    # --------------------------------------------------

    future = candles[
        candles["timestamp"] > signal_time
    ]

    if future.empty:
        continue

    entry_row = future.iloc[0]

    entry_time = entry_row["timestamp"]

    # Never cross trading day
    if entry_time.date() != signal_time.date():
        continue

    entry_price = entry_row["open"]

    target_price = (
        entry_price * (1 + TARGET_PCT)
    )

    stop_price = (
        entry_price * (1 - STOP_PCT)
    )


    # --------------------------------------------------
    # EXACT 30 REAL MINUTES
    # --------------------------------------------------

    end_time = (
        entry_time
        + pd.Timedelta(
            minutes=MAX_HOLD_MINUTES
        )
    )

    trade_candles = candles[
        (candles["timestamp"] >= entry_time)
        & (candles["timestamp"] <= end_time)
        & (
            candles["timestamp"].dt.date
            == entry_time.date()
        )
    ]

    if trade_candles.empty:
        continue


    # --------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------

    exit_price = None
    exit_time = None
    exit_reason = None


    for _, candle in trade_candles.iterrows():

        target_hit = (
            candle["high"]
            >= target_price
        )

        stop_hit = (
            candle["low"]
            <= stop_price
        )


        # Conservative assumption
        if target_hit and stop_hit:

            exit_price = stop_price
            exit_time = candle["timestamp"]
            exit_reason = "STOP_SAME_CANDLE"

            break


        if stop_hit:

            exit_price = stop_price
            exit_time = candle["timestamp"]
            exit_reason = "STOP"

            break


        if target_hit:

            exit_price = target_price
            exit_time = candle["timestamp"]
            exit_reason = "TARGET"

            break


    # --------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------

    if exit_price is None:

        last_candle = trade_candles.iloc[-1]

        exit_price = last_candle["close"]
        exit_time = last_candle["timestamp"]
        exit_reason = "TIMEOUT"


    return_pct = (
        (
            exit_price
            - entry_price
        )
        / entry_price
        * 100
    )


    trades.append(
        {
            "symbol": symbol,
            "instrument_type":
                signal["instrument_type"],
            "strike":
                signal["strike"],

            "signal_time":
                signal_time,

            "entry_time":
                entry_time,

            "exit_time":
                exit_time,

            "prediction_probability":
                signal[
                    "prediction_probability"
                ],

            "entry_price":
                entry_price,

            "target_price":
                target_price,

            "stop_price":
                stop_price,

            "exit_price":
                exit_price,

            "exit_reason":
                exit_reason,

            "net_return_pct":
                return_pct
        }
    )

    last_global_exit = exit_time


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results = pd.DataFrame(trades)

if results.empty:
    print("No trades generated.")
    raise SystemExit


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_parquet(
    OUTPUT_PATH,
    index=False
)


targets = (
    results["exit_reason"]
    == "TARGET"
).sum()

stops = results[
    results["exit_reason"].isin(
        [
            "STOP",
            "STOP_SAME_CANDLE"
        ]
    )
].shape[0]

timeouts = (
    results["exit_reason"]
    == "TIMEOUT"
).sum()

total = len(results)

win_rate = (
    targets / total * 100
)


print("\n----------------------------")
print("BACKTEST RESULTS")
print("----------------------------")

print("Trades:", total)
print("Targets:", targets)
print("Stops:", stops)
print("Timeouts:", timeouts)

print(
    "Win Rate:",
    f"{win_rate:.2f}%"
)

print(
    "Average Return:",
    f"{results['net_return_pct'].mean():.2f}%"
)


print("\nCE / PE")

summary = (
    results
    .groupby("instrument_type")
    .agg(
        trades=("symbol", "count"),

        wins=(
            "exit_reason",
            lambda x:
                (x == "TARGET").sum()
        ),

        avg_return=(
            "net_return_pct",
            "mean"
        )
    )
)

summary["win_rate"] = (
    summary["wins"]
    / summary["trades"]
    * 100
)

print(summary)


print("\nSaved:")
print(OUTPUT_PATH)