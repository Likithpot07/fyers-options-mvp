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
STOP_PCT = 0.10
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
# VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_prediction_columns = {
    "timestamp",
    "symbol",
    "instrument_type",
    "strike",
    "prediction_probability",
    "candidate_setup",
    "fidelity_confirmed",
}

missing_prediction_columns = (
    required_prediction_columns
    - set(predictions.columns)
)

if missing_prediction_columns:
    raise ValueError(
        "Predictions file is missing required columns: "
        + ", ".join(
            sorted(missing_prediction_columns)
        )
    )


# --------------------------------------------------
# FUNNEL HELPERS
# --------------------------------------------------

def print_side_counts(label, frame):
    counts = (
        frame["instrument_type"]
        .value_counts()
        .reindex(["CE", "PE"], fill_value=0)
    )

    print(
        f"{label:<30}"
        f"CE: {int(counts['CE']):>5} | "
        f"PE: {int(counts['PE']):>5} | "
        f"TOTAL: {len(frame):>5}"
    )


print("\n================================")
print("BACKTEST SIGNAL FUNNEL")
print("================================")

# Stage 1 — model confidence only
stage_80 = predictions[
    predictions["prediction_probability"]
    >= CONFIDENCE_THRESHOLD
].copy()

print_side_counts(
    f"{CONFIDENCE_THRESHOLD:.0%}+ model signals",
    stage_80
)

# Stage 2 — setup filter
stage_setup = stage_80[
    stage_80["candidate_setup"] == 1
].copy()

print_side_counts(
    "+ setup filter",
    stage_setup
)

# Stage 3 — Fidelity filter
stage_fidelity = stage_setup[
    stage_setup["fidelity_confirmed"] == 1
].copy()

print_side_counts(
    "+ Fidelity filter",
    stage_fidelity
)

# Stage 4 — choose one best CE/PE opportunity per timestamp,
# exactly as the live system would need to choose between simultaneous signals.
best_candidates = (
    stage_fidelity
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

print_side_counts(
    "+ best candidate/minute",
    best_candidates
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

# Only one active trade at a time across CE + PE.
last_global_exit = None

# Funnel counters after best-candidate selection.
overlap_rejected = {
    "CE": 0,
    "PE": 0
}

missing_symbol_rejected = {
    "CE": 0,
    "PE": 0
}

no_future_rejected = {
    "CE": 0,
    "PE": 0
}

cross_day_rejected = {
    "CE": 0,
    "PE": 0
}

empty_window_rejected = {
    "CE": 0,
    "PE": 0
}


for _, signal in best_candidates.iterrows():

    option_type = signal["instrument_type"]
    symbol = signal["symbol"]
    signal_time = signal["timestamp"]

    # --------------------------------------------------
    # OVERLAP / COOLDOWN FILTER
    # --------------------------------------------------

    if (
        last_global_exit is not None
        and signal_time <= last_global_exit
    ):
        overlap_rejected[option_type] += 1
        continue

    if symbol not in raw_by_symbol:
        missing_symbol_rejected[option_type] += 1
        continue

    candles = raw_by_symbol[symbol]

    # --------------------------------------------------
    # ENTRY = NEXT CANDLE OPEN
    # --------------------------------------------------

    future = candles[
        candles["timestamp"] > signal_time
    ]

    if future.empty:
        no_future_rejected[option_type] += 1
        continue

    entry_row = future.iloc[0]
    entry_time = entry_row["timestamp"]

    # Never cross trading day.
    if entry_time.date() != signal_time.date():
        cross_day_rejected[option_type] += 1
        continue

    entry_price = entry_row["open"]

    target_price = (
        entry_price
        * (1 + TARGET_PCT)
    )

    stop_price = (
        entry_price
        * (1 - STOP_PCT)
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
        empty_window_rejected[option_type] += 1
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

        # Conservative assumption:
        # if target and stop occur in same 1m candle,
        # assume stop happened first.
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
            "instrument_type": option_type,
            "strike": signal["strike"],
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "prediction_probability":
                signal["prediction_probability"],
            "candidate_setup":
                signal["candidate_setup"],
            "fidelity_confirmed":
                signal["fidelity_confirmed"],
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "net_return_pct": return_pct
        }
    )

    last_global_exit = exit_time


# --------------------------------------------------
# POST-BEST-CANDIDATE FUNNEL
# --------------------------------------------------

print("\n--------------------------------")
print("POST-CANDIDATE REJECTIONS")
print("--------------------------------")

for option_type in ["CE", "PE"]:

    best_count = int(
        (
            best_candidates["instrument_type"]
            == option_type
        ).sum()
    )

    after_overlap = (
        best_count
        - overlap_rejected[option_type]
    )

    final_count = sum(
        1
        for trade in trades
        if trade["instrument_type"] == option_type
    )

    print(f"\n{option_type}")
    print("Best candidates:", best_count)
    print(
        "Overlap/cooldown rejected:",
        overlap_rejected[option_type]
    )
    print(
        "After overlap/cooldown:",
        after_overlap
    )
    print(
        "Other data-entry rejects:",
        missing_symbol_rejected[option_type]
        + no_future_rejected[option_type]
        + cross_day_rejected[option_type]
        + empty_window_rejected[option_type]
    )
    print("Final trades:", final_count)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results = pd.DataFrame(trades)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

if results.empty:

    # Still write an empty result file with expected columns,
    # so diagnostics can distinguish "zero trades" from "file missing".
    empty_columns = [
        "symbol",
        "instrument_type",
        "strike",
        "signal_time",
        "entry_time",
        "exit_time",
        "prediction_probability",
        "candidate_setup",
        "fidelity_confirmed",
        "entry_price",
        "target_price",
        "stop_price",
        "exit_price",
        "exit_reason",
        "net_return_pct"
    ]

    pd.DataFrame(
        columns=empty_columns
    ).to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\n----------------------------")
    print("BACKTEST RESULTS")
    print("----------------------------")
    print("Trades: 0")
    print("No trades generated after all filters.")
    print("\nSaved:")
    print(OUTPUT_PATH)

    raise SystemExit


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
        stops=(
            "exit_reason",
            lambda x:
                x.isin(
                    [
                        "STOP",
                        "STOP_SAME_CANDLE"
                    ]
                ).sum()
        ),
        timeouts=(
            "exit_reason",
            lambda x:
                (x == "TIMEOUT").sum()
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
