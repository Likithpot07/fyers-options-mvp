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

TARGET_PCT = 0.10
STOP_PCT = 0.10
MAX_HOLD_MINUTES = 30

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# --------------------------------------------------
# LOAD
# --------------------------------------------------

predictions = pd.read_parquet(
    PREDICTIONS_PATH
)

raw = pd.read_parquet(
    RAW_PATH
)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"],
    utc=True
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    utc=True
)

raw = raw[
    raw["instrument_type"].isin(
        ["CE", "PE"]
    )
].copy()

raw_by_symbol = {
    symbol: (
        group
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    for symbol, group
    in raw.groupby("symbol")
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def summarize_prediction_layer(
    frame,
    threshold,
    option_type,
    label
):

    subset = frame[
        (frame["instrument_type"] == option_type)
        & (
            frame["prediction_probability"]
            >= threshold
        )
    ].copy()

    if subset.empty:
        return {
            "option_type": option_type,
            "threshold": threshold,
            "layer": label,
            "signals": 0,
            "actual_wins": 0,
            "actual_win_rate": 0.0,
            "avg_probability": 0.0,
        }

    actual_wins = int(
        subset["target_hit"].sum()
    )

    return {
        "option_type": option_type,
        "threshold": threshold,
        "layer": label,
        "signals": len(subset),
        "actual_wins": actual_wins,
        "actual_win_rate": (
            actual_wins
            / len(subset)
            * 100
        ),
        "avg_probability": (
            subset[
                "prediction_probability"
            ].mean()
        ),
    }


def backtest_threshold(
    threshold,
    option_type,
    require_setup=False,
    require_fidelity=False,
):

    signals = predictions[
        (predictions["instrument_type"] == option_type)
        & (
            predictions["prediction_probability"]
            >= threshold
        )
    ].copy()

    if require_setup:
        signals = signals[
            signals["candidate_setup"] == 1
        ].copy()

    if require_fidelity:
        signals = signals[
            signals["fidelity_confirmed"] == 1
        ].copy()

    # Best candidate for THIS option type
    # at each timestamp.
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

    trades = []
    last_exit = None

    for _, signal in signals.iterrows():

        signal_time = signal["timestamp"]
        symbol = signal["symbol"]

        # One active trade at a time
        # for this side.
        if (
            last_exit is not None
            and signal_time <= last_exit
        ):
            continue

        if symbol not in raw_by_symbol:
            continue

        candles = raw_by_symbol[symbol]

        future = candles[
            candles["timestamp"]
            > signal_time
        ]

        if future.empty:
            continue

        entry = future.iloc[0]

        if (
            entry["timestamp"].date()
            != signal_time.date()
        ):
            continue

        entry_time = entry["timestamp"]
        entry_price = entry["open"]

        target = (
            entry_price
            * (1 + TARGET_PCT)
        )

        stop = (
            entry_price
            * (1 - STOP_PCT)
        )

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

        outcome = "TIMEOUT"
        exit_price = (
            trade_candles
            .iloc[-1]["close"]
        )
        exit_time = (
            trade_candles
            .iloc[-1]["timestamp"]
        )

        for _, candle in trade_candles.iterrows():

            target_hit = (
                candle["high"]
                >= target
            )

            stop_hit = (
                candle["low"]
                <= stop
            )

            # Conservative:
            # if both happen in the same candle,
            # assume stop first.
            if target_hit and stop_hit:
                outcome = "STOP"
                exit_price = stop
                exit_time = candle["timestamp"]
                break

            if stop_hit:
                outcome = "STOP"
                exit_price = stop
                exit_time = candle["timestamp"]
                break

            if target_hit:
                outcome = "TARGET"
                exit_price = target
                exit_time = candle["timestamp"]
                break

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
                "outcome": outcome,
                "return_pct": return_pct
            }
        )

        last_exit = exit_time

    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
        }

    trades = pd.DataFrame(trades)

    wins = int(
        (
            trades["outcome"]
            == "TARGET"
        ).sum()
    )

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": (
            wins
            / len(trades)
            * 100
        ),
        "avg_return": (
            trades["return_pct"].mean()
        ),
    }


# --------------------------------------------------
# 1. MODEL THRESHOLD QUALITY
#    NO SETUP / NO FIDELITY
# --------------------------------------------------

print("\n================================")
print("1. RAW MODEL THRESHOLD QUALITY")
print("NO SETUP / NO FIDELITY")
print("================================")

rows = []

for option_type in ["CE", "PE"]:

    for threshold in THRESHOLDS:

        rows.append(
            summarize_prediction_layer(
                predictions,
                threshold,
                option_type,
                "raw_model"
            )
        )

raw_threshold_df = pd.DataFrame(rows)

for option_type in ["CE", "PE"]:

    print(f"\n{option_type}")

    side = raw_threshold_df[
        raw_threshold_df["option_type"]
        == option_type
    ]

    print(
        side[
            [
                "threshold",
                "signals",
                "actual_wins",
                "actual_win_rate",
                "avg_probability",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x: f"{x:.2f}",
                "actual_win_rate":
                    lambda x: f"{x:.2f}%",
                "avg_probability":
                    lambda x: f"{x:.4f}",
            }
        )
    )


# --------------------------------------------------
# 2. BACKTEST — MODEL ONLY
# --------------------------------------------------

print("\n================================")
print("2. BACKTEST — MODEL ONLY")
print("NO SETUP / NO FIDELITY")
print("================================")

for option_type in ["CE", "PE"]:

    print(f"\n{option_type}")

    for threshold in THRESHOLDS:

        result = backtest_threshold(
            threshold=threshold,
            option_type=option_type,
            require_setup=False,
            require_fidelity=False,
        )

        print(
            f"{threshold:.2f} | "
            f"Trades: {result['trades']} | "
            f"Wins: {result['wins']} | "
            f"Win Rate: {result['win_rate']:.2f}% | "
            f"Avg Return: {result['avg_return']:.2f}%"
        )


# --------------------------------------------------
# 3. BACKTEST — + SETUP
# --------------------------------------------------

print("\n================================")
print("3. BACKTEST — + SETUP FILTER")
print("================================")

for option_type in ["CE", "PE"]:

    print(f"\n{option_type}")

    for threshold in THRESHOLDS:

        result = backtest_threshold(
            threshold=threshold,
            option_type=option_type,
            require_setup=True,
            require_fidelity=False,
        )

        print(
            f"{threshold:.2f} | "
            f"Trades: {result['trades']} | "
            f"Wins: {result['wins']} | "
            f"Win Rate: {result['win_rate']:.2f}% | "
            f"Avg Return: {result['avg_return']:.2f}%"
        )


# --------------------------------------------------
# 4. BACKTEST — + SETUP + FIDELITY
# --------------------------------------------------

print("\n================================")
print("4. BACKTEST — + SETUP + FIDELITY")
print("================================")

for option_type in ["CE", "PE"]:

    print(f"\n{option_type}")

    for threshold in THRESHOLDS:

        result = backtest_threshold(
            threshold=threshold,
            option_type=option_type,
            require_setup=True,
            require_fidelity=True,
        )

        print(
            f"{threshold:.2f} | "
            f"Trades: {result['trades']} | "
            f"Wins: {result['wins']} | "
            f"Win Rate: {result['win_rate']:.2f}% | "
            f"Avg Return: {result['avg_return']:.2f}%"
        )


print("\n================================")
print("THRESHOLD TUNING COMPLETE")
print("================================")
