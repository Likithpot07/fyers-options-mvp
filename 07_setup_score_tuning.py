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

CONFIDENCE_THRESHOLDS = [
    0.75,
    0.80,
    0.85,
    0.90,
]

SETUP_THRESHOLDS = [
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
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
# BACKTEST
# --------------------------------------------------

def run_backtest(
    option_type,
    confidence,
    min_setup_score,
    require_fidelity=False,
):

    signals = predictions[
        (predictions["instrument_type"] == option_type)
        & (
            predictions["prediction_probability"]
            >= confidence
        )
        & (
            predictions["setup_score"]
            >= min_setup_score
        )
    ].copy()

    if require_fidelity:
        signals = signals[
            signals["fidelity_confirmed"] == 1
        ].copy()

    raw_signal_count = len(signals)

    # Best candidate for this side at each minute
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

    best_candidate_count = len(signals)

    trades = []
    last_exit = None
    overlap_rejected = 0

    for _, signal in signals.iterrows():

        signal_time = signal["timestamp"]
        symbol = signal["symbol"]

        if (
            last_exit is not None
            and signal_time <= last_exit
        ):
            overlap_rejected += 1
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
            "option_type": option_type,
            "confidence": confidence,
            "setup_score": min_setup_score,
            "fidelity": require_fidelity,
            "raw_signals": raw_signal_count,
            "best_candidates": best_candidate_count,
            "overlap_rejected": overlap_rejected,
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
        "option_type": option_type,
        "confidence": confidence,
        "setup_score": min_setup_score,
        "fidelity": require_fidelity,
        "raw_signals": raw_signal_count,
        "best_candidates": best_candidate_count,
        "overlap_rejected": overlap_rejected,
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
# RUN GRID
# --------------------------------------------------

rows = []

for option_type in ["CE", "PE"]:

    for confidence in CONFIDENCE_THRESHOLDS:

        for setup_score in SETUP_THRESHOLDS:

            rows.append(
                run_backtest(
                    option_type=option_type,
                    confidence=confidence,
                    min_setup_score=setup_score,
                    require_fidelity=False,
                )
            )

results = pd.DataFrame(rows)


# --------------------------------------------------
# PRINT — NO FIDELITY
# --------------------------------------------------

print("\n================================")
print("SETUP SCORE TUNING — NO FIDELITY")
print("================================")

for option_type in ["CE", "PE"]:

    for confidence in CONFIDENCE_THRESHOLDS:

        print(
            f"\n{option_type} | "
            f"Confidence >= {confidence:.2f}"
        )

        side = results[
            (results["option_type"] == option_type)
            & (
                results["confidence"]
                == confidence
            )
        ]

        print(
            side[
                [
                    "setup_score",
                    "raw_signals",
                    "best_candidates",
                    "trades",
                    "wins",
                    "win_rate",
                    "avg_return",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "win_rate":
                        lambda x: f"{x:.2f}%",
                    "avg_return":
                        lambda x: f"{x:.2f}%",
                }
            )
        )


# --------------------------------------------------
# FIDELITY CHECK AT BEST-USEFUL CONFIDENCES
# --------------------------------------------------

print("\n================================")
print("FIDELITY COMPARISON")
print("================================")

fidelity_rows = []

for option_type in ["CE", "PE"]:

    for confidence in [0.80, 0.85]:

        for setup_score in SETUP_THRESHOLDS:

            fidelity_rows.append(
                run_backtest(
                    option_type=option_type,
                    confidence=confidence,
                    min_setup_score=setup_score,
                    require_fidelity=True,
                )
            )

fidelity_results = pd.DataFrame(
    fidelity_rows
)

for option_type in ["CE", "PE"]:

    for confidence in [0.80, 0.85]:

        print(
            f"\n{option_type} | "
            f"Confidence >= {confidence:.2f} | "
            "Fidelity required"
        )

        side = fidelity_results[
            (
                fidelity_results["option_type"]
                == option_type
            )
            & (
                fidelity_results["confidence"]
                == confidence
            )
        ]

        print(
            side[
                [
                    "setup_score",
                    "raw_signals",
                    "best_candidates",
                    "trades",
                    "wins",
                    "win_rate",
                    "avg_return",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "win_rate":
                        lambda x: f"{x:.2f}%",
                    "avg_return":
                        lambda x: f"{x:.2f}%",
                }
            )
        )


print("\n================================")
print("SETUP SCORE TUNING COMPLETE")
print("================================")
