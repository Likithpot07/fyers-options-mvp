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

OUTPUT_PARQUET_PATH = Path(
    "data/processed/backtest_comparison.parquet"
)

OUTPUT_CSV_PATH = Path(
    "data/processed/backtest_comparison.csv"
)

# Test both thresholds used in 03.
CONFIDENCE_THRESHOLDS = [
    0.80,
    0.85,
]

# Match current live signal filtering.
SETUP_SCORE_THRESHOLD = 6
REQUIRE_FIDELITY = True

# Controlled experiment from 02 / 03.
STRATEGIES = {
    "10PCT": {
        "probability_column": "prediction_probability_10pct",
        "target_hit_column": "target_hit_10pct",
        "target_pct": 0.10,
        "stop_pct": 0.05,
    },
    "5PCT": {
        "probability_column": "prediction_probability_5pct",
        "target_hit_column": "target_hit_5pct",
        "target_pct": 0.05,
        "stop_pct": 0.05,
    },
}

MAX_HOLD_MINUTES = 30


# --------------------------------------------------
# VALIDATE FILES
# --------------------------------------------------

for path in [
    PREDICTIONS_PATH,
    RAW_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


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
    utc=True,
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    utc=True,
)


# --------------------------------------------------
# VALIDATE PREDICTION COLUMNS
# --------------------------------------------------

required_prediction_columns = [
    "timestamp",
    "symbol",
    "instrument_type",
    "strike",
    "setup_score",
    "fidelity_confirmed",
    "prediction_probability_10pct",
    "prediction_probability_5pct",
    "target_hit_10pct",
    "target_hit_5pct",
]

missing_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

if missing_columns:
    raise ValueError(
        "test_predictions.parquet is missing columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# --------------------------------------------------
# KEEP CE / PE RAW CANDLES ONLY
# --------------------------------------------------

raw = raw[
    raw["instrument_type"].isin(
        [
            "CE",
            "PE",
        ]
    )
].copy()

raw = (
    raw
    .sort_values(
        [
            "symbol",
            "timestamp",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# TEST PERIOD
# --------------------------------------------------

print(
    "=" * 76
)

print(
    "HISTORICAL TRADE BACKTEST"
)

print(
    "=" * 76
)

print(
    "Test period:",
    predictions["timestamp"].min(),
    "->",
    predictions["timestamp"].max(),
)

print(
    "Prediction rows:",
    f"{len(predictions):,}",
)

print()

print(
    "10PCT = +10% target / -5% stop"
)

print(
    "5PCT  =  +5% target / -5% stop"
)

print(
    "Max hold:",
    f"{MAX_HOLD_MINUTES} minutes",
)

print(
    "Setup score >=",
    SETUP_SCORE_THRESHOLD,
)

print(
    "Fidelity required:",
    REQUIRE_FIDELITY,
)


# --------------------------------------------------
# RAW DATA BY SYMBOL
# --------------------------------------------------

raw_by_symbol = {
    symbol: group.reset_index(
        drop=True
    )
    for symbol, group
    in raw.groupby(
        "symbol"
    )
}


# --------------------------------------------------
# CREATE SIGNALS
# --------------------------------------------------

def create_signals(
    strategy_name,
    strategy_config,
    threshold,
):
    probability_column = (
        strategy_config[
            "probability_column"
        ]
    )

    mask = (
        predictions[
            probability_column
        ]
        >= threshold
    )

    mask &= (
        predictions[
            "setup_score"
        ]
        >= SETUP_SCORE_THRESHOLD
    )

    if REQUIRE_FIDELITY:
        mask &= (
            predictions[
                "fidelity_confirmed"
            ]
            .fillna(0)
            .astype(int)
            == 1
        )

    signals = predictions[
        mask
    ].copy()

    raw_signal_count = len(
        signals
    )

    if signals.empty:
        print(
            f"{strategy_name} @ {threshold:.2f} "
            f"| raw signals: 0"
        )

        return signals

    # --------------------------------------------------
    # PICK ONLY THE BEST CONTRACT AT EACH MINUTE
    #
    # A historical minute can contain multiple strikes
    # and both CE / PE rows. Mimic one live decision by
    # choosing the single highest-probability row.
    # --------------------------------------------------

    signals = (
        signals
        .sort_values(
            [
                "timestamp",
                probability_column,
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "timestamp",
            as_index=False,
        )
        .first()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"{strategy_name} @ {threshold:.2f} "
        f"| raw approved: {raw_signal_count:,} "
        f"| best/minute: {len(signals):,}"
    )

    return signals


# --------------------------------------------------
# BACKTEST ONE STRATEGY + THRESHOLD
# --------------------------------------------------

def backtest_strategy(
    signals,
    strategy_name,
    strategy_config,
    threshold,
):
    trades = []

    probability_column = (
        strategy_config[
            "probability_column"
        ]
    )

    target_pct = (
        strategy_config[
            "target_pct"
        ]
    )

    stop_pct = (
        strategy_config[
            "stop_pct"
        ]
    )

    # Only one active trade at a time for this exact
    # strategy + threshold configuration.
    last_global_exit = None

    for _, signal in signals.iterrows():

        symbol = signal[
            "symbol"
        ]

        signal_time = signal[
            "timestamp"
        ]

        # Ignore new signals while the previous trade
        # is still active.
        if (
            last_global_exit is not None
            and signal_time
            <= last_global_exit
        ):
            continue

        if symbol not in raw_by_symbol:
            continue

        candles = raw_by_symbol[
            symbol
        ]

        # --------------------------------------------------
        # ENTRY = NEXT AVAILABLE CANDLE OPEN
        # --------------------------------------------------

        future = candles[
            candles[
                "timestamp"
            ]
            > signal_time
        ]

        if future.empty:
            continue

        entry_row = (
            future.iloc[0]
        )

        entry_time = (
            entry_row[
                "timestamp"
            ]
        )

        # Never cross trading day.
        if (
            entry_time.date()
            != signal_time.date()
        ):
            continue

        entry_price = float(
            entry_row[
                "open"
            ]
        )

        target_price = (
            entry_price
            * (
                1
                + target_pct
            )
        )

        stop_price = (
            entry_price
            * (
                1
                - stop_pct
            )
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
            (
                candles[
                    "timestamp"
                ]
                >= entry_time
            )
            & (
                candles[
                    "timestamp"
                ]
                <= end_time
            )
        ].copy()

        trade_candles = trade_candles[
            trade_candles[
                "timestamp"
            ].dt.date
            == entry_time.date()
        ]

        if trade_candles.empty:
            continue

        # --------------------------------------------------
        # WALK FORWARD
        # --------------------------------------------------

        exit_price = None
        exit_time = None
        exit_reason = None

        for _, candle in (
            trade_candles
            .iterrows()
        ):

            target_hit = (
                candle[
                    "high"
                ]
                >= target_price
            )

            stop_hit = (
                candle[
                    "low"
                ]
                <= stop_price
            )

            # Conservative assumption:
            # if both target and stop appear inside the
            # same 1-minute candle, count STOP first.
            if (
                target_hit
                and stop_hit
            ):

                exit_price = (
                    stop_price
                )

                exit_time = (
                    candle[
                        "timestamp"
                    ]
                )

                exit_reason = (
                    "STOP_SAME_CANDLE"
                )

                break

            if stop_hit:

                exit_price = (
                    stop_price
                )

                exit_time = (
                    candle[
                        "timestamp"
                    ]
                )

                exit_reason = (
                    "STOP"
                )

                break

            if target_hit:

                exit_price = (
                    target_price
                )

                exit_time = (
                    candle[
                        "timestamp"
                    ]
                )

                exit_reason = (
                    "TARGET"
                )

                break

        # --------------------------------------------------
        # TIMEOUT
        # --------------------------------------------------

        if exit_price is None:

            last_candle = (
                trade_candles
                .iloc[-1]
            )

            exit_price = float(
                last_candle[
                    "close"
                ]
            )

            exit_time = (
                last_candle[
                    "timestamp"
                ]
            )

            exit_reason = (
                "TIMEOUT"
            )

        net_return_pct = (
            (
                exit_price
                - entry_price
            )
            / entry_price
            * 100
        )

        hold_minutes = (
            (
                exit_time
                - entry_time
            )
            .total_seconds()
            / 60
        )

        trades.append(
            {
                "strategy": (
                    strategy_name
                ),
                "confidence_threshold": (
                    threshold
                ),
                "symbol": (
                    symbol
                ),
                "instrument_type": (
                    signal[
                        "instrument_type"
                    ]
                ),
                "strike": (
                    signal[
                        "strike"
                    ]
                ),
                "signal_time": (
                    signal_time
                ),
                "entry_time": (
                    entry_time
                ),
                "exit_time": (
                    exit_time
                ),
                "prediction_probability": (
                    float(
                        signal[
                            probability_column
                        ]
                    )
                ),
                "setup_score": (
                    int(
                        signal[
                            "setup_score"
                        ]
                    )
                ),
                "fidelity_confirmed": (
                    int(
                        signal[
                            "fidelity_confirmed"
                        ]
                    )
                ),
                "entry_price": (
                    entry_price
                ),
                "target_pct": (
                    target_pct
                ),
                "stop_pct": (
                    stop_pct
                ),
                "target_price": (
                    target_price
                ),
                "stop_price": (
                    stop_price
                ),
                "exit_price": (
                    exit_price
                ),
                "exit_reason": (
                    exit_reason
                ),
                "hold_minutes": (
                    hold_minutes
                ),
                "net_return_pct": (
                    net_return_pct
                ),
            }
        )

        last_global_exit = (
            exit_time
        )

    return pd.DataFrame(
        trades
    )


# --------------------------------------------------
# RUN ALL COMBINATIONS
# --------------------------------------------------

result_frames = []

signal_counts = []

for strategy_name, strategy_config in (
    STRATEGIES.items()
):

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        signals = create_signals(
            strategy_name=(
                strategy_name
            ),
            strategy_config=(
                strategy_config
            ),
            threshold=(
                threshold
            ),
        )

        signal_counts.append(
            {
                "strategy": (
                    strategy_name
                ),
                "confidence_threshold": (
                    threshold
                ),
                "selected_signal_minutes": (
                    len(signals)
                ),
            }
        )

        if signals.empty:
            continue

        strategy_results = (
            backtest_strategy(
                signals=signals,
                strategy_name=(
                    strategy_name
                ),
                strategy_config=(
                    strategy_config
                ),
                threshold=(
                    threshold
                ),
            )
        )

        if not strategy_results.empty:
            result_frames.append(
                strategy_results
            )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if not result_frames:

    print()
    print(
        "No trades generated."
    )

    raise SystemExit


results = pd.concat(
    result_frames,
    ignore_index=True,
)

results = (
    results
    .sort_values(
        [
            "confidence_threshold",
            "strategy",
            "signal_time",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PARQUET_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_parquet(
    OUTPUT_PARQUET_PATH,
    index=False,
)

results.to_csv(
    OUTPUT_CSV_PATH,
    index=False,
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def summarize_group(
    group
):
    total = len(
        group
    )

    targets = int(
        (
            group[
                "exit_reason"
            ]
            == "TARGET"
        ).sum()
    )

    stops = int(
        group[
            "exit_reason"
        ]
        .isin(
            [
                "STOP",
                "STOP_SAME_CANDLE",
            ]
        )
        .sum()
    )

    timeouts = int(
        (
            group[
                "exit_reason"
            ]
            == "TIMEOUT"
        ).sum()
    )

    win_rate = (
        targets
        / total
        * 100
        if total
        else 0.0
    )

    avg_return = (
        group[
            "net_return_pct"
        ]
        .mean()
        if total
        else 0.0
    )

    total_return = (
        group[
            "net_return_pct"
        ]
        .sum()
        if total
        else 0.0
    )

    avg_probability = (
        group[
            "prediction_probability"
        ]
        .mean()
        if total
        else 0.0
    )

    return pd.Series(
        {
            "trades": total,
            "targets": targets,
            "stops": stops,
            "timeouts": timeouts,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_return": total_return,
            "avg_probability": avg_probability,
        }
    )


summary = (
    results
    .groupby(
        [
            "confidence_threshold",
            "strategy",
        ]
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)


# --------------------------------------------------
# PRINT MAIN COMPARISON
# --------------------------------------------------

print()
print(
    "=" * 96
)

print(
    "BACKTEST RESULTS: 10PCT vs 5PCT"
)

print(
    "=" * 96
)

print(
    f"{'Threshold':>10} "
    f"{'Strategy':>10} "
    f"{'Trades':>8} "
    f"{'Targets':>8} "
    f"{'Stops':>8} "
    f"{'Timeout':>8} "
    f"{'Win Rate':>10} "
    f"{'Avg Ret':>10} "
    f"{'Total Ret':>11}"
)

print(
    "-" * 96
)

for _, row in summary.iterrows():

    print(
        f"{row['confidence_threshold']:>10.2f} "
        f"{row['strategy']:>10} "
        f"{int(row['trades']):>8} "
        f"{int(row['targets']):>8} "
        f"{int(row['stops']):>8} "
        f"{int(row['timeouts']):>8} "
        f"{row['win_rate']:>9.2f}% "
        f"{row['avg_return']:>9.2f}% "
        f"{row['total_return']:>10.2f}%"
    )


# --------------------------------------------------
# CE / PE BREAKDOWN
# --------------------------------------------------

side_summary = (
    results
    .groupby(
        [
            "confidence_threshold",
            "strategy",
            "instrument_type",
        ]
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)

print()
print(
    "=" * 96
)

print(
    "CE / PE BREAKDOWN"
)

print(
    "=" * 96
)

print(
    side_summary[
        [
            "confidence_threshold",
            "strategy",
            "instrument_type",
            "trades",
            "targets",
            "stops",
            "timeouts",
            "win_rate",
            "avg_return",
        ]
    ].to_string(
        index=False,
        formatters={
            "trades": (
                lambda x: f"{int(x)}"
            ),
            "targets": (
                lambda x: f"{int(x)}"
            ),
            "stops": (
                lambda x: f"{int(x)}"
            ),
            "timeouts": (
                lambda x: f"{int(x)}"
            ),
            "win_rate": (
                lambda x: f"{x:.2f}%"
            ),
            "avg_return": (
                lambda x: f"{x:.2f}%"
            ),
        },
    )
)


# --------------------------------------------------
# SIGNAL / TRADE COUNTS
# --------------------------------------------------

signal_count_df = pd.DataFrame(
    signal_counts
)

trade_count_df = (
    results
    .groupby(
        [
            "strategy",
            "confidence_threshold",
        ]
    )
    .size()
    .reset_index(
        name="executed_trades"
    )
)

signal_trade_summary = (
    signal_count_df
    .merge(
        trade_count_df,
        on=[
            "strategy",
            "confidence_threshold",
        ],
        how="left",
    )
)

signal_trade_summary[
    "executed_trades"
] = (
    signal_trade_summary[
        "executed_trades"
    ]
    .fillna(0)
    .astype(int)
)

print()
print(
    "=" * 76
)

print(
    "SIGNALS vs EXECUTED TRADES"
)

print(
    "=" * 76
)

print(
    signal_trade_summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# FINAL
# --------------------------------------------------

print()
print(
    "Saved:"
)

print(
    OUTPUT_PARQUET_PATH
)

print(
    OUTPUT_CSV_PATH
)
