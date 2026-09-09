from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from technical_features import build_features


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_PATH = Path(
    "data/raw/candles_1m.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/training_dataset.parquet"
)

# --------------------------------------------------
# CONTROLLED 10PCT vs 5PCT EXPERIMENT
#
# Same raw candles
# Same technical features
# Same signal time
# Same next-candle-open entry
# Same 30-minute hold
# Same -5% stop
#
# Only the profit target changes:
#
# 10PCT strategy = +10% target / -5% stop
# 5PCT strategy  =  +5% target / -5% stop
# --------------------------------------------------

TARGET_PCT_10PCT = 0.10
STOP_PCT_10PCT = 0.05

TARGET_PCT_5PCT = 0.05
STOP_PCT_5PCT = 0.05

MAX_HOLD_MINUTES = 30

NANOSECONDS_PER_MINUTE = (
    60
    * 1_000_000_000
)


# --------------------------------------------------
# LOAD RAW DATA
# --------------------------------------------------

df = pd.read_parquet(
    INPUT_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

# Keep expiry available for time-decay / theta features.
if "expiry" in df.columns:
    df["expiry"] = pd.to_datetime(
        df["expiry"],
        utc=True,
        errors="coerce",
    )

df = (
    df
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
# BUILD TECHNICAL FEATURES
#
# RSI
# ATR
# EMA trend
# VWAP
# Pivot
# Breakout
# Volume
# CE vs PE strength
# MACD
# LVG
# Time to expiry / theta
# Fidelity chart-structure layer
# Setup score
# etc.
# --------------------------------------------------

print(
    "Building technical features..."
)

options = build_features(
    df
)

print(
    "Feature rows:",
    len(options),
)


# --------------------------------------------------
# FAST TARGET / STOP LABELING
#
# Signal = current candle close
# Entry  = next candle open
#
# This generic function is called twice:
#
# 10PCT = +10% target / -5% stop
# 5PCT  =  +5% target / -5% stop
#
# TIMEOUT = neither target nor stop within
#           30 real minutes.
# --------------------------------------------------

@njit
def calculate_labels(
    timestamps,
    opens,
    highs,
    lows,
    target_pct,
    stop_pct,
    hold_minutes,
):
    n = len(
        opens
    )

    labels = np.zeros(
        n,
        dtype=np.int8,
    )

    outcomes = np.zeros(
        n,
        dtype=np.int8,
    )

    entry_prices = np.empty(
        n,
        dtype=np.float64,
    )

    entry_prices[:] = np.nan

    for i in range(
        n - 1
    ):

        # Signal happens at current candle close.
        # Enter the next available candle open.
        entry_index = (
            i + 1
        )

        entry_price = opens[
            entry_index
        ]

        entry_time = timestamps[
            entry_index
        ]

        entry_prices[
            i
        ] = entry_price

        target_price = (
            entry_price
            * (
                1.0
                + target_pct
            )
        )

        stop_price = (
            entry_price
            * (
                1.0
                - stop_pct
            )
        )

        end_time = (
            entry_time
            + hold_minutes
            * NANOSECONDS_PER_MINUTE
        )

        # outcomes:
        # 0 = TIMEOUT
        # 1 = TARGET
        # 2 = STOP

        for j in range(
            entry_index,
            n,
        ):

            if (
                timestamps[j]
                > end_time
            ):
                break

            target_hit = (
                highs[j]
                >= target_price
            )

            stop_hit = (
                lows[j]
                <= stop_price
            )

            # Conservative assumption:
            # if target and stop are both touched
            # inside the same 1-minute candle,
            # assume STOP happened first.
            if (
                target_hit
                and stop_hit
            ):
                outcomes[i] = 2
                labels[i] = 0
                break

            if stop_hit:
                outcomes[i] = 2
                labels[i] = 0
                break

            if target_hit:
                outcomes[i] = 1
                labels[i] = 1
                break

    return (
        labels,
        outcomes,
        entry_prices,
    )


# --------------------------------------------------
# LABEL ONE OPTION / ONE TRADING DAY
# --------------------------------------------------

def create_labels_fast(
    group
):
    group = (
        group
        .sort_values(
            "timestamp"
        )
        .copy()
    )

    timestamps = (
        group["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    opens = (
        group["open"]
        .to_numpy(
            dtype=np.float64
        )
    )

    highs = (
        group["high"]
        .to_numpy(
            dtype=np.float64
        )
    )

    lows = (
        group["low"]
        .to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------
    # 10PCT LABELS: +10% TARGET / -5% STOP
    # --------------------------------------------------

    (
        labels_10pct,
        outcomes_10pct,
        entry_prices_10pct,
    ) = calculate_labels(
        timestamps,
        opens,
        highs,
        lows,
        TARGET_PCT_10PCT,
        STOP_PCT_10PCT,
        MAX_HOLD_MINUTES,
    )

    # --------------------------------------------------
    # 5PCT LABELS: +5% TARGET / -5% STOP
    # --------------------------------------------------

    (
        labels_5pct,
        outcomes_5pct,
        entry_prices_5pct,
    ) = calculate_labels(
        timestamps,
        opens,
        highs,
        lows,
        TARGET_PCT_5PCT,
        STOP_PCT_5PCT,
        MAX_HOLD_MINUTES,
    )

    # Entry is identical for both experiments because
    # both use the next available candle open.
    group[
        "entry_price"
    ] = entry_prices_5pct

    # Sanity-check the controlled experiment.
    same_entry = (
        np.isnan(
            entry_prices_10pct
        )
        & np.isnan(
            entry_prices_5pct
        )
    ) | np.isclose(
        entry_prices_10pct,
        entry_prices_5pct,
        equal_nan=True,
    )

    if not bool(
        np.all(
            same_entry
        )
    ):
        raise ValueError(
            "10PCT and 5PCT entry prices "
            "do not match."
        )

    outcome_map = {
        0: "TIMEOUT",
        1: "TARGET",
        2: "STOP",
    }

    # --------------------------------------------------
    # EXPLICIT 10PCT GROUND TRUTH
    # --------------------------------------------------

    group[
        "target_hit_10pct"
    ] = labels_10pct

    group[
        "trade_outcome_10pct"
    ] = [
        outcome_map[x]
        for x in outcomes_10pct
    ]

    # --------------------------------------------------
    # EXPLICIT 5PCT GROUND TRUTH
    # --------------------------------------------------

    group[
        "target_hit_5pct"
    ] = labels_5pct

    group[
        "trade_outcome_5pct"
    ] = [
        outcome_map[x]
        for x in outcomes_5pct
    ]

    # --------------------------------------------------
    # TEMPORARY BACKWARD-COMPATIBILITY ALIASES
    #
    # Existing 03 / 09 currently expect:
    #   trade_outcome
    #   target_hit
    #
    # Keep them mapped to the 5PCT labels until those
    # scripts are updated for explicit dual training.
    # --------------------------------------------------

    group[
        "target_hit"
    ] = group[
        "target_hit_5pct"
    ]

    group[
        "trade_outcome"
    ] = group[
        "trade_outcome_5pct"
    ]

    return group


# --------------------------------------------------
# LABEL ALL OPTION/DAY GROUPS
#
# Grouping by symbol + date guarantees that the
# next-candle entry and 30-minute evaluation never
# cross into another trading day.
# --------------------------------------------------

print()
print(
    "Creating 10PCT and 5PCT trade labels..."
)

labelled_groups = []

for _, group in options.groupby(
    [
        "symbol",
        "date",
    ],
    sort=False,
):

    labelled_groups.append(
        create_labels_fast(
            group
        )
    )

options = pd.concat(
    labelled_groups,
    ignore_index=True,
)


# --------------------------------------------------
# FINAL TRAINING DATASET
# --------------------------------------------------

feature_columns = [

    # Identity
    "timestamp",
    "date",
    "symbol",
    "instrument_type",
    "strike",
    "expiry",

    # Raw option candle
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Option momentum
    "return_1m",
    "return_3m",
    "return_5m",
    "range_pct",
    "body_pct",
    "candle_strength",

    # Technical indicators
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

    # MACD
    "macd_line",
    "macd_signal",
    "macd_histogram",

    # LVG
    "lvg_score",
    "lvg_detected",

    # Time decay / theta
    "time_to_expiry_minutes",
    "time_to_expiry_days",
    "theta",

    # Fidelity / chart structure
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

    # NIFTY context
    "nifty_close",

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

    # NIFTY Fidelity / chart structure
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

    # Pivot
    "daily_pivot",
    "pivot_r1",
    "pivot_s1",
    "distance_from_pivot_pct",

    # Strike / moneyness
    "distance_from_strike",
    "distance_from_strike_pct",
    "premium_pct_of_spot",

    # CE / PE relationship
    "ce_close",
    "pe_close",
    "ce_pe_ratio",
    "ce_return_3m",
    "pe_return_3m",
    "ce_minus_pe_return_3m",

    # Setup logic
    "setup_score",
    "candidate_setup",
    "is_ce",

    # Shared entry
    "entry_price",

    # 10PCT ground truth
    "trade_outcome_10pct",
    "target_hit_10pct",

    # 5PCT ground truth
    "trade_outcome_5pct",
    "target_hit_5pct",

    # Temporary backward-compatible aliases
    "trade_outcome",
    "target_hit",
]


# --------------------------------------------------
# VALIDATE FEATURE OUTPUT
# --------------------------------------------------

missing_feature_columns = [
    column
    for column in feature_columns
    if column not in options.columns
]

if missing_feature_columns:
    raise ValueError(
        "build_features()/labeling is missing "
        "required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column
            in missing_feature_columns
        )
    )


dataset = options[
    feature_columns
].copy()


# --------------------------------------------------
# NAN DIAGNOSTICS
# --------------------------------------------------

print()
print(
    "NaN counts by column:"
)

nan_counts = (
    dataset
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    nan_counts[
        nan_counts > 0
    ]
)

print()
print(
    "NaN percentage by column:"
)

nan_pct = (
    dataset
    .isna()
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

print(
    nan_pct[
        nan_pct > 0
    ].round(2)
)


# --------------------------------------------------
# REMOVE INCOMPLETE FEATURE ROWS
# --------------------------------------------------

before_drop = len(
    dataset
)

dataset = (
    dataset
    .dropna()
    .reset_index(
        drop=True
    )
)

print()
print(
    "Dropped incomplete rows:",
    before_drop
    - len(dataset),
)


# --------------------------------------------------
# CONTROLLED-EXPERIMENT SANITY CHECKS
# --------------------------------------------------

# A +10% target cannot logically succeed if the
# +5% target failed under the same entry, stop and
# future-candle path.
impossible_10pct_success = (
    (
        dataset[
            "target_hit_10pct"
        ]
        == 1
    )
    & (
        dataset[
            "target_hit_5pct"
        ]
        == 0
    )
)

if impossible_10pct_success.any():
    raise ValueError(
        "Label sanity check failed: "
        "10PCT target succeeded where "
        "5PCT target did not."
    )

print(
    "Label sanity check: PASSED"
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

dataset.to_parquet(
    OUTPUT_PATH,
    index=False,
)


# --------------------------------------------------
# SUMMARY HELPERS
# --------------------------------------------------

def print_strategy_summary(
    strategy_name,
    outcome_column,
    hit_column,
    target_pct,
    stop_pct,
):
    print()
    print(
        "-" * 48
    )
    print(
        f"{strategy_name} LABEL SUMMARY"
    )
    print(
        "-" * 48
    )

    print(
        "Target / Stop:",
        f"+{target_pct * 100:.0f}% / "
        f"-{stop_pct * 100:.0f}%",
    )

    print()
    print(
        "Outcomes:"
    )

    print(
        dataset[
            outcome_column
        ].value_counts()
    )

    hits = int(
        dataset[
            hit_column
        ].sum()
    )

    total = len(
        dataset
    )

    print()
    print(
        "Successful target trades:",
        hits,
    )

    if total:
        print(
            "Natural success rate:",
            f"{hits / total * 100:.2f}%",
        )

    for option_type in [
        "CE",
        "PE",
    ]:
        side = dataset[
            dataset[
                "instrument_type"
            ]
            == option_type
        ]

        if side.empty:
            continue

        side_hits = int(
            side[
                hit_column
            ].sum()
        )

        print(
            f"{option_type} success rate:",
            f"{side_hits / len(side) * 100:.2f}%",
            f"({side_hits:,}/{len(side):,})",
        )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print(
    "=" * 48
)
print(
    "TRAINING DATASET COMPLETE"
)
print(
    "=" * 48
)

print(
    "Rows:",
    len(dataset),
)

print(
    "Options:",
    dataset[
        "symbol"
    ].nunique(),
)

print(
    "Trading days:",
    dataset[
        "date"
    ].nunique(),
)

print_strategy_summary(
    strategy_name="10PCT",
    outcome_column=(
        "trade_outcome_10pct"
    ),
    hit_column=(
        "target_hit_10pct"
    ),
    target_pct=(
        TARGET_PCT_10PCT
    ),
    stop_pct=(
        STOP_PCT_10PCT
    ),
)

print_strategy_summary(
    strategy_name="5PCT",
    outcome_column=(
        "trade_outcome_5pct"
    ),
    hit_column=(
        "target_hit_5pct"
    ),
    target_pct=(
        TARGET_PCT_5PCT
    ),
    stop_pct=(
        STOP_PCT_5PCT
    ),
)


# --------------------------------------------------
# DIRECT 10PCT vs 5PCT LABEL COMPARISON
# --------------------------------------------------

total = len(
    dataset
)

hits_10pct = int(
    dataset[
        "target_hit_10pct"
    ].sum()
)

hits_5pct = int(
    dataset[
        "target_hit_5pct"
    ].sum()
)

rate_10pct = (
    hits_10pct
    / total
    * 100
    if total
    else 0.0
)

rate_5pct = (
    hits_5pct
    / total
    * 100
    if total
    else 0.0
)

print()
print(
    "=" * 48
)
print(
    "10PCT vs 5PCT NATURAL LABEL COMPARISON"
)
print(
    "=" * 48
)

print(
    f"{'Metric':<24}"
    f"{'10PCT':>12}"
    f"{'5PCT':>12}"
)

print(
    "-" * 48
)

print(
    f"{'Successful targets':<24}"
    f"{hits_10pct:>12,}"
    f"{hits_5pct:>12,}"
)

print(
    f"{'Success rate':<24}"
    f"{rate_10pct:>11.2f}%"
    f"{rate_5pct:>11.2f}%"
)


# --------------------------------------------------
# OTHER FEATURE DIAGNOSTICS
# --------------------------------------------------

print()
print(
    "Setup score distribution:"
)

print(
    dataset[
        "setup_score"
    ]
    .value_counts()
    .sort_index()
)


candidate_count = int(
    dataset[
        "candidate_setup"
    ].sum()
)

print()
print(
    "Candidate setups:",
    candidate_count,
)

if total:
    print(
        "Candidate setup rate:",
        f"{candidate_count / total * 100:.2f}%",
    )


print()
print(
    "Fidelity confirmations:"
)

print(
    dataset[
        "fidelity_confirmed"
    ]
    .value_counts()
    .sort_index()
)


print()
print(
    "LVG detections:"
)

print(
    dataset[
        "lvg_detected"
    ]
    .value_counts()
    .sort_index()
)


print()
print(
    "Saved:",
    OUTPUT_PATH,
)

print()
print(
    "Ground-truth columns created:"
)

print(
    "  trade_outcome_10pct"
)
print(
    "  target_hit_10pct"
)
print(
    "  trade_outcome_5pct"
)
print(
    "  target_hit_5pct"
)
