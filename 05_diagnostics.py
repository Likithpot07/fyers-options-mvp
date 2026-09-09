from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TRAINING_PATH = Path(
    "data/processed/training_dataset.parquet"
)

PREDICTIONS_PATH = Path(
    "data/processed/test_predictions.parquet"
)

BACKTEST_PATH = Path(
    "data/processed/backtest_results.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_10pct.json"
)

CONFIDENCE_THRESHOLD = 0.80


# --------------------------------------------------
# LOAD
# --------------------------------------------------

training = pd.read_parquet(TRAINING_PATH)
predictions = pd.read_parquet(PREDICTIONS_PATH)
backtest = pd.read_parquet(BACKTEST_PATH)

training["timestamp"] = pd.to_datetime(
    training["timestamp"],
    utc=True
)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"],
    utc=True
)

backtest["signal_time"] = pd.to_datetime(
    backtest["signal_time"],
    utc=True
)

training["date"] = training["timestamp"].dt.date
predictions["date"] = predictions["timestamp"].dt.date
backtest["date"] = backtest["signal_time"].dt.date


# --------------------------------------------------
# 1. RAW LABEL DISTRIBUTION
# --------------------------------------------------

print("\n================================")
print("1. NATURAL TARGET RATE")
print("================================")

natural = (
    training
    .groupby("instrument_type")
    .agg(
        rows=("target_hit", "count"),
        wins=("target_hit", "sum")
    )
)

natural["success_rate"] = (
    natural["wins"]
    / natural["rows"]
    * 100
)

print(natural)


# --------------------------------------------------
# 2. MODEL SIGNALS >= 80%
# --------------------------------------------------

print("\n================================")
print("2. MODEL SIGNALS >= 80%")
print("================================")

signals = predictions[
    predictions["prediction_probability"]
    >= CONFIDENCE_THRESHOLD
].copy()

signal_summary = (
    signals
    .groupby("instrument_type")
    .agg(
        signals=("target_hit", "count"),
        actual_wins=("target_hit", "sum"),
        avg_probability=(
            "prediction_probability",
            "mean"
        )
    )
)

signal_summary["actual_success_rate"] = (
    signal_summary["actual_wins"]
    / signal_summary["signals"]
    * 100
)

print(signal_summary)


# --------------------------------------------------
# 3. REAL BACKTEST BY CE / PE
# --------------------------------------------------

print("\n================================")
print("3. REAL BACKTEST")
print("================================")

trade_summary = (
    backtest
    .groupby("instrument_type")
    .agg(
        trades=("symbol", "count"),

        targets=(
            "exit_reason",
            lambda x: (x == "TARGET").sum()
        ),

        stops=(
            "exit_reason",
            lambda x: x.isin(
                ["STOP", "STOP_SAME_CANDLE"]
            ).sum()
        ),

        avg_return=(
            "net_return_pct",
            "mean"
        )
    )
)

trade_summary["win_rate"] = (
    trade_summary["targets"]
    / trade_summary["trades"]
    * 100
)

print(trade_summary)


# --------------------------------------------------
# 4. CONFIDENCE CALIBRATION
# --------------------------------------------------

print("\n================================")
print("4. CONFIDENCE CALIBRATION")
print("================================")

bins = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    1.01
]

predictions["confidence_band"] = pd.cut(
    predictions["prediction_probability"],
    bins=bins,
    right=False
)

calibration = (
    predictions
    .groupby(
        "confidence_band",
        observed=True
    )
    .agg(
        predictions=("target_hit", "count"),
        actual_wins=("target_hit", "sum"),
        avg_probability=(
            "prediction_probability",
            "mean"
        )
    )
)

calibration["actual_win_rate"] = (
    calibration["actual_wins"]
    / calibration["predictions"]
    * 100
)

print(calibration)


# --------------------------------------------------
# 5. HIGH CONFIDENCE BACKTEST
# --------------------------------------------------

print("\n================================")
print("5. BACKTEST BY CONFIDENCE")
print("================================")

backtest["confidence_band"] = pd.cut(
    backtest["prediction_probability"],
    bins=[
        0.80,
        0.85,
        0.90,
        0.95,
        1.01
    ],
    right=False
)

bt_confidence = (
    backtest
    .groupby(
        "confidence_band",
        observed=True
    )
    .agg(
        trades=("symbol", "count"),

        targets=(
            "exit_reason",
            lambda x: (x == "TARGET").sum()
        ),

        avg_return=(
            "net_return_pct",
            "mean"
        )
    )
)

bt_confidence["win_rate"] = (
    bt_confidence["targets"]
    / bt_confidence["trades"]
    * 100
)

print(bt_confidence)


# --------------------------------------------------
# 6. SIGNAL OVERLAP
# --------------------------------------------------

print("\n================================")
print("6. SIGNAL OVERLAP")
print("================================")

# How many high-confidence signals occur
# for same option in same 30-minute period?

signals = signals.sort_values(
    ["symbol", "timestamp"]
)

overlap_counts = []

for symbol, group in signals.groupby("symbol"):

    times = group["timestamp"].tolist()

    for timestamp in times:

        end = timestamp + pd.Timedelta(
            minutes=30
        )

        count = sum(
            1
            for t in times
            if timestamp <= t <= end
        )

        overlap_counts.append(count)


if overlap_counts:

    overlap_series = pd.Series(
        overlap_counts
    )

    print(
        "Average signals inside 30m window:",
        round(overlap_series.mean(), 2)
    )

    print(
        "Median:",
        round(overlap_series.median(), 2)
    )

    print(
        "Maximum:",
        overlap_series.max()
    )


# --------------------------------------------------
# 7. SIGNALS PER DAY
# --------------------------------------------------

print("\n================================")
print("7. SIGNALS PER DAY")
print("================================")

daily_signals = (
    signals
    .groupby("date")
    .size()
)

print(daily_signals)

print(
    "\nAverage signals/day:",
    round(daily_signals.mean(), 2)
)


# --------------------------------------------------
# 8. ACTUAL TRADES PER DAY
# --------------------------------------------------

print("\n================================")
print("8. REAL TRADES PER DAY")
print("================================")

daily_trades = (
    backtest
    .groupby("date")
    .agg(
        trades=("symbol", "count"),

        targets=(
            "exit_reason",
            lambda x: (x == "TARGET").sum()
        ),

        avg_return=(
            "net_return_pct",
            "mean"
        )
    )
)

daily_trades["win_rate"] = (
    daily_trades["targets"]
    / daily_trades["trades"]
    * 100
)

print(daily_trades)


# --------------------------------------------------
# 9. BEST / WORST STRIKES
# --------------------------------------------------

print("\n================================")
print("9. PERFORMANCE BY STRIKE")
print("================================")

strike_summary = (
    backtest
    .groupby(
        ["instrument_type", "strike"]
    )
    .agg(
        trades=("symbol", "count"),

        targets=(
            "exit_reason",
            lambda x: (x == "TARGET").sum()
        ),

        avg_return=(
            "net_return_pct",
            "mean"
        )
    )
)

strike_summary["win_rate"] = (
    strike_summary["targets"]
    / strike_summary["trades"]
    * 100
)

print(
    strike_summary.sort_values(
        "win_rate",
        ascending=False
    )
)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n================================")
print("10. XGBOOST FEATURE IMPORTANCE")
print("================================")

model = XGBClassifier()
model.load_model(MODEL_PATH)

booster = model.get_booster()

importance = booster.get_score(
    importance_type="gain"
)

importance_df = pd.DataFrame(
    list(importance.items()),
    columns=[
        "feature",
        "importance"
    ]
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

overall_target_rate = (
    (backtest["exit_reason"] == "TARGET").mean()
    * 100
)

overall_avg_return = backtest[
    "net_return_pct"
].mean()

print("\n================================")
print("DIAGNOSTICS COMPLETE")
print("================================")

print("Training rows:", len(training))
print("80%+ model signals:", len(signals))
print("Actual trades:", len(backtest))

print(
    "Overall backtest target rate:",
    f"{overall_target_rate:.2f}%"
)

print(
    "Overall average return:",
    f"{overall_avg_return:.2f}%"
)