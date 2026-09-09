from pathlib import Path

import hashlib
import json
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

MODEL_DIR = Path("models")

CE_MODEL_PATH = MODEL_DIR / "xgboost_ce_10pct.json"
PE_MODEL_PATH = MODEL_DIR / "xgboost_pe_10pct.json"
FEATURES_PATH = MODEL_DIR / "features.json"

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

if not backtest.empty:
    backtest["signal_time"] = pd.to_datetime(
        backtest["signal_time"],
        utc=True
    )

training["date"] = training["timestamp"].dt.date
predictions["date"] = predictions["timestamp"].dt.date

if not backtest.empty:
    backtest["date"] = backtest["signal_time"].dt.date


# --------------------------------------------------
# FEATURE MANIFEST VERIFICATION
# --------------------------------------------------

print("\n================================")
print("0. FEATURE MANIFEST VERIFICATION")
print("================================")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Missing feature manifest: {FEATURES_PATH}"
    )

with open(FEATURES_PATH, "r") as f:
    saved_features = json.load(f)

if not isinstance(saved_features, list):
    raise ValueError(
        "models/features.json must contain a JSON list."
    )

if len(saved_features) != len(set(saved_features)):
    raise ValueError(
        "Duplicate feature names found in models/features.json."
    )

missing_training_features = [
    feature
    for feature in saved_features
    if feature not in training.columns
]

missing_prediction_features = [
    feature
    for feature in saved_features
    if feature not in predictions.columns
]

if missing_training_features:
    raise ValueError(
        "Training dataset is missing features from models/features.json:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_training_features
        )
    )

if missing_prediction_features:
    raise ValueError(
        "Prediction dataset is missing features from models/features.json:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_prediction_features
        )
    )

feature_hash = hashlib.sha256(
    "\n".join(saved_features).encode("utf-8")
).hexdigest()

print("Feature count:", len(saved_features))
print("Feature SHA256:", feature_hash)
print("Training dataset feature check: PASS")
print("Prediction dataset feature check: PASS")


# --------------------------------------------------
# LOAD BOTH MODELS + VERIFY FEATURE ORDER
# --------------------------------------------------

models = {}

for option_type, model_path in {
    "CE": CE_MODEL_PATH,
    "PE": PE_MODEL_PATH,
}.items():

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {option_type} model: {model_path}"
        )

    model = XGBClassifier()
    model.load_model(model_path)

    booster_features = model.get_booster().feature_names

    if booster_features != saved_features:
        raise RuntimeError(
            f"{option_type} model feature order does not match "
            "models/features.json."
        )

    print(
        f"{option_type} model feature check: PASS "
        f"({len(booster_features)} features)"
    )

    models[option_type] = model


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
# 2. MODEL SIGNALS >= THRESHOLD
# --------------------------------------------------

print("\n================================")
print(
    f"2. MODEL SIGNALS >= "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)
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
# 3. SIGNAL FUNNEL BY CE / PE
# --------------------------------------------------

print("\n================================")
print("3. SIGNAL FUNNEL BY CE / PE")
print("================================")

def side_count(frame, option_type):
    return int(
        (
            frame["instrument_type"]
            == option_type
        ).sum()
    )

stage_conf = predictions[
    predictions["prediction_probability"]
    >= CONFIDENCE_THRESHOLD
].copy()

stage_setup = stage_conf[
    stage_conf["candidate_setup"] == 1
].copy()

stage_fidelity = stage_setup[
    stage_setup["fidelity_confirmed"] == 1
].copy()

stage_best = (
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

for option_type in ["CE", "PE"]:

    final_trades = (
        side_count(backtest, option_type)
        if not backtest.empty
        else 0
    )

    print(f"\n{option_type}")
    print(
        f"{CONFIDENCE_THRESHOLD:.0%}+ model signals:",
        side_count(stage_conf, option_type)
    )
    print(
        "After setup filter:",
        side_count(stage_setup, option_type)
    )
    print(
        "After Fidelity filter:",
        side_count(stage_fidelity, option_type)
    )
    print(
        "After best-candidate filter:",
        side_count(stage_best, option_type)
    )
    print(
        "Final backtest trades:",
        final_trades
    )


# --------------------------------------------------
# 4. REAL BACKTEST BY CE / PE
# --------------------------------------------------

print("\n================================")
print("4. REAL BACKTEST")
print("================================")

if backtest.empty:

    print("No backtest trades available.")

else:

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
            timeouts=(
                "exit_reason",
                lambda x: (x == "TIMEOUT").sum()
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
# 5. CONFIDENCE CALIBRATION — CE / PE SEPARATE
# --------------------------------------------------

print("\n================================")
print("5. CONFIDENCE CALIBRATION BY SIDE")
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
        [
            "instrument_type",
            "confidence_band"
        ],
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
# 6. BACKTEST BY CONFIDENCE + SIDE
# --------------------------------------------------

print("\n================================")
print("6. BACKTEST BY CONFIDENCE + SIDE")
print("================================")

if backtest.empty:

    print("No backtest trades available.")

else:

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
            [
                "instrument_type",
                "confidence_band"
            ],
            observed=True
        )
        .agg(
            trades=("symbol", "count"),
            targets=(
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

    bt_confidence["win_rate"] = (
        bt_confidence["targets"]
        / bt_confidence["trades"]
        * 100
    )

    print(bt_confidence)


# --------------------------------------------------
# 7. SIGNAL OVERLAP BY SIDE
# --------------------------------------------------

print("\n================================")
print("7. SIGNAL OVERLAP BY SIDE")
print("================================")

for option_type in ["CE", "PE"]:

    side_signals = signals[
        signals["instrument_type"] == option_type
    ].sort_values(
        ["symbol", "timestamp"]
    )

    overlap_counts = []

    for _, group in side_signals.groupby("symbol"):

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

    print(f"\n{option_type}")

    if overlap_counts:

        overlap_series = pd.Series(
            overlap_counts
        )

        print(
            "Average signals inside 30m window:",
            round(
                overlap_series.mean(),
                2
            )
        )

        print(
            "Median:",
            round(
                overlap_series.median(),
                2
            )
        )

        print(
            "Maximum:",
            overlap_series.max()
        )

    else:

        print("No high-confidence signals.")


# --------------------------------------------------
# 8. SIGNALS PER DAY + SIDE
# --------------------------------------------------

print("\n================================")
print("8. SIGNALS PER DAY + SIDE")
print("================================")

daily_signals = (
    signals
    .groupby(
        [
            "date",
            "instrument_type"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(daily_signals)


# --------------------------------------------------
# 9. REAL TRADES PER DAY
# --------------------------------------------------

print("\n================================")
print("9. REAL TRADES PER DAY")
print("================================")

if backtest.empty:

    print("No backtest trades available.")

else:

    daily_trades = (
        backtest
        .groupby(
            [
                "date",
                "instrument_type"
            ]
        )
        .agg(
            trades=("symbol", "count"),
            targets=(
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

    daily_trades["win_rate"] = (
        daily_trades["targets"]
        / daily_trades["trades"]
        * 100
    )

    print(daily_trades)


# --------------------------------------------------
# 10. PERFORMANCE BY STRIKE
# --------------------------------------------------

print("\n================================")
print("10. PERFORMANCE BY STRIKE")
print("================================")

if backtest.empty:

    print("No backtest trades available.")

else:

    strike_summary = (
        backtest
        .groupby(
            [
                "instrument_type",
                "strike"
            ]
        )
        .agg(
            trades=("symbol", "count"),
            targets=(
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
# 11. FEATURE IMPORTANCE — CE / PE SEPARATE
# --------------------------------------------------

print("\n================================")
print("11. XGBOOST FEATURE IMPORTANCE")
print("================================")

for option_type in ["CE", "PE"]:

    print(f"\n{option_type} TOP 20 FEATURES")

    booster = (
        models[option_type]
        .get_booster()
    )

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

    if importance_df.empty:

        print("No feature importance available.")
        continue

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    unknown_features = [
        feature
        for feature in importance_df["feature"]
        if feature not in saved_features
    ]

    if unknown_features:
        raise RuntimeError(
            f"{option_type} model reports features not present "
            "in models/features.json: "
            + ", ".join(unknown_features)
        )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n================================")
print("DIAGNOSTICS COMPLETE")
print("================================")

print("Training rows:", len(training))
print(
    f"{CONFIDENCE_THRESHOLD:.0%}+ model signals:",
    len(signals)
)

print(
    "Actual trades:",
    len(backtest)
)

if backtest.empty:

    print("Overall backtest target rate: N/A")
    print("Overall average return: N/A")

else:

    overall_target_rate = (
        (
            backtest["exit_reason"]
            == "TARGET"
        ).mean()
        * 100
    )

    overall_avg_return = (
        backtest[
            "net_return_pct"
        ].mean()
    )

    print(
        "Overall backtest target rate:",
        f"{overall_target_rate:.2f}%"
    )

    print(
        "Overall average return:",
        f"{overall_avg_return:.2f}%"
    )
