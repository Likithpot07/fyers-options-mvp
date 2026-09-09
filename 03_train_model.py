from pathlib import Path

import hashlib
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)


INPUT_PATH = Path("data/processed/training_dataset.parquet")
PREDICTIONS_PATH = Path("data/processed/test_predictions.parquet")
PREDICTIONS_CSV_PATH = Path("data/processed/test_predictions.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TEST_DATE_RATIO = 0.20


FEATURES = [
    # Existing option momentum / candle features
    "return_1m",
    "return_3m",
    "return_5m",
    "range_pct",
    "body_pct",
    "candle_strength",

    # Existing indicators
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

    # Fidelity / chart-structure features
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

    # NIFTY Fidelity / chart-structure context
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

    # Existing pivot / moneyness / CE-PE relation
    "distance_from_pivot_pct",
    "distance_from_strike_pct",
    "premium_pct_of_spot",
    "ce_pe_ratio",
    "ce_return_3m",
    "pe_return_3m",
    "ce_minus_pe_return_3m",

    # Updated setup score
    "setup_score",
]

TARGET = "target_hit"

# --------------------------------------------------
# FEATURE MANIFEST / INTEGRITY CHECKS
# --------------------------------------------------
if len(FEATURES) != len(set(FEATURES)):
    raise ValueError("Duplicate feature names found in FEATURES.")

FEATURES_SHA256 = hashlib.sha256(
    "\n".join(FEATURES).encode("utf-8")
).hexdigest()

print("\nMODEL FEATURE MANIFEST")
print("----------------------------")
print("Feature count:", len(FEATURES))
print("Feature SHA256:", FEATURES_SHA256)
print("----------------------------")


# --------------------------------------------------
# LOAD
# --------------------------------------------------

df = pd.read_parquet(INPUT_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df["date"] = df["timestamp"].dt.date


# --------------------------------------------------
# VALIDATE INPUT DATASET
# --------------------------------------------------

required_columns = FEATURES + [
    "instrument_type",
    TARGET,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Training dataset is missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


dates = sorted(df["date"].unique())

if len(dates) < 2:
    raise ValueError(
        "Need at least 2 trading days for a time-based train/test split."
    )

test_days = max(
    1,
    int(len(dates) * TEST_DATE_RATIO)
)

# Always leave at least one date for training.
if test_days >= len(dates):
    test_days = len(dates) - 1

train_dates = dates[:-test_days]
test_dates = dates[-test_days:]


print(
    "Train:",
    train_dates[0],
    "→",
    train_dates[-1]
)

print(
    "Test:",
    test_dates[0],
    "→",
    test_dates[-1]
)


# --------------------------------------------------
# TRAIN ONE MODEL PER OPTION TYPE
# --------------------------------------------------

all_predictions = []


for option_type in ["CE", "PE"]:

    print("\n================================")
    print(option_type, "MODEL")
    print("================================")

    subset = df[
        df["instrument_type"] == option_type
    ].copy()

    train = subset[
        subset["date"].isin(train_dates)
    ].copy()

    test = subset[
        subset["date"].isin(test_dates)
    ].copy()


    if train.empty:
        raise ValueError(
            f"No training rows available for {option_type}."
        )

    if test.empty:
        raise ValueError(
            f"No test rows available for {option_type}."
        )


    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


    positive = y_train.sum()
    negative = len(y_train) - positive

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1
    )


    print("Train rows:", len(train))
    print("Test rows:", len(test))

    print(
        "Train success:",
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        "Test success:",
        f"{y_test.mean() * 100:.2f}%"
    )

    print(
        "Scale pos weight:",
        round(
            float(scale_pos_weight),
            4
        )
    )


    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )

    booster_feature_names = model.get_booster().feature_names

    if booster_feature_names != FEATURES:
        raise RuntimeError(
            f"{option_type} model feature order does not match FEATURES."
        )

    print(
        "Feature order verified:",
        len(booster_feature_names),
        "features"
    )


    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)


    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )


    if y_test.nunique() > 1:

        print(
            "ROC AUC:",
            round(
                roc_auc_score(
                    y_test,
                    probabilities
                ),
                4
            )
        )


    test["prediction_probability"] = probabilities
    test["prediction_class_50"] = predictions


    # --------------------------------------------------
    # HIGH-CONFIDENCE ANALYSIS
    # --------------------------------------------------

    high = test[
        test["prediction_probability"] >= 0.80
    ]

    print("80%+ signals:", len(high))

    if len(high):

        print(
            "80%+ actual success:",
            f"{high['target_hit'].mean() * 100:.2f}%"
        )


    # Optional Fidelity-filter diagnostic
    if "fidelity_confirmed" in test.columns:

        high_fidelity = test[
            (test["prediction_probability"] >= 0.80)
            & (test["fidelity_confirmed"] == 1)
        ]

        print(
            "80%+ + Fidelity confirmed:",
            len(high_fidelity)
        )

        if len(high_fidelity):

            print(
                "80%+ + Fidelity success:",
                f"{high_fidelity['target_hit'].mean() * 100:.2f}%"
            )


    # --------------------------------------------------
    # SAVE INDIVIDUAL MODEL
    # --------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"xgboost_{option_type.lower()}_10pct.json"
    )

    model.save_model(
        model_path
    )

    print(
        "Saved model:",
        model_path
    )


    all_predictions.append(
        test
    )


# --------------------------------------------------
# COMBINE CE + PE PREDICTIONS
# --------------------------------------------------

combined = pd.concat(
    all_predictions,
    ignore_index=True
)

combined = combined.sort_values(
    "timestamp"
)


# Save parquet
combined.to_parquet(
    PREDICTIONS_PATH,
    index=False
)


# Save CSV too
combined.to_csv(
    PREDICTIONS_CSV_PATH,
    index=False
)


# Save exact model feature order
features_path = MODEL_DIR / "features.json"

with open(
    features_path,
    "w"
) as f:

    json.dump(
        FEATURES,
        f,
        indent=4
    )

with open(
    features_path,
    "r"
) as f:

    saved_features = json.load(f)

if saved_features != FEATURES:
    raise RuntimeError(
        "models/features.json does not match the features used for training."
    )

saved_hash = hashlib.sha256(
    "\n".join(saved_features).encode("utf-8")
).hexdigest()

if saved_hash != FEATURES_SHA256:
    raise RuntimeError("features.json hash verification failed.")

print(
    "\nfeatures.json verified:",
    len(saved_features),
    "features | SHA256:",
    saved_hash
)


# --------------------------------------------------
# COMBINED SUMMARY
# --------------------------------------------------

print("\n================================")
print("COMBINED")
print("================================")

high = combined[
    combined["prediction_probability"] >= 0.80
]

print(
    "80%+ signals:",
    len(high)
)

if len(high):

    print(
        "Actual success:",
        f"{high['target_hit'].mean() * 100:.2f}%"
    )


if "fidelity_confirmed" in combined.columns:

    high_fidelity = combined[
        (combined["prediction_probability"] >= 0.80)
        & (combined["fidelity_confirmed"] == 1)
    ]

    print(
        "80%+ + Fidelity confirmed:",
        len(high_fidelity)
    )

    if len(high_fidelity):

        print(
            "80%+ + Fidelity actual success:",
            f"{high_fidelity['target_hit'].mean() * 100:.2f}%"
        )


print("\nSaved:")
print(PREDICTIONS_PATH)
print(PREDICTIONS_CSV_PATH)
print(MODEL_DIR / "features.json")
