from pathlib import Path
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

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TEST_DATE_RATIO = 0.20


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

    "setup_score",
]

TARGET = "target_hit"


# --------------------------------------------------
# LOAD
# --------------------------------------------------

df = pd.read_parquet(INPUT_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df["date"] = df["timestamp"].dt.date

dates = sorted(df["date"].unique())

test_days = max(
    1,
    int(len(dates) * TEST_DATE_RATIO)
)

train_dates = dates[:-test_days]
test_dates = dates[-test_days:]


print("Train:", train_dates[0], "→", train_dates[-1])
print("Test:", test_dates[0], "→", test_dates[-1])


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


    high = test[
        test["prediction_probability"] >= 0.80
    ]

    print("80%+ signals:", len(high))

    if len(high):

        print(
            "80%+ actual success:",
            f"{high['target_hit'].mean() * 100:.2f}%"
        )


    # Save individual model
    model.save_model(
        MODEL_DIR /
        f"xgboost_{option_type.lower()}_10pct.json"
    )

    all_predictions.append(test)


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

combined.to_parquet(
    PREDICTIONS_PATH,
    index=False
)


with open(
    MODEL_DIR / "features.json",
    "w"
) as f:

    json.dump(
        FEATURES,
        f,
        indent=4
    )


print("\n================================")
print("COMBINED")
print("================================")

high = combined[
    combined["prediction_probability"] >= 0.80
]

print("80%+ signals:", len(high))

if len(high):

    print(
        "Actual success:",
        f"{high['target_hit'].mean() * 100:.2f}%"
    )

print("\nSaved:")
print(PREDICTIONS_PATH)