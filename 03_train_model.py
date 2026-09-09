from pathlib import Path

import json
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/training_dataset.parquet"
)

PREDICTIONS_PATH = Path(
    "data/processed/test_predictions.parquet"
)

PREDICTIONS_CSV_PATH = Path(
    "data/processed/test_predictions.csv"
)

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# CONTROLLED EXPERIMENT CONFIG
# --------------------------------------------------

TEST_DATE_RATIO = 0.20

STRATEGIES = {
    "10PCT": {
        "target_column": "target_hit_10pct",
        "outcome_column": "trade_outcome_10pct",
        "model_suffix": "10pct",
        "target_pct": 0.10,
        "stop_pct": 0.05,
    },
    "5PCT": {
        "target_column": "target_hit_5pct",
        "outcome_column": "trade_outcome_5pct",
        "model_suffix": "5pct",
        "target_pct": 0.05,
        "stop_pct": 0.05,
    },
}


# --------------------------------------------------
# MODEL FEATURES
#
# IMPORTANT:
# These are identical for 10PCT and 5PCT.
# Only the target label changes.
# --------------------------------------------------

FEATURES = [

    # Option momentum / candle features
    "return_1m",
    "return_3m",
    "return_5m",
    "range_pct",
    "body_pct",
    "candle_strength",

    # Indicators
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

    # NIFTY fidelity / structure
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

    # Pivot / moneyness / CE-PE relationship
    "distance_from_pivot_pct",
    "distance_from_strike_pct",
    "premium_pct_of_spot",

    "ce_pe_ratio",
    "ce_return_3m",
    "pe_return_3m",
    "ce_minus_pe_return_3m",

    # Setup score
    "setup_score",
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_parquet(
    INPUT_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df["date"] = (
    df["timestamp"]
    .dt.date
)


# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------

required_columns = (
    FEATURES
    + [
        "timestamp",
        "date",
        "symbol",
        "instrument_type",
        "strike",
        "fidelity_confirmed",
        "target_hit_10pct",
        "trade_outcome_10pct",
        "target_hit_5pct",
        "trade_outcome_5pct",
    ]
)

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
            for column
            in missing_columns
        )
    )


# --------------------------------------------------
# SAME CHRONOLOGICAL SPLIT FOR ALL FOUR MODELS
# --------------------------------------------------

dates = sorted(
    df["date"].unique()
)

if len(dates) < 2:
    raise ValueError(
        "Need at least 2 trading days "
        "for train/test split."
    )

test_days = max(
    1,
    int(
        len(dates)
        * TEST_DATE_RATIO
    ),
)

if test_days >= len(dates):
    test_days = (
        len(dates)
        - 1
    )

train_dates = (
    dates[
        :-test_days
    ]
)

test_dates = (
    dates[
        -test_days:
    ]
)

print(
    "=" * 72
)

print(
    "CONTROLLED 10PCT vs 5PCT TRAINING"
)

print(
    "=" * 72
)

print(
    "Total rows:",
    f"{len(df):,}",
)

print(
    "Trading days:",
    len(dates),
)

print(
    "Train:",
    train_dates[0],
    "->",
    train_dates[-1],
)

print(
    "Test :",
    test_dates[0],
    "->",
    test_dates[-1],
)

print(
    "Train days:",
    len(train_dates),
)

print(
    "Test days :",
    len(test_dates),
)

print()
print(
    "10PCT = +10% target / -5% stop"
)

print(
    "5PCT  =  +5% target / -5% stop"
)


# --------------------------------------------------
# THRESHOLD ANALYSIS
# --------------------------------------------------

def print_threshold_table(
    test,
    probabilities,
    y_test,
    title,
):
    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    print()
    print(
        "-" * 72
    )

    print(
        f"{title} THRESHOLD ANALYSIS"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Threshold':>10} "
        f"{'Signals':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Success':>12}"
    )

    print(
        "-" * 60
    )

    for threshold in thresholds:

        selected = (
            probabilities
            >= threshold
        )

        count = int(
            selected.sum()
        )

        if count == 0:

            print(
                f"{threshold:>10.2f} "
                f"{0:>10} "
                f"{'N/A':>12} "
                f"{'N/A':>10} "
                f"{'N/A':>12}"
            )

            continue

        selected_y = (
            y_test[
                selected
            ]
        )

        precision = precision_score(
            y_test,
            selected.astype(int),
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            selected.astype(int),
            zero_division=0,
        )

        success = (
            selected_y.mean()
        )

        print(
            f"{threshold:>10.2f} "
            f"{count:>10} "
            f"{precision * 100:>11.2f}% "
            f"{recall * 100:>9.2f}% "
            f"{success * 100:>11.2f}%"
        )

    # --------------------------------------------------
    # THRESHOLD + FIDELITY
    # --------------------------------------------------

    print()
    print(
        "-" * 72
    )

    print(
        f"{title} THRESHOLD + FIDELITY"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Threshold':>10} "
        f"{'Signals':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Success':>12}"
    )

    print(
        "-" * 60
    )

    fidelity = (
        test[
            "fidelity_confirmed"
        ]
        .fillna(0)
        .astype(int)
        .to_numpy()
        == 1
    )

    for threshold in thresholds:

        selected = (
            (
                probabilities
                >= threshold
            )
            & fidelity
        )

        count = int(
            selected.sum()
        )

        if count == 0:

            print(
                f"{threshold:>10.2f} "
                f"{0:>10} "
                f"{'N/A':>12} "
                f"{'N/A':>10} "
                f"{'N/A':>12}"
            )

            continue

        selected_y = (
            y_test[
                selected
            ]
        )

        precision = (
            selected_y.mean()
        )

        recall = (
            selected_y.sum()
            / y_test.sum()
            if y_test.sum() > 0
            else 0
        )

        success = (
            selected_y.mean()
        )

        print(
            f"{threshold:>10.2f} "
            f"{count:>10} "
            f"{precision * 100:>11.2f}% "
            f"{recall * 100:>9.2f}% "
            f"{success * 100:>11.2f}%"
        )


# --------------------------------------------------
# MODEL FACTORY
#
# Same hyperparameters for all four models.
# Class weighting is calculated separately because
# the positive rate naturally differs by target.
# --------------------------------------------------

def build_model(
    scale_pos_weight
):
    return XGBClassifier(
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
        n_jobs=-1,
    )


# --------------------------------------------------
# PREPARE ONE SHARED HELD-OUT TEST FRAME
#
# IMPORTANT:
# Keep the ORIGINAL df index while predictions are being
# attached. Each model's `test` dataframe preserves those
# same original indices, so probabilities can be written
# back to the exact historical rows they came from.
#
# Do NOT sort/reset_index here. We sort only AFTER all
# four models have written their probabilities.
# --------------------------------------------------

test_output = df[
    df["date"].isin(
        test_dates
    )
].copy()

test_output[
    "prediction_probability_10pct"
] = pd.NA

test_output[
    "prediction_class_50_10pct"
] = pd.NA

test_output[
    "prediction_probability_5pct"
] = pd.NA

test_output[
    "prediction_class_50_5pct"
] = pd.NA


# --------------------------------------------------
# TRAIN ALL FOUR MODELS
# --------------------------------------------------

training_summaries = []

for (
    strategy_name,
    strategy_config,
) in STRATEGIES.items():

    target_column = (
        strategy_config[
            "target_column"
        ]
    )

    outcome_column = (
        strategy_config[
            "outcome_column"
        ]
    )

    model_suffix = (
        strategy_config[
            "model_suffix"
        ]
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"{strategy_name} MODELS"
    )

    print(
        "=" * 72
    )

    print(
        "Ground truth:",
        target_column,
    )

    print(
        "Outcome:",
        outcome_column,
    )

    print(
        "Target / Stop:",
        f"+{strategy_config['target_pct'] * 100:.0f}% / "
        f"-{strategy_config['stop_pct'] * 100:.0f}%",
    )

    for option_type in [
        "CE",
        "PE",
    ]:

        print()
        print(
            "=" * 36
        )

        print(
            strategy_name,
            option_type,
            "MODEL",
        )

        print(
            "=" * 36
        )

        subset = df[
            df[
                "instrument_type"
            ]
            == option_type
        ].copy()

        train = subset[
            subset[
                "date"
            ].isin(
                train_dates
            )
        ].copy()

        test = subset[
            subset[
                "date"
            ].isin(
                test_dates
            )
        ].copy()

        if train.empty:
            raise ValueError(
                f"No training rows available "
                f"for {strategy_name} {option_type}."
            )

        if test.empty:
            raise ValueError(
                f"No test rows available "
                f"for {strategy_name} {option_type}."
            )

        X_train = (
            train[
                FEATURES
            ]
        )

        y_train = (
            train[
                target_column
            ]
            .astype(int)
        )

        X_test = (
            test[
                FEATURES
            ]
        )

        y_test = (
            test[
                target_column
            ]
            .astype(int)
        )

        positive = int(
            y_train.sum()
        )

        negative = (
            len(y_train)
            - positive
        )

        scale_pos_weight = (
            negative
            / positive
            if positive > 0
            else 1.0
        )

        print(
            "Train rows:",
            f"{len(train):,}",
        )

        print(
            "Test rows :",
            f"{len(test):,}",
        )

        print(
            "Train success:",
            f"{y_train.mean() * 100:.2f}%",
        )

        print(
            "Test success :",
            f"{y_test.mean() * 100:.2f}%",
        )

        print(
            "Scale pos weight:",
            round(
                float(
                    scale_pos_weight
                ),
                4,
            ),
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        model = build_model(
            scale_pos_weight
        )

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------
        # PREDICT
        # --------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions_50 = (
            probabilities
            >= 0.50
        ).astype(int)

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        precision_50 = precision_score(
            y_test,
            predictions_50,
            zero_division=0,
        )

        recall_50 = recall_score(
            y_test,
            predictions_50,
            zero_division=0,
        )

        print(
            "Precision @ 0.50:",
            round(
                precision_50,
                4,
            ),
        )

        print(
            "Recall @ 0.50:",
            round(
                recall_50,
                4,
            ),
        )

        auc = None

        if (
            y_test.nunique()
            > 1
        ):
            auc = roc_auc_score(
                y_test,
                probabilities,
            )

            print(
                "ROC AUC:",
                round(
                    auc,
                    4,
                ),
            )

        # --------------------------------------------------
        # THRESHOLD ANALYSIS
        # --------------------------------------------------

        print_threshold_table(
            test=test,
            probabilities=probabilities,
            y_test=y_test,
            title=(
                f"{strategy_name} {option_type}"
            ),
        )

        # --------------------------------------------------
        # HIGH-CONFIDENCE SUMMARY
        # --------------------------------------------------

        for threshold in [
            0.80,
            0.85,
        ]:

            high_mask = (
                probabilities
                >= threshold
            )

            high_count = int(
                high_mask.sum()
            )

            print()
            print(
                f"{int(threshold * 100)}%+ signals:",
                high_count,
            )

            if high_count:

                high_success = (
                    y_test[
                        high_mask
                    ].mean()
                )

                print(
                    f"{int(threshold * 100)}%+ actual success:",
                    f"{high_success * 100:.2f}%",
                )

            fidelity_mask = (
                test[
                    "fidelity_confirmed"
                ]
                .fillna(0)
                .astype(int)
                .to_numpy()
                == 1
            )

            high_fidelity_mask = (
                high_mask
                & fidelity_mask
            )

            high_fidelity_count = int(
                high_fidelity_mask.sum()
            )

            print(
                f"{int(threshold * 100)}%+ + Fidelity confirmed:",
                high_fidelity_count,
            )

            if high_fidelity_count:

                high_fidelity_success = (
                    y_test[
                        high_fidelity_mask
                    ].mean()
                )

                print(
                    f"{int(threshold * 100)}%+ + Fidelity success:",
                    f"{high_fidelity_success * 100:.2f}%",
                )

        # --------------------------------------------------
        # SAVE MODEL
        # --------------------------------------------------

        model_path = (
            MODEL_DIR
            / (
                f"xgboost_"
                f"{option_type.lower()}_"
                f"{model_suffix}.json"
            )
        )

        model.save_model(
            model_path
        )

        print()
        print(
            "Saved model:",
            model_path,
        )

        # --------------------------------------------------
        # ADD THIS MODEL'S PROBABILITIES TO THE
        # EXACT SAME HISTORICAL ROWS
        #
        # `test.index` is inherited from the original df.
        # `probabilities` is produced in the same row order
        # as X_test/test, so this is exact row-level alignment.
        # --------------------------------------------------

        test_indices = test.index

        if len(test_indices) != len(probabilities):
            raise ValueError(
                f"Prediction alignment error for "
                f"{strategy_name} {option_type}: "
                f"{len(test_indices)} rows but "
                f"{len(probabilities)} probabilities."
            )

        if strategy_name == "10PCT":

            test_output.loc[
                test_indices,
                "prediction_probability_10pct",
            ] = probabilities

            test_output.loc[
                test_indices,
                "prediction_class_50_10pct",
            ] = predictions_50

        elif strategy_name == "5PCT":

            test_output.loc[
                test_indices,
                "prediction_probability_5pct",
            ] = probabilities

            test_output.loc[
                test_indices,
                "prediction_class_50_5pct",
            ] = predictions_50

        training_summaries.append(
            {
                "strategy": strategy_name,
                "option_type": option_type,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_success_rate": (
                    y_train.mean()
                ),
                "test_success_rate": (
                    y_test.mean()
                ),
                "scale_pos_weight": (
                    scale_pos_weight
                ),
                "precision_50": (
                    precision_50
                ),
                "recall_50": (
                    recall_50
                ),
                "roc_auc": auc,
                "model_path": str(
                    model_path
                ),
            }
        )


# --------------------------------------------------
# VALIDATE CONTROLLED OUTPUT
# --------------------------------------------------

probability_columns = [
    "prediction_probability_10pct",
    "prediction_probability_5pct",
]

missing_probability_rows = (
    test_output[
        probability_columns
    ]
    .isna()
    .any(axis=1)
)

if (
    missing_probability_rows
    .any()
):
    count = int(
        missing_probability_rows.sum()
    )

    raise ValueError(
        f"{count} test rows are missing "
        "10PCT or 5PCT probabilities."
    )

# --------------------------------------------------
# ALIGNMENT SANITY CHECK
#
# Recalculate high-confidence success directly from the
# aligned shared output and compare it against the same
# rows/targets later in the report. This also makes any
# future row-order bug immediately visible.
# --------------------------------------------------

for option_type in [
    "CE",
    "PE",
]:
    side = test_output[
        test_output[
            "instrument_type"
        ]
        == option_type
    ]

    expected_rows = len(
        df[
            (
                df["date"].isin(
                    test_dates
                )
            )
            & (
                df["instrument_type"]
                == option_type
            )
        ]
    )

    if len(side) != expected_rows:
        raise ValueError(
            f"Alignment sanity check failed for {option_type}: "
            f"expected {expected_rows} rows, found {len(side)}."
        )

print()
print(
    "Prediction row alignment: PASSED"
)


# --------------------------------------------------
# CONVERT PREDICTION COLUMNS TO NUMERIC
# --------------------------------------------------

test_output[
    "prediction_probability_10pct"
] = pd.to_numeric(
    test_output[
        "prediction_probability_10pct"
    ]
)

test_output[
    "prediction_probability_5pct"
] = pd.to_numeric(
    test_output[
        "prediction_probability_5pct"
    ]
)

test_output[
    "prediction_class_50_10pct"
] = pd.to_numeric(
    test_output[
        "prediction_class_50_10pct"
    ]
).astype(int)

test_output[
    "prediction_class_50_5pct"
] = pd.to_numeric(
    test_output[
        "prediction_class_50_5pct"
    ]
).astype(int)


# --------------------------------------------------
# SORT ONLY AFTER ROW-LEVEL ALIGNMENT IS COMPLETE
# --------------------------------------------------

test_output = (
    test_output
    .sort_values(
        [
            "timestamp",
            "instrument_type",
            "symbol",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# BACKWARD-COMPATIBILITY ALIASES
#
# Older 04-08/09-style scripts may still expect:
#
#   prediction_probability
#   prediction_class_50
#
# Keep these mapped to 5PCT for now.
# New comparison code should use explicit columns.
# --------------------------------------------------

test_output[
    "prediction_probability"
] = test_output[
    "prediction_probability_5pct"
]

test_output[
    "prediction_class_50"
] = test_output[
    "prediction_class_50_5pct"
]


# --------------------------------------------------
# SAVE HELD-OUT TEST PREDICTIONS
# --------------------------------------------------

test_output.to_parquet(
    PREDICTIONS_PATH,
    index=False,
)

test_output.to_csv(
    PREDICTIONS_CSV_PATH,
    index=False,
)

print()
print(
    "Saved held-out predictions:"
)

print(
    PREDICTIONS_PATH
)

print(
    PREDICTIONS_CSV_PATH
)


# --------------------------------------------------
# SAVE FEATURE ORDER
# --------------------------------------------------

with (
    MODEL_DIR
    / "features.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        FEATURES,
        f,
        indent=4,
    )

print()
print(
    "Saved feature order:"
)

print(
    MODEL_DIR
    / "features.json"
)


# --------------------------------------------------
# MODEL TRAINING SUMMARY TABLE
# --------------------------------------------------

summary_df = pd.DataFrame(
    training_summaries
)

print()
print(
    "=" * 92
)

print(
    "FOUR-MODEL TRAINING SUMMARY"
)

print(
    "=" * 92
)

display_summary = (
    summary_df[
        [
            "strategy",
            "option_type",
            "train_rows",
            "test_rows",
            "train_success_rate",
            "test_success_rate",
            "precision_50",
            "recall_50",
            "roc_auc",
        ]
    ]
    .copy()
)

for column in [
    "train_success_rate",
    "test_success_rate",
    "precision_50",
    "recall_50",
    "roc_auc",
]:
    display_summary[
        column
    ] = (
        display_summary[
            column
        ]
        .apply(
            lambda value: (
                f"{value * 100:.2f}%"
                if pd.notna(value)
                else "N/A"
            )
        )
    )

print(
    display_summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# DIRECT HELD-OUT COMPARISON
# --------------------------------------------------

print()
print(
    "=" * 92
)

print(
    "HELD-OUT TEST PERIOD: 10PCT vs 5PCT"
)

print(
    "=" * 92
)

for option_type in [
    "CE",
    "PE",
]:

    side = test_output[
        test_output[
            "instrument_type"
        ]
        == option_type
    ]

    print()
    print(
        option_type
    )

    print(
        "-" * 64
    )

    print(
        f"{'Metric':<24}"
        f"{'10PCT':>20}"
        f"{'5PCT':>20}"
    )

    print(
        "-" * 64
    )

    natural_10 = (
        side[
            "target_hit_10pct"
        ].mean()
    )

    natural_5 = (
        side[
            "target_hit_5pct"
        ].mean()
    )

    print(
        f"{'Natural success':<24}"
        f"{natural_10 * 100:>19.2f}%"
        f"{natural_5 * 100:>19.2f}%"
    )

    for threshold in [
        0.80,
        0.85,
    ]:

        mask_10 = (
            side[
                "prediction_probability_10pct"
            ]
            >= threshold
        )

        mask_5 = (
            side[
                "prediction_probability_5pct"
            ]
            >= threshold
        )

        count_10 = int(
            mask_10.sum()
        )

        count_5 = int(
            mask_5.sum()
        )

        success_10 = (
            side.loc[
                mask_10,
                "target_hit_10pct",
            ].mean()
            if count_10
            else float("nan")
        )

        success_5 = (
            side.loc[
                mask_5,
                "target_hit_5pct",
            ].mean()
            if count_5
            else float("nan")
        )

        print(
            f"{int(threshold * 100)}%+ signals"
            f"{'':<13}"
            f"{count_10:>20}"
            f"{count_5:>20}"
        )

        value_10 = (
            f"{success_10 * 100:.2f}%"
            if pd.notna(
                success_10
            )
            else "N/A"
        )

        value_5 = (
            f"{success_5 * 100:.2f}%"
            if pd.notna(
                success_5
            )
            else "N/A"
        )

        print(
            f"{int(threshold * 100)}%+ success"
            f"{'':<13}"
            f"{value_10:>20}"
            f"{value_5:>20}"
        )


# --------------------------------------------------
# SIGNAL AGREEMENT
# --------------------------------------------------

print()
print(
    "=" * 92
)

print(
    "MODEL AGREEMENT ON HELD-OUT ROWS"
)

print(
    "=" * 92
)

for threshold in [
    0.80,
    0.85,
]:

    signal_10 = (
        test_output[
            "prediction_probability_10pct"
        ]
        >= threshold
    )

    signal_5 = (
        test_output[
            "prediction_probability_5pct"
        ]
        >= threshold
    )

    both = (
        signal_10
        & signal_5
    )

    only_10 = (
        signal_10
        & ~signal_5
    )

    only_5 = (
        signal_5
        & ~signal_10
    )

    neither = (
        ~signal_10
        & ~signal_5
    )

    print()
    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        "Both models signal:",
        int(
            both.sum()
        ),
    )

    print(
        "Only 10PCT signals:",
        int(
            only_10.sum()
        ),
    )

    print(
        "Only 5PCT signals :",
        int(
            only_5.sum()
        ),
    )

    print(
        "Neither signals   :",
        int(
            neither.sum()
        ),
    )


# --------------------------------------------------
# FINAL
# --------------------------------------------------

print()
print(
    "=" * 72
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 72
)

print(
    "Generated models:"
)

print(
    "  models/xgboost_ce_10pct.json"
)

print(
    "  models/xgboost_pe_10pct.json"
)

print(
    "  models/xgboost_ce_5pct.json"
)

print(
    "  models/xgboost_pe_5pct.json"
)

print()
print(
    "Same train/test dates were used "
    "for all four models."
)
