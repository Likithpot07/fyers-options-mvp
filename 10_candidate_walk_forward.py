import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/processed/training_dataset.parquet"

RANDOM_STATE = 42

# Minimum history before testing
MIN_TRAIN_DAYS = 30

# Refit every trading day
TOP_PCTS = [10, 5, 2, 1]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("10 CANDIDATE-ONLY WALK-FORWARD TEST")
print("=" * 90)

df = pd.read_parquet(DATA_PATH)

df = df[df["candidate_setup"] == 1].copy()

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df["date"] = df["timestamp"].dt.date

df["target"] = (df["trade_outcome"] == "TARGET").astype(int)

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Candidate rows: {len(df):,}")
print(f"Trading days: {df['date'].nunique():,}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

EXCLUDE_COLUMNS = {
    "target",
    "trade_outcome",
    "target_hit",
    "entry_price",
    "timestamp",
    "date",
    "symbol",
    "instrument_type",
    "expiry",
    "strike",
    "candidate_setup",

    # Deliberately excluded from this experiment.
    "fidelity_confirmed",
}


feature_columns = []

for col in df.columns:

    if col in EXCLUDE_COLUMNS:
        continue

    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    lower = col.lower()

    # Avoid direct outcome/future leakage.
    bad_parts = [
        "outcome",
        "target",
        "future",
        "profit",
        "return_5",
    ]

    if any(x in lower for x in bad_parts):
        continue

    feature_columns.append(col)


print(f"Feature count: {len(feature_columns)}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def make_model(y):

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    scale_pos_weight = (
        negatives / positives
        if positives > 0
        else 1.0
    )

    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


# ------------------------------------------------------------
# TRADING-DAY LIST
# ------------------------------------------------------------

dates = sorted(df["date"].unique())

print(f"Minimum training days: {MIN_TRAIN_DAYS}")
print()


# ------------------------------------------------------------
# WALK FORWARD
# ------------------------------------------------------------

all_results = []

for i in range(MIN_TRAIN_DAYS, len(dates)):

    test_date = dates[i]

    train_dates = dates[:i]

    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"] == test_date].copy()

    if len(test_df) == 0:
        continue

    # --------------------------------------------------------
    # TRAIN CE
    # --------------------------------------------------------

    side_predictions = []

    for side in ["CE", "PE"]:

        train_side = train_df[
            train_df["instrument_type"] == side
        ].copy()

        test_side = test_df[
            test_df["instrument_type"] == side
        ].copy()

        if len(train_side) == 0 or len(test_side) == 0:
            continue

        X_train = train_side[feature_columns]
        y_train = train_side["target"]

        X_test = test_side[feature_columns]
        y_test = test_side["target"]

        model = make_model(y_train)

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        scores = model.predict_proba(X_test)[:, 1]

        temp = test_side[
            [
                "timestamp",
                "symbol",
                "instrument_type",
                "strike",
                "expiry",
                "entry_price",
                "trade_outcome",
                "target",
                "setup_score",
            ]
        ].copy()

        temp["model_score"] = scores

        side_predictions.append(temp)

    if not side_predictions:
        continue

    day = pd.concat(
        side_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # RANK ALL CE + PE CANDIDATES FOR THE DAY
    # --------------------------------------------------------

    day = day.sort_values(
        "model_score",
        ascending=False
    ).reset_index(drop=True)

    day_count = len(day)

    day_result = {
        "date": test_date,
        "candidates": day_count,
        "baseline_success": day["target"].mean(),
    }

    # --------------------------------------------------------
    # TOP PERCENTILE RESULTS
    # --------------------------------------------------------

    for pct in TOP_PCTS:

        n = max(
            1,
            int(np.ceil(day_count * pct / 100))
        )

        selected = day.head(n)

        success = selected["target"].mean()

        key = f"top_{pct}_success"

        day_result[key] = success

        day_result[f"top_{pct}_count"] = len(selected)

    # --------------------------------------------------------
    # TOP 1 SINGLE TRADE
    # --------------------------------------------------------

    best = day.iloc[0]

    day_result["best_score"] = best["model_score"]
    day_result["best_target"] = best["target"]
    day_result["best_side"] = best["instrument_type"]
    day_result["best_symbol"] = best["symbol"]

    all_results.append(day_result)

    print(
        f"{test_date} | "
        f"Candidates: {day_count:4d} | "
        f"Baseline: {day_result['baseline_success']:.1%} | "
        f"Top10: {day_result['top_10_success']:.1%} | "
        f"Top5: {day_result['top_5_success']:.1%} | "
        f"Top2: {day_result['top_2_success']:.1%} | "
        f"Top1: {day_result['top_1_success']:.1%} | "
        f"Best: {best['instrument_type']} "
        f"{best['model_score']:.3f} "
        f"{'TARGET' if best['target'] else best['trade_outcome']}"
    )


# ------------------------------------------------------------
# RESULTS DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame(all_results)

if results.empty:
    raise RuntimeError("No walk-forward results were generated.")


# ------------------------------------------------------------
# SAVE DAILY RESULTS
# ------------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

output_path = (
    "data/processed/"
    "candidate_walk_forward_results.parquet"
)

results.to_parquet(
    output_path,
    index=False,
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 90)
print("WALK-FORWARD SUMMARY")
print("=" * 90)

print(f"Test days: {len(results):,}")

print()
print("Daily-average success rates:")
print(
    f"Baseline candidate: {results['baseline_success'].mean():.2%}"
)

for pct in TOP_PCTS:

    col = f"top_{pct}_success"

    print(
        f"Top {pct:>2}%:            "
        f"{results[col].mean():.2%}"
    )


# ------------------------------------------------------------
# POOLED RESULTS
# ------------------------------------------------------------

print()
print("Pooled results:")

total_candidates = results["candidates"].sum()

weighted_baseline = (
    (results["baseline_success"] * results["candidates"]).sum()
    / total_candidates
)

print(
    f"All candidates: "
    f"{weighted_baseline:.2%}"
)

for pct in TOP_PCTS:

    count_col = f"top_{pct}_count"
    success_col = f"top_{pct}_success"

    total_selected = results[count_col].sum()

    weighted_success = (
        (
            results[success_col]
            * results[count_col]
        ).sum()
        / total_selected
    )

    print(
        f"Top {pct:>2}%: "
        f"{total_selected:,} trades | "
        f"{weighted_success:.2%} success"
    )


# ------------------------------------------------------------
# BEST SINGLE TRADE PER DAY
# ------------------------------------------------------------

best_daily_rate = results["best_target"].mean()

print()
print(
    f"Best-ranked candidate each day: "
    f"{best_daily_rate:.2%} success"
)

print()
print("Best-ranked side distribution:")

print(
    results["best_side"].value_counts()
)


# ------------------------------------------------------------
# RECENT PERFORMANCE
# ------------------------------------------------------------

print()
print("=" * 90)
print("RECENT WALK-FORWARD PERFORMANCE")
print("=" * 90)

recent_n = min(30, len(results))

recent = results.tail(recent_n)

print(
    f"Last {recent_n} test days:"
)

print(
    f"Baseline: "
    f"{recent['baseline_success'].mean():.2%}"
)

for pct in TOP_PCTS:

    print(
        f"Top {pct:>2}%: "
        f"{recent[f'top_{pct}_success'].mean():.2%}"
    )

print(
    f"Best/day: "
    f"{recent['best_target'].mean():.2%}"
)


# ------------------------------------------------------------
# MODEL SCORE DIAGNOSTICS
# ------------------------------------------------------------

print()
print("=" * 90)
print("BEST SCORE DIAGNOSTICS")
print("=" * 90)

print(
    f"Average best score: "
    f"{results['best_score'].mean():.4f}"
)

print(
    f"Median best score: "
    f"{results['best_score'].median():.4f}"
)

print(
    f"Best score >= 0.80: "
    f"{(results['best_score'] >= 0.80).mean():.2%} of days"
)

print(
    f"Best score >= 0.85: "
    f"{(results['best_score'] >= 0.85).mean():.2%} of days"
)

print(
    f"Best score >= 0.90: "
    f"{(results['best_score'] >= 0.90).mean():.2%} of days"
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print()
print("=" * 90)
print("COMPLETE")
print("=" * 90)

print(
    f"Saved: {output_path}"
)