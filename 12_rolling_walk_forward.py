import os
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_PATH = "data/processed/training_dataset.parquet"

RANDOM_STATE = 42

# Only the most recent N trading days are used for training.
TRAINING_WINDOW_DAYS = 60

# Don't start testing until enough history exists.
MIN_TRAIN_DAYS = 60

TOP_PCTS = [10, 5, 2, 1]


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("12 ROLLING-WINDOW CANDIDATE WALK-FORWARD")
print("=" * 90)

df = pd.read_parquet(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"] = df["timestamp"].dt.date

df = df[df["candidate_setup"] == 1].copy()

df["target"] = (
    df["trade_outcome"] == "TARGET"
).astype(int)

df = df.sort_values("timestamp").reset_index(drop=True)

dates = sorted(df["date"].unique())

print(f"Candidate rows: {len(df):,}")
print(f"Trading days: {len(dates):,}")
print(f"Date range: {dates[0]} -> {dates[-1]}")
print(f"Training window: {TRAINING_WINDOW_DAYS} trading days")


# ============================================================
# FEATURES
# ============================================================

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

    # Existing rule deliberately excluded.
    "fidelity_confirmed",
}

bad_parts = [
    "outcome",
    "target",
    "future",
    "profit",
    "return_5",
]

feature_columns = []

for col in df.columns:

    if col in EXCLUDE_COLUMNS:
        continue

    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    lower = col.lower()

    if any(part in lower for part in bad_parts):
        continue

    feature_columns.append(col)

print(f"Feature count: {len(feature_columns)}")


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# WALK FORWARD
# ============================================================

results = []

for i in range(MIN_TRAIN_DAYS, len(dates)):

    test_date = dates[i]

    # --------------------------------------------------------
    # ROLLING TRAINING WINDOW
    # --------------------------------------------------------

    train_dates = dates[
        i - TRAINING_WINDOW_DAYS:i
    ]

    train_df = df[
        df["date"].isin(train_dates)
    ].copy()

    test_df = df[
        df["date"] == test_date
    ].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        continue

    predictions = []

    # --------------------------------------------------------
    # CE / PE SEPARATE
    # --------------------------------------------------------

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

        predictions.append(temp)

    if not predictions:
        continue

    day = pd.concat(
        predictions,
        ignore_index=True,
    )

    day = day.sort_values(
        "model_score",
        ascending=False
    ).reset_index(drop=True)

    n_candidates = len(day)

    result = {
        "date": test_date,
        "candidates": n_candidates,
        "baseline_success": day["target"].mean(),
    }

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    for pct in TOP_PCTS:

        n = max(
            1,
            int(np.ceil(n_candidates * pct / 100))
        )

        selected = day.head(n)

        result[f"top_{pct}_count"] = len(selected)
        result[f"top_{pct}_success"] = (
            selected["target"].mean()
        )

    # --------------------------------------------------------
    # BEST TRADE OF THE DAY
    # --------------------------------------------------------

    best = day.iloc[0]

    result["best_score"] = best["model_score"]
    result["best_target"] = best["target"]
    result["best_side"] = best["instrument_type"]

    results.append(result)

    print(
        f"{test_date} | "
        f"Train: {train_dates[0]}->{train_dates[-1]} | "
        f"N={n_candidates:4d} | "
        f"Base {result['baseline_success']:.1%} | "
        f"Top10 {result['top_10_success']:.1%} | "
        f"Top5 {result['top_5_success']:.1%} | "
        f"Top2 {result['top_2_success']:.1%} | "
        f"Top1 {result['top_1_success']:.1%}"
    )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(results)

if results.empty:
    raise RuntimeError("No results generated.")

os.makedirs("data/processed", exist_ok=True)

output_path = (
    "data/processed/"
    "candidate_rolling_60d_walk_forward.parquet"
)

results.to_parquet(
    output_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("ROLLING 60-DAY WALK-FORWARD SUMMARY")
print("=" * 90)

print(f"Test days: {len(results):,}")

print()
print("Daily-average success:")

print(
    f"Baseline: "
    f"{results['baseline_success'].mean():.2%}"
)

for pct in TOP_PCTS:

    print(
        f"Top {pct:>2}%:  "
        f"{results[f'top_{pct}_success'].mean():.2%}"
    )


# ============================================================
# POOLED
# ============================================================

print()
print("Pooled results:")

total_candidates = results["candidates"].sum()

baseline = (
    (
        results["baseline_success"]
        * results["candidates"]
    ).sum()
    / total_candidates
)

print(
    f"All candidates: "
    f"{baseline:.2%}"
)

for pct in TOP_PCTS:

    count_col = f"top_{pct}_count"
    success_col = f"top_{pct}_success"

    total = results[count_col].sum()

    success = (
        (
            results[success_col]
            * results[count_col]
        ).sum()
        / total
    )

    lift = (
        success / baseline
        if baseline > 0
        else np.nan
    )

    print(
        f"Top {pct:>2}%: "
        f"{total:,} trades | "
        f"{success:.2%} success | "
        f"{lift:.2f}x baseline"
    )


# ============================================================
# RECENT 30 DAYS
# ============================================================

print()
print("=" * 90)
print("LATEST 30 TEST DAYS")
print("=" * 90)

recent = results.tail(
    min(30, len(results))
)

print(
    f"Baseline: "
    f"{recent['baseline_success'].mean():.2%}"
)

for pct in TOP_PCTS:

    print(
        f"Top {pct:>2}%:  "
        f"{recent[f'top_{pct}_success'].mean():.2%}"
    )


# ============================================================
# LAST 10 DAYS
# ============================================================

print()
print("=" * 90)
print("LATEST 10 TEST DAYS")
print("=" * 90)

recent10 = results.tail(
    min(10, len(results))
)

print(
    f"Baseline: "
    f"{recent10['baseline_success'].mean():.2%}"
)

for pct in TOP_PCTS:

    print(
        f"Top {pct:>2}%:  "
        f"{recent10[f'top_{pct}_success'].mean():.2%}"
    )


# ============================================================
# BEST PER DAY
# ============================================================

print()
print(
    f"Best-ranked candidate/day: "
    f"{results['best_target'].mean():.2%}"
)

print()
print("Best-ranked side:")

print(
    results["best_side"].value_counts()
)


# ============================================================
# SCORE
# ============================================================

print()
print("=" * 90)
print("BEST SCORE")
print("=" * 90)

print(
    f"Average: "
    f"{results['best_score'].mean():.4f}"
)

print(
    f"Median: "
    f"{results['best_score'].median():.4f}"
)

print(
    f">=0.80: "
    f"{(results['best_score'] >= 0.80).mean():.2%}"
)

print(
    f">=0.85: "
    f"{(results['best_score'] >= 0.85).mean():.2%}"
)

print(
    f">=0.90: "
    f"{(results['best_score'] >= 0.90).mean():.2%}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 90)
print("COMPLETE")
print("=" * 90)

print(f"Saved: {output_path}")
