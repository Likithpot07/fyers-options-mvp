import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/processed/training_dataset.parquet"

CE_MODEL_PATH = "models/candidate_xgboost_ce_5pct.json"
PE_MODEL_PATH = "models/candidate_xgboost_pe_5pct.json"

RANDOM_STATE = 42


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 80)
print("09 CANDIDATE-ONLY MODEL EXPERIMENT")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print(f"Full dataset rows: {len(df):,}")

if "candidate_setup" not in df.columns:
    raise ValueError("candidate_setup column not found")

df = df[df["candidate_setup"] == 1].copy()
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Candidate rows: {len(df):,}")

print()
print("Candidate outcome distribution:")
print(df["trade_outcome"].value_counts(dropna=False))

print()
print(f"Candidate natural success rate: {(df['trade_outcome'] == 'TARGET').mean():.2%}")


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

df["target"] = (df["trade_outcome"] == "TARGET").astype(int)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

# These are never allowed to enter the model.
# They directly describe the outcome or are identifiers.
EXCLUDE_COLUMNS = {
    "target",
    "trade_outcome",
    "target_hit",
    "entry_price",
    "timestamp",
    "symbol",
    "instrument_type",
    "expiry",
    "strike",
    "candidate_setup",

    # Do not let the first experiment simply learn the
    # existing fidelity rule.
    "fidelity_confirmed",
}


# fidelity_score is intentionally retained for this first experiment.
# If the model still fails to rank well, we can test removing it next.


# Only numeric columns can be passed to XGBoost.
feature_columns = []

for col in df.columns:
    if col in EXCLUDE_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        feature_columns.append(col)


# Remove obvious target/outcome-style columns if present.
bad_name_parts = [
    "outcome",
    "target",
    "future",
    "profit",
    "return_5",
    "stop_hit",
]

feature_columns = [
    c for c in feature_columns
    if not any(part in c.lower() for part in bad_name_parts)
]

print()
print(f"Feature count: {len(feature_columns)}")

print()
print("Features:")
for f in feature_columns:
    print(f"  {f}")


# ------------------------------------------------------------
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ------------------------------------------------------------

dates = pd.Series(df["timestamp"].dt.date.unique()).sort_values().reset_index(drop=True)

split_idx = int(len(dates) * 0.80)

train_end_date = dates.iloc[split_idx - 1]
test_start_date = dates.iloc[split_idx]

train_mask = df["timestamp"].dt.date <= train_end_date
test_mask = df["timestamp"].dt.date >= test_start_date

train = df.loc[train_mask].copy()
test = df.loc[test_mask].copy()

print()
print("=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print(f"Train: {train['timestamp'].min()} -> {train['timestamp'].max()}")
print(f"Test : {test['timestamp'].min()} -> {test['timestamp'].max()}")

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")


# ------------------------------------------------------------
# METRIC HELPERS
# ------------------------------------------------------------

def ranking_report(name, y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    baseline = y_true.mean()

    print(f"Baseline candidate success: {baseline:.2%}")

    try:
        auc = roc_auc_score(y_true, scores)
        print(f"AUC: {auc:.4f}")
    except Exception:
        print("AUC: unavailable")

    print()
    print("Score ranking:")
    print(
        f"{'Top %':>8} "
        f"{'Signals':>10} "
        f"{'Success':>12} "
        f"{'Lift':>10} "
        f"{'Avg Score':>12}"
    )

    for pct in [50, 25, 10, 5, 2, 1]:

        n = max(1, int(len(scores) * pct / 100))

        order = np.argsort(scores)[::-1]
        selected = order[:n]

        success = y_true[selected].mean()
        lift = success / baseline if baseline > 0 else np.nan
        avg_score = scores[selected].mean()

        print(
            f"{pct:>7.0f}% "
            f"{n:>10,} "
            f"{success:>11.2%} "
            f"{lift:>9.2f}x "
            f"{avg_score:>11.4f}"
        )

    print()
    print("Absolute probability buckets:")

    buckets = [
        (0.00, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 1.01),
    ]

    for lo, hi in buckets:
        mask = (scores >= lo) & (scores < hi)

        count = mask.sum()

        if count == 0:
            continue

        success = y_true[mask].mean()

        print(
            f"{lo:.2f}-{hi:.2f}: "
            f"{count:,} rows | "
            f"{success:.2%} success"
        )


# ------------------------------------------------------------
# TRAIN ONE MODEL
# ------------------------------------------------------------

def train_side(side_name, train_df, test_df, model_path):

    print()
    print("=" * 80)
    print(f"{side_name} CANDIDATE MODEL")
    print("=" * 80)

    train_side_df = train_df[train_df["instrument_type"] == side_name].copy()
    test_side_df = test_df[test_df["instrument_type"] == side_name].copy()

    print(f"Train rows: {len(train_side_df):,}")
    print(f"Test rows : {len(test_side_df):,}")

    if len(train_side_df) == 0 or len(test_side_df) == 0:
        print("Not enough data for this side.")
        return None

    X_train = train_side_df[feature_columns].copy()
    y_train = train_side_df["target"].astype(int)

    X_test = test_side_df[feature_columns].copy()
    y_test = test_side_df["target"].astype(int)

    train_success = y_train.mean()
    test_success = y_test.mean()

    print(f"Train success: {train_success:.2%}")
    print(f"Test success : {test_success:.2%}")

    positives = y_train.sum()
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
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

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    scores = model.predict_proba(X_test)[:, 1]

    ranking_report(
        side_name,
        y_test.values,
        scores,
    )

    os.makedirs("models", exist_ok=True)

    model.save_model(model_path)

    print()
    print(f"Saved model: {model_path}")

    return {
        "model": model,
        "test_df": test_side_df,
        "scores": scores,
    }


# ------------------------------------------------------------
# TRAIN CE + PE
# ------------------------------------------------------------

ce_result = train_side(
    "CE",
    train,
    test,
    CE_MODEL_PATH,
)

pe_result = train_side(
    "PE",
    train,
    test,
    PE_MODEL_PATH,
)


# ------------------------------------------------------------
# COMBINED TEST RESULT
# ------------------------------------------------------------

print()
print("=" * 80)
print("COMBINED CANDIDATE RESULTS")
print("=" * 80)

combined_parts = []

for result in [ce_result, pe_result]:

    if result is None:
        continue

    temp = result["test_df"].copy()
    temp["model_score"] = result["scores"]

    combined_parts.append(temp)

if combined_parts:

    combined = pd.concat(
        combined_parts,
        ignore_index=True,
    )

    ranking_report(
        "COMBINED CE + PE",
        combined["target"].values,
        combined["model_score"].values,
    )


# ------------------------------------------------------------
# SAVE FEATURE LIST
# ------------------------------------------------------------

feature_path = "models/candidate_features.json"

with open(feature_path, "w") as f:
    json.dump(feature_columns, f, indent=2)

print()
print(f"Saved feature list: {feature_path}")

print()
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)