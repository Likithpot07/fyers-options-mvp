import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


DATA_PATH = Path("data/processed/training_dataset.parquet")
OUTPUT_PATH = Path("data/processed/regime_adaptive_predictions.parquet")

# ============================================================
# CONFIG
# ============================================================

TRAIN_DAYS = 60
MIN_TRAIN_DAYS = 30

# We deliberately do NOT use a fixed probability threshold.
# The objective is ranking candidate setups within each day.
TOP_PCTS = [0.10, 0.05, 0.02, 0.01]

RANDOM_STATE = 42

TARGET_COLUMN = "target_hit"
CANDIDATE_COLUMN = "candidate_setup"

# Features that can indicate regime.
REGIME_FEATURES = [
    "nifty_trend_up",
    "nifty_trend_down",
    "nifty_return_1m",
    "nifty_return_3m",
    "nifty_return_5m",
    "nifty_rsi14",
    "nifty_ema_spread",
    "nifty_distance_to_vwap",
    "nifty_distance_to_support",
    "nifty_distance_to_resistance",
    "nifty_upper_wick_ratio",
    "nifty_lower_wick_ratio",
    "nifty_false_breakout_up_5",
    "nifty_false_breakout_down_5",
    "nifty_breakout_strength_up_5",
    "nifty_breakout_strength_down_5",
]


# ============================================================
# HELPERS
# ============================================================

def make_regime(row):
    """
    Simple deterministic regime classification.

    We intentionally use broad regimes rather than trying to
    optimize dozens of regime labels.
    """

    trend_up = row.get("nifty_trend_up", 0)
    trend_down = row.get("nifty_trend_down", 0)

    rsi = row.get("nifty_rsi14", np.nan)
    ema = row.get("nifty_ema_spread", np.nan)
    ret5 = row.get("nifty_return_5m", np.nan)

    if pd.isna(rsi):
        rsi = 50.0
    if pd.isna(ema):
        ema = 0.0
    if pd.isna(ret5):
        ret5 = 0.0

    # Strong directional regimes
    if trend_up == 1 and ema > 0 and ret5 > 0:
        return "BULL"

    if trend_down == 1 and ema < 0 and ret5 < 0:
        return "BEAR"

    # Momentum extremes
    if rsi >= 65 and ret5 > 0:
        return "BULL_MOMENTUM"

    if rsi <= 35 and ret5 < 0:
        return "BEAR_MOMENTUM"

    return "SIDEWAYS"


def safe_auc(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, p)
    except Exception:
        return np.nan


def fit_model(train_df, feature_cols):
    X = train_df[feature_cols]
    y = train_df[TARGET_COLUMN].astype(int)

    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos == 0 or neg == 0:
        return None

    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=8,
        gamma=0.15,
        reg_alpha=0.15,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X, y)
    return model


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("REGIME-ADAPTIVE FYERS MODEL")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset rows: {len(df):,}")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

if CANDIDATE_COLUMN not in df.columns:
    raise ValueError(f"Missing {CANDIDATE_COLUMN}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Missing {TARGET_COLUMN}")

# Candidate-only modeling.
df = df[df[CANDIDATE_COLUMN] == 1].copy()

df["date"] = df["timestamp"].dt.date

print(f"Candidate rows: {len(df):,}")
print(f"Trading days: {df['date'].nunique()}")

# ============================================================
# REGIME
# ============================================================

for col in REGIME_FEATURES:
    if col not in df.columns:
        df[col] = np.nan

df["regime"] = df.apply(make_regime, axis=1)

print("\nREGIME DISTRIBUTION")
print("-" * 80)

regime_stats = (
    df.groupby("regime")[TARGET_COLUMN]
    .agg(["count", "mean"])
    .sort_values("count", ascending=False)
)

for regime, row in regime_stats.iterrows():
    print(
        f"{regime:16s} "
        f"N={int(row['count']):8,d} "
        f"Success={row['mean'] * 100:6.2f}%"
    )


# ============================================================
# FEATURES
# ============================================================

DROP_COLS = {
    TARGET_COLUMN,
    CANDIDATE_COLUMN,
    "timestamp",
    "date",
    "symbol",
    "expiry",
    "instrument_type",
    "regime",

    # Avoid leakage / direct outcome fields
    "trade_outcome",
    "target_hit",
}

feature_cols = []

for col in df.columns:
    if col in DROP_COLS:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        feature_cols.append(col)

# Explicitly exclude the historical model's prediction if present.
feature_cols = [
    c for c in feature_cols
    if not c.lower().startswith("prediction")
    and c.lower() not in {"probability", "model_probability"}
]

print(f"\nFeatures used: {len(feature_cols)}")


# ============================================================
# CHRONOLOGICAL WALK FORWARD
# ============================================================

days = sorted(df["date"].unique())

results = []

print("\n" + "=" * 80)
print("WALK-FORWARD")
print("=" * 80)

for i, test_date in enumerate(days):

    if i < MIN_TRAIN_DAYS:
        continue

    train_start_idx = max(0, i - TRAIN_DAYS)
    train_dates = days[train_start_idx:i]

    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"] == test_date].copy()

    if len(train) < 5000 or len(test) == 0:
        continue

    # --------------------------------------------------------
    # Regime-adaptive training
    #
    # If enough observations exist for the current regime,
    # preferentially train on that regime.
    # Otherwise fall back to the complete rolling window.
    # --------------------------------------------------------

    current_regimes = test["regime"].value_counts()

    predictions = []

    for instrument_type in ["CE", "PE"]:

        test_side = test[test["instrument_type"] == instrument_type].copy()

        if len(test_side) == 0:
            continue

        current_regime = test_side["regime"].mode()

        if len(current_regime):
            current_regime = current_regime.iloc[0]
        else:
            current_regime = "SIDEWAYS"

        train_side = train[
            train["instrument_type"] == instrument_type
        ].copy()

        regime_train = train_side[
            train_side["regime"] == current_regime
        ].copy()

        # Require meaningful regime sample.
        if len(regime_train) >= 5000:
            adaptive_train = regime_train
            mode = "REGIME"
        else:
            adaptive_train = train_side
            mode = "ROLLING"

        model = fit_model(adaptive_train, feature_cols)

        if model is None:
            continue

        test_side["model_probability"] = model.predict_proba(
            test_side[feature_cols]
        )[:, 1]

        test_side["training_mode"] = mode
        test_side["training_rows"] = len(adaptive_train)

        predictions.append(test_side)

    if not predictions:
        continue

    pred = pd.concat(predictions, ignore_index=True)

    # --------------------------------------------------------
    # Daily ranking
    # --------------------------------------------------------

    pred = pred.sort_values(
        "model_probability",
        ascending=False
    ).reset_index(drop=True)

    n = len(pred)

    pred["rank"] = np.arange(1, n + 1)
    pred["rank_pct"] = pred["rank"] / n

    base_rate = pred[TARGET_COLUMN].mean()

    row = {
        "date": test_date,
        "n": n,
        "baseline": base_rate,
    }

    for pct in TOP_PCTS:
        k = max(1, int(np.ceil(n * pct)))
        selected = pred.head(k)

        row[f"top{int(pct * 100)}"] = selected[TARGET_COLUMN].mean()
        row[f"top{int(pct * 100)}_n"] = len(selected)

    # Best single signal of the day
    row["top1_trade"] = pred.iloc[0][TARGET_COLUMN]

    # CE / PE
    for side in ["CE", "PE"]:
        side_df = pred[pred["instrument_type"] == side]

        if len(side_df):
            row[f"{side.lower()}_n"] = len(side_df)
            row[f"{side.lower()}_success"] = side_df[TARGET_COLUMN].mean()
        else:
            row[f"{side.lower()}_n"] = 0
            row[f"{side.lower()}_success"] = np.nan

    # AUC
    row["auc"] = safe_auc(
        pred[TARGET_COLUMN],
        pred["model_probability"]
    )

    # Current regime
    row["dominant_regime"] = pred["regime"].mode().iloc[0]

    results.append(row)

    print(
        f"{test_date} | "
        f"N={n:5d} | "
        f"Base={base_rate*100:5.1f}% | "
        f"Top10={row['top10']*100:5.1f}% | "
        f"Top5={row['top5']*100:5.1f}% | "
        f"Top2={row['top2']*100:5.1f}% | "
        f"Top1={row['top1']*100:5.1f}% | "
        f"AUC={row['auc']:.3f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No walk-forward results produced.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_parquet(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(f"Test days: {len(results_df)}")

for name in ["baseline", "top10", "top5", "top2", "top1"]:
    if name not in results_df:
        continue

    print(
        f"{name.upper():10s} "
        f"daily avg = {results_df[name].mean()*100:6.2f}% | "
        f"median = {results_df[name].median()*100:6.2f}%"
    )

print("\nPOOLED APPROXIMATION")

for pct in TOP_PCTS:
    col = f"top{int(pct * 100)}"
    ncol = f"{col}_n"

    if col not in results_df:
        continue

    weighted = (
        (results_df[col] * results_df[ncol]).sum()
        / results_df[ncol].sum()
    )

    print(
        f"{col.upper():8s} "
        f"N={int(results_df[ncol].sum()):,} "
        f"Success={weighted*100:6.2f}%"
    )

print("\nCE / PE")

for side in ["ce", "pe"]:
    if f"{side}_success" in results_df:
        valid = results_df[f"{side}_success"].dropna()

        print(
            f"{side.upper():4s} "
            f"Daily avg success={valid.mean()*100:6.2f}%"
        )

print("\nREGIME PERFORMANCE")

for regime in sorted(results_df["dominant_regime"].dropna().unique()):

    r = results_df[
        results_df["dominant_regime"] == regime
    ]

    print(
        f"{regime:16s} "
        f"Days={len(r):3d} | "
        f"Base={r['baseline'].mean()*100:6.2f}% | "
        f"Top10={r['top10'].mean()*100:6.2f}% | "
        f"Top5={r['top5'].mean()*100:6.2f}% | "
        f"Top2={r['top2'].mean()*100:6.2f}% | "
        f"Top1={r['top1'].mean()*100:6.2f}%"
    )

print("\nSaved:")
print(OUTPUT_PATH)

print("\nDONE")