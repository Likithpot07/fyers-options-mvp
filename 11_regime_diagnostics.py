import numpy as np
import pandas as pd


DATA_PATH = "data/processed/training_dataset.parquet"

print("=" * 90)
print("11 REGIME / FEATURE DIAGNOSTICS")
print("=" * 90)

df = pd.read_parquet(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"] = df["timestamp"].dt.date

# Candidate setups only
df = df[df["candidate_setup"] == 1].copy()

df["success"] = (df["trade_outcome"] == "TARGET").astype(int)

print(f"Candidate rows: {len(df):,}")
print(f"Trading days: {df['date'].nunique():,}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
print(f"Overall success: {df['success'].mean():.2%}")


# ============================================================
# PERIODS
# ============================================================

dates = sorted(df["date"].unique())

recent_days = min(30, len(dates))

recent_dates = set(dates[-recent_days:])

recent = df[df["date"].isin(recent_dates)].copy()
older = df[~df["date"].isin(recent_dates)].copy()

print()
print("=" * 90)
print("PERIOD COMPARISON")
print("=" * 90)

print(
    f"Older period:  {older['date'].min()} -> {older['date'].max()} "
    f"({older['date'].nunique()} days)"
)

print(
    f"Recent period: {recent['date'].min()} -> {recent['date'].max()} "
    f"({recent['date'].nunique()} days)"
)

print()
print(f"Older success : {older['success'].mean():.2%}")
print(f"Recent success: {recent['success'].mean():.2%}")


# ============================================================
# CE / PE
# ============================================================

print()
print("=" * 90)
print("CE vs PE")
print("=" * 90)

for side in ["CE", "PE"]:

    a = older[older["instrument_type"] == side]
    r = recent[recent["instrument_type"] == side]

    print()
    print(side)

    print(
        f"Older : {len(a):,} rows | "
        f"{a['success'].mean():.2%} success"
    )

    print(
        f"Recent: {len(r):,} rows | "
        f"{r['success'].mean():.2%} success"
    )

    if len(a) > 0 and len(r) > 0:
        print(
            f"Change: "
            f"{(r['success'].mean() - a['success'].mean()) * 100:+.2f} pp"
        )


# ============================================================
# SETUP SCORE
# ============================================================

print()
print("=" * 90)
print("SETUP SCORE")
print("=" * 90)

if "setup_score" in df.columns:

    for period_name, period in [
        ("OLDER", older),
        ("RECENT", recent),
    ]:

        print()
        print(period_name)

        grouped = (
            period
            .groupby("setup_score")["success"]
            .agg(["count", "mean"])
            .reset_index()
        )

        for _, row in grouped.iterrows():

            print(
                f"Score {int(row['setup_score']):2d}: "
                f"{int(row['count']):7,} rows | "
                f"{row['mean']:.2%}"
            )


# ============================================================
# FIDELITY SCORE
# ============================================================

print()
print("=" * 90)
print("FIDELITY SCORE")
print("=" * 90)

if "fidelity_score" in df.columns:

    bins = [-np.inf, 0, 0.25, 0.50, 0.75, 1.0, np.inf]
    labels = [
        "<=0",
        "0-0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-1.00",
        ">1",
    ]

    for period_name, period in [
        ("OLDER", older),
        ("RECENT", recent),
    ]:

        print()
        print(period_name)

        temp = period.copy()

        temp["fidelity_bucket"] = pd.cut(
            temp["fidelity_score"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        grouped = (
            temp
            .groupby("fidelity_bucket", observed=False)["success"]
            .agg(["count", "mean"])
            .reset_index()
        )

        for _, row in grouped.iterrows():

            if row["count"] == 0:
                continue

            print(
                f"{str(row['fidelity_bucket']):>10}: "
                f"{int(row['count']):7,} rows | "
                f"{row['mean']:.2%}"
            )


# ============================================================
# MONEyness
# ============================================================

print()
print("=" * 90)
print("MONEYNESS")
print("=" * 90)


def classify_moneyness(x):

    x = abs(float(x))

    if x <= 0.005:
        return "ATM <=0.5%"

    if x <= 0.010:
        return "0.5%-1%"

    return ">1%"


if "distance_from_strike_pct" in df.columns:

    for period_name, period in [
        ("OLDER", older),
        ("RECENT", recent),
    ]:

        temp = period.copy()

        temp["moneyness_bucket"] = temp[
            "distance_from_strike_pct"
        ].apply(classify_moneyness)

        grouped = (
            temp
            .groupby("moneyness_bucket")["success"]
            .agg(["count", "mean"])
            .reset_index()
        )

        print()
        print(period_name)

        for _, row in grouped.iterrows():

            print(
                f"{row['moneyness_bucket']:>12}: "
                f"{int(row['count']):7,} rows | "
                f"{row['mean']:.2%}"
            )


# ============================================================
# TIME TO EXPIRY
# ============================================================

print()
print("=" * 90)
print("TIME TO EXPIRY")
print("=" * 90)

if "time_to_expiry_minutes" in df.columns:

    bins = [
        -np.inf,
        60,
        120,
        240,
        480,
        960,
        np.inf,
    ]

    labels = [
        "<1h",
        "1-2h",
        "2-4h",
        "4-8h",
        "8-16h",
        ">16h",
    ]

    for period_name, period in [
        ("OLDER", older),
        ("RECENT", recent),
    ]:

        temp = period.copy()

        temp["expiry_bucket"] = pd.cut(
            temp["time_to_expiry_minutes"],
            bins=bins,
            labels=labels,
        )

        grouped = (
            temp
            .groupby("expiry_bucket", observed=False)["success"]
            .agg(["count", "mean"])
            .reset_index()
        )

        print()
        print(period_name)

        for _, row in grouped.iterrows():

            if row["count"] == 0:
                continue

            print(
                f"{str(row['expiry_bucket']):>8}: "
                f"{int(row['count']):7,} rows | "
                f"{row['mean']:.2%}"
            )


# ============================================================
# NIFTY TREND
# ============================================================

print()
print("=" * 90)
print("NIFTY TREND")
print("=" * 90)

trend_columns = [
    "nifty_trend_up",
    "nifty_trend_down",
]

for col in trend_columns:

    if col not in df.columns:
        continue

    print()
    print(col)

    for period_name, period in [
        ("OLDER", older),
        ("RECENT", recent),
    ]:

        temp = period[period[col].notna()].copy()

        grouped = (
            temp
            .groupby(col)["success"]
            .agg(["count", "mean"])
            .reset_index()
        )

        print(period_name)

        for _, row in grouped.iterrows():

            print(
                f"  {col}={row[col]}: "
                f"{int(row['count']):7,} rows | "
                f"{row['mean']:.2%}"
            )


# ============================================================
# IMPORTANT FEATURE CORRELATIONS
# ============================================================

print()
print("=" * 90)
print("FEATURE -> SUCCESS CORRELATION")
print("=" * 90)

exclude = {
    "success",
    "candidate_setup",
    "target_hit",
    "entry_price",
}

numeric_features = []

for col in df.columns:

    if col in exclude:
        continue

    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    numeric_features.append(col)


def correlation_table(period):

    rows = []

    for col in numeric_features:

        x = period[col]

        if x.isna().all():
            continue

        corr = x.corr(period["success"])

        if pd.isna(corr):
            continue

        rows.append(
            {
                "feature": col,
                "correlation": corr,
                "abs_corr": abs(corr),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr", ascending=False)
    )


for period_name, period in [
    ("OLDER", older),
    ("RECENT", recent),
]:

    table = correlation_table(period)

    print()
    print(period_name)

    print()
    print("Strongest positive relationships:")

    for _, row in (
        table
        .sort_values("correlation", ascending=False)
        .head(15)
        .iterrows()
    ):

        print(
            f"{row['feature']:40s} "
            f"{row['correlation']:+.4f}"
        )

    print()
    print("Strongest negative relationships:")

    for _, row in (
        table
        .sort_values("correlation", ascending=True)
        .head(15)
        .iterrows()
    ):

        print(
            f"{row['feature']:40s} "
            f"{row['correlation']:+.4f}"
        )


# ============================================================
# FEATURE STABILITY
# ============================================================

print()
print("=" * 90)
print("FEATURE STABILITY: OLDER vs RECENT")
print("=" * 90)

old_corr = correlation_table(older).set_index("feature")["correlation"]
recent_corr = correlation_table(recent).set_index("feature")["correlation"]

stability = pd.concat(
    [
        old_corr.rename("older"),
        recent_corr.rename("recent"),
    ],
    axis=1,
).dropna()

stability["change"] = (
    stability["recent"] - stability["older"]
)

stability["sign_flip"] = (
    np.sign(stability["older"])
    != np.sign(stability["recent"])
)

print()
print("Largest absolute correlation changes:")

for feature, row in (
    stability
    .assign(abs_change=lambda x: x["change"].abs())
    .sort_values("abs_change", ascending=False)
    .head(25)
    .iterrows()
):

    print(
        f"{feature:40s} "
        f"older={row['older']:+.4f} "
        f"recent={row['recent']:+.4f} "
        f"change={row['change']:+.4f} "
        f"{'SIGN FLIP' if row['sign_flip'] else ''}"
    )


# ============================================================
# MODEL SCORE RESULT FROM WALK FORWARD
# ============================================================

WF_PATH = (
    "data/processed/"
    "candidate_walk_forward_results.parquet"
)

if __import__("os").path.exists(WF_PATH):

    wf = pd.read_parquet(WF_PATH)

    wf["date"] = pd.to_datetime(
        wf["date"]
    ).dt.date

    print()
    print("=" * 90)
    print("WALK-FORWARD DECAY")
    print("=" * 90)

    print()
    print("10-day rolling performance:")

    for pct in [10, 5, 2, 1]:

        col = f"top_{pct}_success"

        rolling = wf[col].rolling(10).mean()

        if rolling.notna().any():

            print(
                f"Top {pct}% final 10-day average: "
                f"{rolling.iloc[-1]:.2%}"
            )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)