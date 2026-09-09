from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_PATH = Path("data/processed/training_dataset.parquet")
WALK_FORWARD_PATH = Path("data/processed/walk_forward_results.parquet")
DAILY_STATS_PATH = Path("data/processed/walk_forward_daily_stats.parquet")
OUTPUT_DIR = Path("data/processed/regime_analysis")

STRATEGY_NAME = "PE_075_SETUP6"

REGIME_FEATURES = [
    "prediction_probability",
    "setup_score",
    "atr_pct",
    "rsi_14",
    "ema_spread_pct",
    "volume_ratio_20",
    "macd_histogram",
    "nifty_atr_pct",
    "nifty_rsi_14",
    "nifty_ema_spread_pct",
    "nifty_return_5m",
    "nifty_trend_up",
    "nifty_trend_down",
    "time_to_expiry_minutes",
    "time_to_expiry_days",
    "theta",
    "distance_from_vwap_pct",
    "distance_from_pivot_pct",
    "distance_from_strike_pct",
    "premium_pct_of_spot",
    "ce_pe_ratio",
]

training = pd.read_parquet(TRAINING_PATH)
trades = pd.read_parquet(WALK_FORWARD_PATH)
daily_stats = pd.read_parquet(DAILY_STATS_PATH)

training["timestamp"] = pd.to_datetime(training["timestamp"], utc=True)
trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
daily_stats["date"] = pd.to_datetime(daily_stats["date"]).dt.date

trades = trades[trades["strategy"] == STRATEGY_NAME].copy()
daily_stats = daily_stats[daily_stats["strategy"] == STRATEGY_NAME].copy()

if trades.empty:
    raise ValueError(f"No walk-forward trades found for strategy: {STRATEGY_NAME}")

print("\n================================")
print("09 REGIME ANALYSIS")
print("================================")
print("Strategy:", STRATEGY_NAME)
print("Trades:", len(trades))
print("Test days:", daily_stats["date"].nunique())

available_training_features = [
    f for f in REGIME_FEATURES
    if f in training.columns and f not in trades.columns
]

feature_lookup = (
    training[["timestamp", "symbol"] + available_training_features]
    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    .rename(columns={"timestamp": "signal_time"})
)

trades = trades.merge(
    feature_lookup,
    on=["signal_time", "symbol"],
    how="left",
    validate="many_to_one"
)

if available_training_features:
    probe = "nifty_atr_pct" if "nifty_atr_pct" in available_training_features else available_training_features[0]
    unmatched = int(trades[probe].isna().sum())
    print("Signal-feature merge unmatched:", unmatched, "/", len(trades))

trades["is_win"] = (trades["outcome"] == "TARGET").astype(int)
trades["is_stop"] = trades["outcome"].isin(["STOP", "STOP_SAME_CANDLE"]).astype(int)
trades["date"] = trades["signal_time"].dt.date
trades["signal_time_ist"] = trades["signal_time"].dt.tz_convert("Asia/Kolkata")

minutes = trades["signal_time_ist"].dt.hour * 60 + trades["signal_time_ist"].dt.minute
trades["time_bucket"] = pd.cut(
    minutes,
    bins=[555, 600, 660, 720, 780, 840, 900, 931],
    labels=[
        "09:15-10:00",
        "10:00-11:00",
        "11:00-12:00",
        "12:00-13:00",
        "13:00-14:00",
        "14:00-15:00",
        "15:00-15:30",
    ],
    right=False,
    include_lowest=True,
)

print("\n================================")
print("1. OVERALL TRADE RESULTS")
print("================================")
total = len(trades)
wins = int(trades["is_win"].sum())
stops = int(trades["is_stop"].sum())
timeouts = int((trades["outcome"] == "TIMEOUT").sum())
print("Trades:", total)
print("Wins:", wins)
print("Stops:", stops)
print("Timeouts:", timeouts)
print("Win rate:", f"{wins / total * 100:.2f}%")
print("Average return:", f"{trades['return_pct'].mean():.2f}%")

daily_trade_summary = (
    trades.groupby("date")
    .agg(
        trades=("outcome", "count"),
        wins=("is_win", "sum"),
        stops=("is_stop", "sum"),
        avg_return=("return_pct", "mean"),
        total_return=("return_pct", "sum"),
    )
    .reset_index()
)

daily_trade_summary["win_rate"] = (
    daily_trade_summary["wins"] / daily_trade_summary["trades"] * 100
)
daily_trade_summary["day_result"] = np.where(
    daily_trade_summary["total_return"] > 0,
    "PROFITABLE_DAY",
    np.where(
        daily_trade_summary["total_return"] < 0,
        "LOSING_DAY",
        "FLAT_DAY",
    ),
)

daily_all = (
    daily_stats[
        [
            "date",
            "confidence_signals",
            "setup_signals",
            "final_filter_signals",
            "best_candidates",
            "trades",
            "wins",
        ]
    ]
    .drop_duplicates(subset=["date"], keep="last")
    .merge(
        daily_trade_summary[
            [
                "date",
                "stops",
                "avg_return",
                "total_return",
                "win_rate",
                "day_result",
            ]
        ],
        on="date",
        how="left",
    )
)

daily_all["day_result"] = daily_all["day_result"].fillna("NO_TRADE_DAY")
daily_all["stops"] = daily_all["stops"].fillna(0).astype(int)

print("\n================================")
print("2. PERFORMANCE BY DAY")
print("================================")
print(daily_all.to_string(index=False))

analysis_features = [f for f in REGIME_FEATURES if f in trades.columns]

comparison_rows = []
for feature in analysis_features:
    if not pd.api.types.is_numeric_dtype(trades[feature]):
        continue
    win_vals = trades.loc[trades["is_win"] == 1, feature].dropna()
    nonwin_vals = trades.loc[trades["is_win"] == 0, feature].dropna()
    if win_vals.empty or nonwin_vals.empty:
        continue
    comparison_rows.append(
        {
            "feature": feature,
            "win_mean": win_vals.mean(),
            "nonwin_mean": nonwin_vals.mean(),
            "difference": win_vals.mean() - nonwin_vals.mean(),
            "win_median": win_vals.median(),
            "nonwin_median": nonwin_vals.median(),
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
if not comparison_df.empty:
    comparison_df["abs_difference"] = comparison_df["difference"].abs()
    comparison_df = comparison_df.sort_values("abs_difference", ascending=False)

print("\n================================")
print("3. WINNING VS NON-WINNING TRADES")
print("================================")
if comparison_df.empty:
    print("No comparable numeric regime features.")
else:
    print(
        comparison_df[
            [
                "feature",
                "win_mean",
                "nonwin_mean",
                "difference",
                "win_median",
                "nonwin_median",
            ]
        ].to_string(index=False)
    )

bucket_frames = []

time_summary = (
    trades.groupby("time_bucket", observed=True)
    .agg(
        trades=("is_win", "count"),
        wins=("is_win", "sum"),
        avg_return=("return_pct", "mean"),
    )
    .reset_index()
)
if not time_summary.empty:
    time_summary["win_rate"] = time_summary["wins"] / time_summary["trades"] * 100
    time_summary["feature"] = "time_of_day"
    time_summary["bucket"] = time_summary["time_bucket"].astype(str)
    bucket_frames.append(
        time_summary[["feature", "bucket", "trades", "wins", "win_rate", "avg_return"]]
    )

for feature in [
    "nifty_atr_pct",
    "atr_pct",
    "nifty_ema_spread_pct",
    "nifty_rsi_14",
    "time_to_expiry_minutes",
    "theta",
    "setup_score",
    "prediction_probability",
]:
    if feature not in trades.columns:
        continue
    if trades[feature].dropna().nunique() < 4:
        continue
    try:
        bucket_col = f"{feature}_bucket"
        trades[bucket_col] = pd.qcut(trades[feature], q=4, duplicates="drop")
    except ValueError:
        continue

    summary = (
        trades.groupby(bucket_col, observed=True)
        .agg(
            trades=("is_win", "count"),
            wins=("is_win", "sum"),
            avg_return=("return_pct", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = summary["wins"] / summary["trades"] * 100
    summary["feature"] = feature
    summary["bucket"] = summary[bucket_col].astype(str)
    bucket_frames.append(
        summary[["feature", "bucket", "trades", "wins", "win_rate", "avg_return"]]
    )

bucket_analysis = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()

print("\n================================")
print("4. REGIME BUCKET RESULTS")
print("================================")
if bucket_analysis.empty:
    print("Not enough variation to build buckets.")
else:
    print(bucket_analysis.to_string(index=False))

print("\n================================")
print("5. NIFTY TREND REGIME")
print("================================")
trend_summary = pd.DataFrame()
if {"nifty_trend_up", "nifty_trend_down"}.issubset(trades.columns):
    trades["nifty_regime"] = np.select(
        [
            trades["nifty_trend_up"] == 1,
            trades["nifty_trend_down"] == 1,
        ],
        [
            "UPTREND",
            "DOWNTREND",
        ],
        default="SIDEWAYS",
    )
    trend_summary = (
        trades.groupby("nifty_regime")
        .agg(
            trades=("is_win", "count"),
            wins=("is_win", "sum"),
            avg_return=("return_pct", "mean"),
        )
    )
    trend_summary["win_rate"] = trend_summary["wins"] / trend_summary["trades"] * 100
    print(trend_summary)
else:
    print("NIFTY trend columns unavailable.")

print("\n================================")
print("6. EXPIRY / THETA REGIME")
print("================================")
expiry_summary = pd.DataFrame()
if "time_to_expiry_minutes" in trades.columns:
    trades["expiry_bucket"] = pd.cut(
        trades["time_to_expiry_minutes"],
        bins=[-np.inf, 60, 180, 360, 1440, 4320, np.inf],
        labels=[
            "<= 1 hour",
            "1-3 hours",
            "3-6 hours",
            "6-24 hours",
            "1-3 days",
            "> 3 days",
        ],
        right=True,
    )
    agg_kwargs = {
        "trades": ("is_win", "count"),
        "wins": ("is_win", "sum"),
        "avg_return": ("return_pct", "mean"),
    }
    if "theta" in trades.columns:
        agg_kwargs["avg_theta"] = ("theta", "mean")

    expiry_summary = (
        trades.groupby("expiry_bucket", observed=True)
        .agg(**agg_kwargs)
    )
    expiry_summary["win_rate"] = expiry_summary["wins"] / expiry_summary["trades"] * 100
    print(expiry_summary)
else:
    print("time_to_expiry_minutes unavailable.")

print("\n================================")
print("7. INDIA VIX STATUS")
print("================================")
vix_columns = [c for c in training.columns if "vix" in c.lower()]
if vix_columns:
    print("VIX-related columns found:", vix_columns)
else:
    print("India VIX is NOT present in the current training dataset.")
    print("Using ATR / NIFTY ATR as current volatility proxies.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trade_output = OUTPUT_DIR / "pe_075_setup6_trades_with_regimes.csv"
daily_output = OUTPUT_DIR / "pe_075_setup6_daily_regimes.csv"
comparison_output = OUTPUT_DIR / "pe_075_setup6_win_loss_feature_comparison.csv"
bucket_output = OUTPUT_DIR / "pe_075_setup6_bucket_analysis.csv"

trades.to_csv(trade_output, index=False)
daily_all.to_csv(daily_output, index=False)
comparison_df.to_csv(comparison_output, index=False)
bucket_analysis.to_csv(bucket_output, index=False)

print("\n================================")
print("REGIME ANALYSIS COMPLETE")
print("================================")
print("Saved:")
print(trade_output)
print(daily_output)
print(comparison_output)
print(bucket_output)
