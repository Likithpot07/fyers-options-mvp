from pathlib import Path

import hashlib
import json

import pandas as pd
from xgboost import XGBClassifier


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_PATH = Path(
    "data/processed/training_dataset.parquet"
)

RAW_PATH = Path(
    "data/raw/candles_1m.parquet"
)

FEATURES_PATH = Path(
    "models/features.json"
)

OUTPUT_PATH = Path(
    "data/processed/walk_forward_results.parquet"
)

MIN_TRAIN_DAYS = 8

TARGET_PCT = 0.10
STOP_PCT = 0.10
MAX_HOLD_MINUTES = 30


# --------------------------------------------------
# STRATEGIES TO VALIDATE
# --------------------------------------------------
# These are NOT final live rules.
# They are candidate CE / PE rules suggested by
# the threshold + setup diagnostics.

STRATEGIES = [
    {
        "name": "CE_085_MODEL_ONLY",
        "instrument_type": "CE",
        "confidence": 0.85,
        "min_setup_score": 0,
        "require_fidelity": False,
    },
    {
        "name": "CE_090_MODEL_ONLY",
        "instrument_type": "CE",
        "confidence": 0.90,
        "min_setup_score": 0,
        "require_fidelity": False,
    },
    {
        "name": "PE_075_SETUP6",
        "instrument_type": "PE",
        "confidence": 0.75,
        "min_setup_score": 6,
        "require_fidelity": False,
    },
    {
        "name": "PE_080_SETUP5",
        "instrument_type": "PE",
        "confidence": 0.80,
        "min_setup_score": 5,
        "require_fidelity": False,
    },
    {
        "name": "PE_080_SETUP6",
        "instrument_type": "PE",
        "confidence": 0.80,
        "min_setup_score": 6,
        "require_fidelity": False,
    },
    {
        "name": "PE_080_SETUP5_FIDELITY",
        "instrument_type": "PE",
        "confidence": 0.80,
        "min_setup_score": 5,
        "require_fidelity": True,
    },
]


# --------------------------------------------------
# LOAD FEATURE MANIFEST
# --------------------------------------------------

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Missing feature manifest: {FEATURES_PATH}"
    )

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

if not isinstance(FEATURES, list):
    raise ValueError(
        "models/features.json must contain a JSON list."
    )

if len(FEATURES) != len(set(FEATURES)):
    raise ValueError(
        "Duplicate feature names found in models/features.json."
    )

FEATURES_SHA256 = hashlib.sha256(
    "\n".join(FEATURES).encode("utf-8")
).hexdigest()

print("\n================================")
print("FEATURE MANIFEST")
print("================================")
print("Feature count:", len(FEATURES))
print("Feature SHA256:", FEATURES_SHA256)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_parquet(
    DATA_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df["date"] = df["timestamp"].dt.date


required_columns = set(
    FEATURES
    + [
        "timestamp",
        "date",
        "symbol",
        "instrument_type",
        "strike",
        "target_hit",
        "setup_score",
        "fidelity_confirmed",
    ]
)

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Training dataset is missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in sorted(missing_columns)
        )
    )


raw = pd.read_parquet(
    RAW_PATH
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    utc=True
)

raw = raw[
    raw["instrument_type"].isin(
        ["CE", "PE"]
    )
].copy()

raw_by_symbol = {
    symbol: (
        group
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    for symbol, group
    in raw.groupby("symbol")
}


dates = sorted(
    df["date"].unique()
)

print("Trading days:", len(dates))
print("Minimum training days:", MIN_TRAIN_DAYS)

if len(dates) <= MIN_TRAIN_DAYS:
    raise ValueError(
        "Not enough trading days for walk-forward testing."
    )


# --------------------------------------------------
# BACKTEST ONE TEST DAY / ONE STRATEGY
# --------------------------------------------------

def backtest_day(
    signals,
    strategy_name,
    option_type,
):

    signals = (
        signals
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

    best_candidate_count = len(signals)

    trades = []
    last_exit = None
    overlap_rejected = 0
    data_rejected = 0

    for _, signal in signals.iterrows():

        signal_time = signal["timestamp"]
        symbol = signal["symbol"]

        # One active trade at a time
        # within this strategy.
        if (
            last_exit is not None
            and signal_time <= last_exit
        ):
            overlap_rejected += 1
            continue

        if symbol not in raw_by_symbol:
            data_rejected += 1
            continue

        candles = raw_by_symbol[symbol]

        future = candles[
            candles["timestamp"]
            > signal_time
        ]

        if future.empty:
            data_rejected += 1
            continue

        entry = future.iloc[0]
        entry_time = entry["timestamp"]

        if (
            entry_time.date()
            != signal_time.date()
        ):
            data_rejected += 1
            continue

        entry_price = entry["open"]

        target_price = (
            entry_price
            * (1 + TARGET_PCT)
        )

        stop_price = (
            entry_price
            * (1 - STOP_PCT)
        )

        end_time = (
            entry_time
            + pd.Timedelta(
                minutes=MAX_HOLD_MINUTES
            )
        )

        trade_candles = candles[
            (candles["timestamp"] >= entry_time)
            & (candles["timestamp"] <= end_time)
            & (
                candles["timestamp"].dt.date
                == entry_time.date()
            )
        ]

        if trade_candles.empty:
            data_rejected += 1
            continue

        outcome = "TIMEOUT"
        exit_price = (
            trade_candles.iloc[-1]["close"]
        )
        exit_time = (
            trade_candles.iloc[-1]["timestamp"]
        )

        for _, candle in trade_candles.iterrows():

            target_hit = (
                candle["high"]
                >= target_price
            )

            stop_hit = (
                candle["low"]
                <= stop_price
            )

            # Conservative:
            # if both target and stop occur in same 1m candle,
            # assume STOP happened first.
            if target_hit and stop_hit:
                outcome = "STOP_SAME_CANDLE"
                exit_price = stop_price
                exit_time = candle["timestamp"]
                break

            if stop_hit:
                outcome = "STOP"
                exit_price = stop_price
                exit_time = candle["timestamp"]
                break

            if target_hit:
                outcome = "TARGET"
                exit_price = target_price
                exit_time = candle["timestamp"]
                break

        return_pct = (
            (
                exit_price
                - entry_price
            )
            / entry_price
            * 100
        )

        trades.append(
            {
                "strategy": strategy_name,
                "instrument_type": option_type,
                "date": signal_time.date(),
                "symbol": symbol,
                "strike": signal["strike"],
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "prediction_probability":
                    signal["prediction_probability"],
                "setup_score":
                    signal["setup_score"],
                "fidelity_confirmed":
                    signal["fidelity_confirmed"],
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "exit_price": exit_price,
                "outcome": outcome,
                "return_pct": return_pct,
            }
        )

        last_exit = exit_time

    return (
        trades,
        best_candidate_count,
        overlap_rejected,
        data_rejected,
    )


# --------------------------------------------------
# WALK FORWARD
# --------------------------------------------------

all_trades = []
daily_stats = []

for i in range(
    MIN_TRAIN_DAYS,
    len(dates)
):

    train_dates = dates[:i]
    test_date = dates[i]

    print(
        "\n================================"
    )
    print(
        f"TEST DATE: {test_date} | "
        f"TRAIN DAYS: {len(train_dates)}"
    )
    print(
        "================================"
    )

    for option_type in ["CE", "PE"]:

        train = df[
            (df["instrument_type"] == option_type)
            & df["date"].isin(train_dates)
        ].copy()

        test = df[
            (df["instrument_type"] == option_type)
            & (df["date"] == test_date)
        ].copy()

        if train.empty or test.empty:
            print(
                f"{option_type}: no train/test rows"
            )
            continue

        X_train = train[FEATURES]
        y_train = train["target_hit"]

        positive = y_train.sum()
        negative = len(y_train) - positive

        scale_pos_weight = (
            negative / positive
            if positive > 0
            else 1
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

        booster_features = (
            model
            .get_booster()
            .feature_names
        )

        if booster_features != FEATURES:
            raise RuntimeError(
                f"{option_type} walk-forward model "
                "feature order mismatch."
            )

        test["prediction_probability"] = (
            model.predict_proba(
                test[FEATURES]
            )[:, 1]
        )

        option_strategies = [
            strategy
            for strategy in STRATEGIES
            if (
                strategy["instrument_type"]
                == option_type
            )
        ]

        for strategy in option_strategies:

            signals = test[
                test["prediction_probability"]
                >= strategy["confidence"]
            ].copy()

            confidence_signals = len(signals)

            signals = signals[
                signals["setup_score"]
                >= strategy["min_setup_score"]
            ].copy()

            setup_signals = len(signals)

            if strategy["require_fidelity"]:
                signals = signals[
                    signals["fidelity_confirmed"]
                    == 1
                ].copy()

            final_filter_signals = len(signals)

            (
                trades,
                best_candidates,
                overlap_rejected,
                data_rejected,
            ) = backtest_day(
                signals=signals,
                strategy_name=strategy["name"],
                option_type=option_type,
            )

            all_trades.extend(
                trades
            )

            wins = sum(
                trade["outcome"] == "TARGET"
                for trade in trades
            )

            daily_stats.append(
                {
                    "date": test_date,
                    "strategy": strategy["name"],
                    "instrument_type": option_type,
                    "confidence_signals":
                        confidence_signals,
                    "setup_signals":
                        setup_signals,
                    "final_filter_signals":
                        final_filter_signals,
                    "best_candidates":
                        best_candidates,
                    "overlap_rejected":
                        overlap_rejected,
                    "data_rejected":
                        data_rejected,
                    "trades":
                        len(trades),
                    "wins":
                        wins,
                }
            )

            print(
                f"{strategy['name']:<27} | "
                f"Conf: {confidence_signals:>4} | "
                f"Setup: {setup_signals:>4} | "
                f"Final: {final_filter_signals:>4} | "
                f"Best: {best_candidates:>3} | "
                f"Trades: {len(trades):>2} | "
                f"Wins: {wins:>2}"
            )


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results = pd.DataFrame(
    all_trades
)

daily_stats_df = pd.DataFrame(
    daily_stats
)

daily_stats_path = (
    OUTPUT_PATH.parent
    / "walk_forward_daily_stats.parquet"
)

daily_stats_df.to_parquet(
    daily_stats_path,
    index=False
)


if results.empty:

    print("\nNo walk-forward trades.")

    # Still save an empty result file.
    pd.DataFrame(
        columns=[
            "strategy",
            "instrument_type",
            "date",
            "symbol",
            "strike",
            "signal_time",
            "entry_time",
            "exit_time",
            "prediction_probability",
            "setup_score",
            "fidelity_confirmed",
            "entry_price",
            "target_price",
            "stop_price",
            "exit_price",
            "outcome",
            "return_pct",
        ]
    ).to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("Saved:")
    print(OUTPUT_PATH)
    print(daily_stats_path)

    raise SystemExit


results.to_parquet(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# SUMMARY BY STRATEGY
# --------------------------------------------------

print("\n================================")
print("WALK-FORWARD RESULTS BY STRATEGY")
print("================================")

summary = (
    results
    .groupby(
        [
            "instrument_type",
            "strategy"
        ]
    )
    .agg(
        trades=("outcome", "count"),
        wins=(
            "outcome",
            lambda x:
                (x == "TARGET").sum()
        ),
        stops=(
            "outcome",
            lambda x:
                x.isin(
                    [
                        "STOP",
                        "STOP_SAME_CANDLE"
                    ]
                ).sum()
        ),
        timeouts=(
            "outcome",
            lambda x:
                (x == "TIMEOUT").sum()
        ),
        avg_return=(
            "return_pct",
            "mean"
        ),
        median_return=(
            "return_pct",
            "median"
        ),
    )
)

summary["win_rate"] = (
    summary["wins"]
    / summary["trades"]
    * 100
)

print(summary)


# --------------------------------------------------
# SUMMARY INCLUDING ZERO-TRADE DAYS
# --------------------------------------------------

print("\n================================")
print("DAILY FUNNEL SUMMARY")
print("================================")

funnel_summary = (
    daily_stats_df
    .groupby(
        [
            "instrument_type",
            "strategy"
        ]
    )
    .agg(
        test_days=("date", "nunique"),
        confidence_signals=(
            "confidence_signals",
            "sum"
        ),
        setup_signals=(
            "setup_signals",
            "sum"
        ),
        final_filter_signals=(
            "final_filter_signals",
            "sum"
        ),
        best_candidates=(
            "best_candidates",
            "sum"
        ),
        overlap_rejected=(
            "overlap_rejected",
            "sum"
        ),
        data_rejected=(
            "data_rejected",
            "sum"
        ),
        trades=("trades", "sum"),
        wins=("wins", "sum"),
    )
)

funnel_summary["win_rate"] = (
    funnel_summary["wins"]
    / funnel_summary["trades"]
    .replace(0, pd.NA)
    * 100
)

print(funnel_summary)


# --------------------------------------------------
# PER DAY
# --------------------------------------------------

print("\n================================")
print("PER DAY")
print("================================")

daily_results = (
    results
    .groupby(
        [
            "date",
            "instrument_type",
            "strategy"
        ]
    )
    .agg(
        trades=("outcome", "count"),
        wins=(
            "outcome",
            lambda x:
                (x == "TARGET").sum()
        ),
        avg_return=(
            "return_pct",
            "mean"
        )
    )
)

daily_results["win_rate"] = (
    daily_results["wins"]
    / daily_results["trades"]
    * 100
)

print(daily_results)


print("\nSaved:")
print(OUTPUT_PATH)
print(daily_stats_path)
