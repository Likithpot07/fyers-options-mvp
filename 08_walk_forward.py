from pathlib import Path

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

OUTPUT_PATH = Path(
    "data/processed/walk_forward_results.parquet"
)

MIN_TRAIN_DAYS = 8

CONFIDENCE_THRESHOLD = 0.85
SETUP_SCORE_THRESHOLD = 5

TARGET_PCT = 0.10
STOP_PCT = 0.05
MAX_HOLD_MINUTES = 30


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

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

    "setup_score"
]


# --------------------------------------------------
# LOAD
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df["date"] = df["timestamp"].dt.date


# PE ONLY
df = df[
    df["instrument_type"] == "PE"
].copy()


raw = pd.read_parquet(RAW_PATH)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    utc=True
)

raw = raw[
    raw["instrument_type"] == "PE"
].copy()


raw_by_symbol = {
    symbol: group.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    for symbol, group
    in raw.groupby("symbol")
}


dates = sorted(
    df["date"].unique()
)


print("Trading days:", len(dates))
print("Minimum training days:", MIN_TRAIN_DAYS)


# --------------------------------------------------
# BACKTEST ONE TEST DAY
# --------------------------------------------------

def backtest_day(signals):

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

    trades = []

    last_exit = None


    for _, signal in signals.iterrows():

        signal_time = signal["timestamp"]
        symbol = signal["symbol"]

        # One active trade at a time
        if (
            last_exit is not None
            and signal_time <= last_exit
        ):
            continue

        if symbol not in raw_by_symbol:
            continue


        candles = raw_by_symbol[symbol]

        future = candles[
            candles["timestamp"]
            > signal_time
        ]

        if future.empty:
            continue


        entry = future.iloc[0]

        entry_time = entry["timestamp"]

        if (
            entry_time.date()
            != signal_time.date()
        ):
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


            # Conservative
            if target_hit and stop_hit:

                outcome = "STOP"

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


        trades.append({

            "date":
                signal_time.date(),

            "symbol":
                symbol,

            "signal_time":
                signal_time,

            "entry_time":
                entry_time,

            "exit_time":
                exit_time,

            "prediction_probability":
                signal[
                    "prediction_probability"
                ],

            "setup_score":
                signal["setup_score"],

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "outcome":
                outcome,

            "return_pct":
                return_pct
        })


        last_exit = exit_time


    return trades


# --------------------------------------------------
# WALK FORWARD
# --------------------------------------------------

all_trades = []


for i in range(
    MIN_TRAIN_DAYS,
    len(dates)
):

    train_dates = dates[:i]
    test_date = dates[i]


    train = df[
        df["date"].isin(train_dates)
    ].copy()

    test = df[
        df["date"] == test_date
    ].copy()


    if train.empty or test.empty:
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


    test["prediction_probability"] = (
        model.predict_proba(
            test[FEATURES]
        )[:, 1]
    )


    signals = test[

        (
            test["prediction_probability"]
            >= CONFIDENCE_THRESHOLD
        )

        &

        (
            test["setup_score"]
            >= SETUP_SCORE_THRESHOLD
        )

    ].copy()


    trades = backtest_day(signals)

    all_trades.extend(trades)


    wins = sum(
        t["outcome"] == "TARGET"
        for t in trades
    )


    print(
        f"{test_date} | "
        f"Train days: {len(train_dates)} | "
        f"Signals: {len(signals)} | "
        f"Trades: {len(trades)} | "
        f"Wins: {wins}"
    )


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

results = pd.DataFrame(all_trades)


if results.empty:

    print("\nNo walk-forward trades.")
    raise SystemExit


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_parquet(
    OUTPUT_PATH,
    index=False
)


total = len(results)

wins = (
    results["outcome"]
    == "TARGET"
).sum()

stops = (
    results["outcome"]
    == "STOP"
).sum()

timeouts = (
    results["outcome"]
    == "TIMEOUT"
).sum()


win_rate = (
    wins / total * 100
)


print("\n================================")
print("WALK-FORWARD RESULTS")
print("================================")

print("Trades:", total)
print("Wins:", wins)
print("Stops:", stops)
print("Timeouts:", timeouts)

print(
    "Win Rate:",
    f"{win_rate:.2f}%"
)

print(
    "Average Return:",
    f"{results['return_pct'].mean():.2f}%"
)

print(
    "Median Return:",
    f"{results['return_pct'].median():.2f}%"
)


print("\nPER DAY")

daily = (
    results
    .groupby("date")
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

daily["win_rate"] = (
    daily["wins"]
    / daily["trades"]
    * 100
)

print(daily)


print("\nSaved:")
print(OUTPUT_PATH)