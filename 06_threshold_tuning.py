from pathlib import Path
import pandas as pd

PREDICTIONS_PATH = Path("data/processed/test_predictions.parquet")
RAW_PATH = Path("data/raw/candles_1m.parquet")

TARGET_PCT = 0.10
STOP_PCT = 0.05
MAX_HOLD_MINUTES = 30

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]


predictions = pd.read_parquet(PREDICTIONS_PATH)
raw = pd.read_parquet(RAW_PATH)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"], utc=True
)

raw["timestamp"] = pd.to_datetime(
    raw["timestamp"], utc=True
)

raw = raw[
    raw["instrument_type"] == "PE"
].copy()

raw_by_symbol = {
    symbol: group.sort_values("timestamp").reset_index(drop=True)
    for symbol, group in raw.groupby("symbol")
}


def backtest_threshold(threshold):

    signals = predictions[
        (predictions["instrument_type"] == "PE")
        & (predictions["candidate_setup"] == 1)
        & (predictions["prediction_probability"] >= threshold)
    ].copy()

    # Best PE candidate each minute
    signals = (
        signals
        .sort_values(
            ["timestamp", "prediction_probability"],
            ascending=[True, False]
        )
        .groupby("timestamp", as_index=False)
        .first()
    )

    trades = []
    last_exit = None

    for _, signal in signals.iterrows():

        signal_time = signal["timestamp"]
        symbol = signal["symbol"]

        if last_exit is not None and signal_time <= last_exit:
            continue

        if symbol not in raw_by_symbol:
            continue

        candles = raw_by_symbol[symbol]

        future = candles[
            candles["timestamp"] > signal_time
        ]

        if future.empty:
            continue

        entry = future.iloc[0]

        if entry["timestamp"].date() != signal_time.date():
            continue

        entry_time = entry["timestamp"]
        entry_price = entry["open"]

        target = entry_price * 1.10
        stop = entry_price * 0.95

        end_time = entry_time + pd.Timedelta(
            minutes=MAX_HOLD_MINUTES
        )

        trade_candles = candles[
            (candles["timestamp"] >= entry_time)
            & (candles["timestamp"] <= end_time)
        ]

        outcome = "TIMEOUT"
        exit_price = trade_candles.iloc[-1]["close"]
        exit_time = trade_candles.iloc[-1]["timestamp"]

        for _, candle in trade_candles.iterrows():

            target_hit = candle["high"] >= target
            stop_hit = candle["low"] <= stop

            if target_hit and stop_hit:
                outcome = "STOP"
                exit_price = stop
                exit_time = candle["timestamp"]
                break

            if stop_hit:
                outcome = "STOP"
                exit_price = stop
                exit_time = candle["timestamp"]
                break

            if target_hit:
                outcome = "TARGET"
                exit_price = target
                exit_time = candle["timestamp"]
                break

        return_pct = (
            (exit_price - entry_price)
            / entry_price
            * 100
        )

        trades.append({
            "outcome": outcome,
            "return_pct": return_pct
        })

        last_exit = exit_time

    if not trades:
        return threshold, 0, 0, 0

    trades = pd.DataFrame(trades)

    win_rate = (
        (trades["outcome"] == "TARGET").mean()
        * 100
    )

    avg_return = trades["return_pct"].mean()

    return (
        threshold,
        len(trades),
        win_rate,
        avg_return
    )


print("\nTHRESHOLD RESULTS\n")

for threshold in THRESHOLDS:

    threshold, trades, win_rate, avg_return = (
        backtest_threshold(threshold)
    )

    print(
        f"{threshold:.2f} | "
        f"Trades: {trades} | "
        f"Win Rate: {win_rate:.2f}% | "
        f"Avg Return: {avg_return:.2f}%"
    )