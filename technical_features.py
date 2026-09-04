import numpy as np
import pandas as pd


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(group, period=14):

    previous_close = group["close"].shift(1)

    tr = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - previous_close).abs(),
            (group["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def build_features(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df["date"] = df["timestamp"].dt.date

    nifty = df[
        df["instrument_type"] == "INDEX"
    ].copy()

    options = df[
        df["instrument_type"].isin(["CE", "PE"])
    ].copy()


    # ==========================================
    # NIFTY FEATURES
    # ==========================================

    def nifty_day(g):

        g = g.sort_values("timestamp").copy()

        g["nifty_return_1m"] = g["close"].pct_change(1)
        g["nifty_return_3m"] = g["close"].pct_change(3)
        g["nifty_return_5m"] = g["close"].pct_change(5)

        g["nifty_range_pct"] = (
            (g["high"] - g["low"])
            / g["close"]
        )

        g["nifty_rsi_14"] = rsi(
            g["close"],
            14
        )

        g["nifty_atr_14"] = atr(g, 14)

        g["nifty_atr_pct"] = (
            g["nifty_atr_14"]
            / g["close"]
        )

        ema9 = g["close"].ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = g["close"].ewm(
            span=21,
            adjust=False
        ).mean()

        g["nifty_ema_spread_pct"] = (
            (ema9 - ema21)
            / g["close"]
        )

        g["nifty_trend_up"] = (
            ema9 > ema21
        ).astype(int)

        g["nifty_trend_down"] = (
            ema9 < ema21
        ).astype(int)

        prior_high = (
            g["high"]
            .rolling(15)
            .max()
            .shift(1)
        )

        prior_low = (
            g["low"]
            .rolling(15)
            .min()
            .shift(1)
        )

        g["nifty_breakout_up_15"] = (
            g["close"] > prior_high
        ).astype(int)

        g["nifty_breakout_down_15"] = (
            g["close"] < prior_low
        ).astype(int)

        return g


    nifty = pd.concat(
        [
            nifty_day(g)
            for _, g in nifty.groupby(
                "date",
                sort=False
            )
        ],
        ignore_index=True
    )


    # ==========================================
    # PREVIOUS-DAY PIVOT
    # ==========================================

    daily = (
        nifty
        .groupby("date")
        .agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )
        .sort_index()
    )

    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_close"] = daily["day_close"].shift(1)

    daily["daily_pivot"] = (
        daily["prev_high"]
        + daily["prev_low"]
        + daily["prev_close"]
    ) / 3

    daily["pivot_r1"] = (
        2 * daily["daily_pivot"]
        - daily["prev_low"]
    )

    daily["pivot_s1"] = (
        2 * daily["daily_pivot"]
        - daily["prev_high"]
    )

    nifty = nifty.merge(
        daily[
            [
                "daily_pivot",
                "pivot_r1",
                "pivot_s1"
            ]
        ],
        left_on="date",
        right_index=True,
        how="left"
    )

    nifty["distance_from_pivot_pct"] = (
        (
            nifty["close"]
            - nifty["daily_pivot"]
        )
        / nifty["close"]
    )


    nifty_features = nifty[
        [
            "timestamp",
            "close",

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

            "daily_pivot",
            "pivot_r1",
            "pivot_s1",

            "distance_from_pivot_pct"
        ]
    ].rename(
        columns={
            "close": "nifty_close"
        }
    )


    # ==========================================
    # OPTION FEATURES
    # ==========================================

    def option_day(g):

        g = g.sort_values("timestamp").copy()

        g["return_1m"] = g["close"].pct_change(1)
        g["return_3m"] = g["close"].pct_change(3)
        g["return_5m"] = g["close"].pct_change(5)

        g["range_pct"] = (
            (g["high"] - g["low"])
            / g["close"]
        )

        g["body_pct"] = (
            (g["close"] - g["open"])
            / g["open"]
        )

        candle_range = (
            g["high"] - g["low"]
        ).replace(0, np.nan)

        g["candle_strength"] = (
            (g["close"] - g["open"])
            / candle_range
        )


        # --------------------------
        # RSI
        # --------------------------

        g["rsi_14"] = rsi(
            g["close"],
            14
        )


        # --------------------------
        # ATR
        # --------------------------

        g["atr_14"] = atr(g, 14)

        g["atr_pct"] = (
            g["atr_14"]
            / g["close"]
        )


        # --------------------------
        # EMA TREND
        # --------------------------

        ema9 = g["close"].ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = g["close"].ewm(
            span=21,
            adjust=False
        ).mean()

        g["ema_spread_pct"] = (
            (ema9 - ema21)
            / g["close"]
        )

        g["trend_up"] = (
            ema9 > ema21
        ).astype(int)


        # --------------------------
        # VOLUME
        # --------------------------

        volume_avg_5 = (
            g["volume"]
            .rolling(5)
            .mean()
        )

        volume_avg_20 = (
            g["volume"]
            .rolling(20)
            .mean()
        )

        g["volume_ratio_5"] = (
            g["volume"]
            / volume_avg_5.replace(0, np.nan)
        )

        g["volume_ratio_20"] = (
            g["volume"]
            / volume_avg_20.replace(0, np.nan)
        )


        # --------------------------
        # VWAP
        # --------------------------

        typical_price = (
            g["high"]
            + g["low"]
            + g["close"]
        ) / 3

        cumulative_volume = (
            g["volume"].cumsum()
        )

        cumulative_value = (
            typical_price
            * g["volume"]
        ).cumsum()

        g["vwap"] = (
            cumulative_value
            / cumulative_volume.replace(0, np.nan)
        )

        g["distance_from_vwap_pct"] = (
            (g["close"] - g["vwap"])
            / g["close"]
        )


        # --------------------------
        # BREAKOUT
        # IMPORTANT: shifted → no future leakage
        # --------------------------

        prior_high_5 = (
            g["high"]
            .rolling(5)
            .max()
            .shift(1)
        )

        prior_low_5 = (
            g["low"]
            .rolling(5)
            .min()
            .shift(1)
        )

        g["distance_from_high_5"] = (
            (g["close"] - prior_high_5)
            / g["close"]
        )

        g["distance_from_low_5"] = (
            (g["close"] - prior_low_5)
            / g["close"]
        )

        g["breakout_up_5"] = (
            g["close"] > prior_high_5
        ).astype(int)

        return g


    options = pd.concat(
        [
            option_day(g)
            for _, g in options.groupby(
                ["symbol", "date"],
                sort=False
            )
        ],
        ignore_index=True
    )


    # ==========================================
    # MERGE NIFTY
    # ==========================================

    options = options.merge(
        nifty_features,
        on="timestamp",
        how="left"
    )


    # ==========================================
    # MONEYNESS
    # ==========================================

    options["distance_from_strike"] = (
        options["nifty_close"]
        - options["strike"]
    )

    options["distance_from_strike_pct"] = (
        options["distance_from_strike"]
        / options["nifty_close"]
    )

    options["premium_pct_of_spot"] = (
        options["close"]
        / options["nifty_close"]
    )

    options["is_ce"] = (
        options["instrument_type"] == "CE"
    ).astype(int)


    # ==========================================
    # CE vs PE
    # ==========================================

    pair_close = options.pivot_table(
        index=["timestamp", "strike"],
        columns="instrument_type",
        values="close"
    ).reset_index()

    pair_close = pair_close.rename(
        columns={
            "CE": "ce_close",
            "PE": "pe_close"
        }
    )

    pair_returns = options.pivot_table(
        index=["timestamp", "strike"],
        columns="instrument_type",
        values="return_3m"
    ).reset_index()

    pair_returns = pair_returns.rename(
        columns={
            "CE": "ce_return_3m",
            "PE": "pe_return_3m"
        }
    )

    options = options.merge(
        pair_close,
        on=["timestamp", "strike"],
        how="left"
    )

    options = options.merge(
        pair_returns,
        on=["timestamp", "strike"],
        how="left"
    )

    options["ce_pe_ratio"] = (
        options["ce_close"]
        / options["pe_close"].replace(0, np.nan)
    )

    options["ce_minus_pe_return_3m"] = (
        options["ce_return_3m"]
        - options["pe_return_3m"]
    )


    # ==========================================
    # RULE-BASED SETUP SCORE
    # ==========================================

    option_breakout = (
        options["breakout_up_5"] == 1
    )

    above_vwap = (
        options["distance_from_vwap_pct"] > 0
    )

    option_trend = (
        options["trend_up"] == 1
    )

    volume_spike = (
        options["volume_ratio_20"] >= 1.3
    )

    momentum = (
        options["rsi_14"] >= 55
    )


    market_alignment = np.where(
        options["is_ce"] == 1,

        options["nifty_trend_up"] == 1,

        options["nifty_trend_down"] == 1
    )


    pivot_alignment = np.where(
        options["is_ce"] == 1,

        options["nifty_close"]
        > options["daily_pivot"],

        options["nifty_close"]
        < options["daily_pivot"]
    )


    relative_strength = np.where(
        options["is_ce"] == 1,

        options["ce_return_3m"]
        > options["pe_return_3m"],

        options["pe_return_3m"]
        > options["ce_return_3m"]
    )


    options["setup_score"] = (
        option_breakout.astype(int)
        + above_vwap.astype(int)
        + option_trend.astype(int)
        + volume_spike.astype(int)
        + momentum.astype(int)
        + market_alignment.astype(int)
        + pivot_alignment.astype(int)
        + relative_strength.astype(int)
    )


    options["candidate_setup"] = (
        options["setup_score"] >= 4
    ).astype(int)


    return options