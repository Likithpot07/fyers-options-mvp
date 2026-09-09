import numpy as np
import pandas as pd
from scipy.special import ndtr


def _safe_div(numerator, denominator):
    """Divide while avoiding infinities from zero denominators."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


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

    rs = _safe_div(avg_gain, avg_loss)

    return 100 - (100 / (1 + rs))


def atr(group, period=14):
    previous_close = group["close"].shift(1)

    true_range = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - previous_close).abs(),
            (group["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()



# ==================================================
# OPTION TIME-DECAY / THETA HELPERS
# ==================================================

RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.0
MIN_IV = 0.01
MAX_IV = 5.00
IV_BISECTION_STEPS = 30


def _normalize_expiry_to_market_close(expiry):
    """
    Normalize an option expiry value to 15:30 IST on its expiry date.

    This keeps FYERS historical expiry timestamps and Upstox date-only
    expiries consistent.
    """
    expiry = pd.to_datetime(
        expiry,
        utc=True,
        errors="coerce"
    )

    expiry_ist = expiry.dt.tz_convert(
        "Asia/Kolkata"
    )

    expiry_date_text = expiry_ist.dt.strftime(
        "%Y-%m-%d"
    )

    market_close_local = pd.to_datetime(
        expiry_date_text + " 15:30:00",
        errors="coerce"
    ).dt.tz_localize(
        "Asia/Kolkata",
        nonexistent="NaT",
        ambiguous="NaT"
    )

    return market_close_local.dt.tz_convert("UTC")


def _black_scholes_price(
    spot,
    strike,
    time_years,
    volatility,
    is_ce
):
    """
    Vectorized Black-Scholes option value.

    Used only to estimate IV so that historical and live theta are derived
    consistently from the same candle inputs.
    """
    sqrt_t = np.sqrt(time_years)

    sigma_sqrt_t = (
        volatility
        * sqrt_t
    )

    d1 = (
        np.log(spot / strike)
        + (
            RISK_FREE_RATE
            - DIVIDEND_YIELD
            + 0.5 * volatility ** 2
        ) * time_years
    ) / sigma_sqrt_t

    d2 = (
        d1
        - sigma_sqrt_t
    )

    discounted_spot = (
        spot
        * np.exp(
            -DIVIDEND_YIELD
            * time_years
        )
    )

    discounted_strike = (
        strike
        * np.exp(
            -RISK_FREE_RATE
            * time_years
        )
    )

    call_price = (
        discounted_spot
        * ndtr(d1)
        - discounted_strike
        * ndtr(d2)
    )

    put_price = (
        discounted_strike
        * ndtr(-d2)
        - discounted_spot
        * ndtr(-d1)
    )

    return np.where(
        is_ce,
        call_price,
        put_price
    )


def _estimate_iv_and_theta(options):
    """
    Estimate implied volatility from option close price and calculate
    Black-Scholes theta per calendar day.

    Historical candles do not contain broker Greeks, so deriving theta from
    the same OHLC + spot + strike + expiry inputs keeps training/live feature
    definitions consistent.
    """
    options = options.copy()

    expiry_utc = _normalize_expiry_to_market_close(
        options["expiry"]
    )

    options["expiry"] = expiry_utc

    time_minutes = (
        (
            expiry_utc
            - options["timestamp"]
        ).dt.total_seconds()
        / 60.0
    )

    # Avoid negative values after expiry.
    time_minutes = time_minutes.clip(
        lower=0
    )

    options["time_to_expiry_minutes"] = (
        time_minutes
    )

    options["time_to_expiry_days"] = (
        time_minutes
        / (60.0 * 24.0)
    )

    # Black-Scholes time is expressed in years.
    time_years = (
        options["time_to_expiry_days"]
        / 365.0
    )

    spot = options[
        "nifty_close"
    ].astype(float).to_numpy()

    strike = options[
        "strike"
    ].astype(float).to_numpy()

    premium = options[
        "close"
    ].astype(float).to_numpy()

    t = time_years.astype(float).to_numpy()

    is_ce = (
        options["instrument_type"]
        .eq("CE")
        .to_numpy()
    )

    valid = (
        np.isfinite(spot)
        & np.isfinite(strike)
        & np.isfinite(premium)
        & np.isfinite(t)
        & (spot > 0)
        & (strike > 0)
        & (premium > 0)
        & (t > 0)
    )

    iv = np.full(
        len(options),
        np.nan,
        dtype=float
    )

    theta = np.full(
        len(options),
        np.nan,
        dtype=float
    )

    if not valid.any():
        options["implied_volatility_est"] = iv
        options["theta"] = theta
        return options

    s = spot[valid]
    k = strike[valid]
    p = premium[valid]
    tv = t[valid]
    ce = is_ce[valid]

    low = np.full(
        len(s),
        MIN_IV,
        dtype=float
    )

    high = np.full(
        len(s),
        MAX_IV,
        dtype=float
    )

    # Vectorized bisection for implied volatility.
    for _ in range(IV_BISECTION_STEPS):

        mid = (
            low
            + high
        ) / 2.0

        model_price = _black_scholes_price(
            s,
            k,
            tv,
            mid,
            ce
        )

        price_too_high = (
            model_price > p
        )

        high = np.where(
            price_too_high,
            mid,
            high
        )

        low = np.where(
            price_too_high,
            low,
            mid
        )

    sigma = (
        low
        + high
    ) / 2.0

    sqrt_t = np.sqrt(tv)

    d1 = (
        np.log(s / k)
        + (
            RISK_FREE_RATE
            - DIVIDEND_YIELD
            + 0.5 * sigma ** 2
        ) * tv
    ) / (
        sigma
        * sqrt_t
    )

    d2 = (
        d1
        - sigma
        * sqrt_t
    )

    normal_pdf_d1 = (
        np.exp(
            -0.5 * d1 ** 2
        )
        / np.sqrt(2.0 * np.pi)
    )

    discounted_spot = (
        s
        * np.exp(
            -DIVIDEND_YIELD
            * tv
        )
    )

    discounted_strike = (
        k
        * np.exp(
            -RISK_FREE_RATE
            * tv
        )
    )

    common_theta = -(
        discounted_spot
        * normal_pdf_d1
        * sigma
    ) / (
        2.0
        * sqrt_t
    )

    call_theta_annual = (
        common_theta
        - RISK_FREE_RATE
        * discounted_strike
        * ndtr(d2)
        + DIVIDEND_YIELD
        * discounted_spot
        * ndtr(d1)
    )

    put_theta_annual = (
        common_theta
        + RISK_FREE_RATE
        * discounted_strike
        * ndtr(-d2)
        - DIVIDEND_YIELD
        * discounted_spot
        * ndtr(-d1)
    )

    theta_per_day = np.where(
        ce,
        call_theta_annual,
        put_theta_annual
    ) / 365.0

    iv[valid] = sigma
    theta[valid] = theta_per_day

    options["implied_volatility_est"] = iv
    options["theta"] = theta

    return options


def _add_candle_structure_features(g, prefix=""):
    """
    Add continuous candle-structure + compression/breakout features.

    These features intentionally describe the price geometry mathematically
    instead of adding dozens of named textbook candlestick-pattern flags.
    All rolling breakout levels are shifted so the current candle never
    sees future information.
    """
    g = g.copy()

    p = prefix

    candle_range = (g["high"] - g["low"]).clip(lower=0)
    body = g["close"] - g["open"]
    body_abs = body.abs()

    upper_wick = (
        g["high"]
        - pd.concat([g["open"], g["close"]], axis=1).max(axis=1)
    ).clip(lower=0)

    lower_wick = (
        pd.concat([g["open"], g["close"]], axis=1).min(axis=1)
        - g["low"]
    ).clip(lower=0)

    has_range = candle_range > 0

    # Candle geometry
    # Flat candles (high == low) are valid observations, not missing data.
    g[f"{p}body_to_range_ratio"] = np.where(
        has_range,
        body_abs / candle_range.where(has_range),
        0.0
    )

    g[f"{p}upper_wick_ratio"] = np.where(
        has_range,
        upper_wick / candle_range.where(has_range),
        0.0
    )

    g[f"{p}lower_wick_ratio"] = np.where(
        has_range,
        lower_wick / candle_range.where(has_range),
        0.0
    )

    g[f"{p}close_location_in_range"] = np.where(
        has_range,
        (g["close"] - g["low"]) / candle_range.where(has_range),
        0.5
    )

    # Relative candle size
    previous_range = candle_range.shift(1)
    prior_3_range_avg = candle_range.shift(1).rolling(3).mean()

    g[f"{p}range_vs_prev"] = np.where(
        previous_range > 0,
        candle_range / previous_range,
        0.0
    )

    g[f"{p}range_compression_3"] = np.where(
        prior_3_range_avg > 0,
        candle_range / prior_3_range_avg,
        0.0
    )

    # Inside bar = full current candle contained within previous candle.
    g[f"{p}inside_bar"] = (
        (g["high"] <= g["high"].shift(1))
        & (g["low"] >= g["low"].shift(1))
    ).astype(int)

    # NR4 = current range is <= each of the previous 3 ranges.
    prior_3_min_range = candle_range.shift(1).rolling(3).min()

    g[f"{p}nr4"] = (
        candle_range <= prior_3_min_range
    ).astype(int)

    # Short-term market structure.
    g[f"{p}higher_highs_3"] = (
        (g["high"] > g["high"].shift(1))
        & (g["high"].shift(1) > g["high"].shift(2))
    ).astype(int)

    g[f"{p}lower_lows_3"] = (
        (g["low"] < g["low"].shift(1))
        & (g["low"].shift(1) < g["low"].shift(2))
    ).astype(int)

    # Prior support/resistance; shifted to avoid leakage.
    prior_high_5 = g["high"].rolling(5).max().shift(1)
    prior_low_5 = g["low"].rolling(5).min().shift(1)

    prior_high_15 = g["high"].rolling(15).max().shift(1)
    prior_low_15 = g["low"].rolling(15).min().shift(1)

    g[f"{p}distance_to_resistance_15"] = _safe_div(
        prior_high_15 - g["close"],
        g["close"]
    )

    g[f"{p}distance_to_support_15"] = _safe_div(
        g["close"] - prior_low_15,
        g["close"]
    )

    # Breakout magnitude is continuous, unlike a simple 0/1 breakout flag.
    g[f"{p}breakout_strength_up_5"] = (
        ((g["close"] - prior_high_5) / g["close"])
        .clip(lower=0)
    )

    g[f"{p}breakout_strength_down_5"] = (
        ((prior_low_5 - g["close"]) / g["close"])
        .clip(lower=0)
    )

    breakout_up = g["close"] > prior_high_5
    breakout_down = g["close"] < prior_low_5

    # Two consecutive closes breaking their own prior 5-bar levels.
    g[f"{p}breakout_confirm_up_5"] = (
        breakout_up
        & breakout_up.shift(1, fill_value=False)
    ).astype(int)

    g[f"{p}breakout_confirm_down_5"] = (
        breakout_down
        & breakout_down.shift(1, fill_value=False)
    ).astype(int)

    # Intrabar level break followed by close back inside = failed breakout.
    g[f"{p}false_breakout_up_5"] = (
        (g["high"] > prior_high_5)
        & (g["close"] <= prior_high_5)
    ).astype(int)

    g[f"{p}false_breakout_down_5"] = (
        (g["low"] < prior_low_5)
        & (g["close"] >= prior_low_5)
    ).astype(int)

    return g


def build_features(df):
    """
    Build intraday NIFTY + option features.

    Existing feature names used by the trained model are preserved.
    New structural features are added as extra columns so this remains
    backward-compatible until the model is retrained with the new feature set.
    """
    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "instrument_type",
        "strike",
        "expiry"
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "build_features missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    df["date"] = df["timestamp"].dt.date

    nifty = df[
        df["instrument_type"] == "INDEX"
    ].copy()

    options = df[
        df["instrument_type"].isin(["CE", "PE"])
    ].copy()

    if nifty.empty:
        raise ValueError("No INDEX rows available for feature generation.")

    if options.empty:
        raise ValueError("No CE/PE rows available for feature generation.")

    # ==================================================
    # NIFTY FEATURES
    # ==================================================

    def nifty_day(g):
        g = g.sort_values("timestamp").copy()

        g["nifty_return_1m"] = g["close"].pct_change(1)
        g["nifty_return_3m"] = g["close"].pct_change(3)
        g["nifty_return_5m"] = g["close"].pct_change(5)

        g["nifty_range_pct"] = _safe_div(
            g["high"] - g["low"],
            g["close"]
        )

        g["nifty_rsi_14"] = rsi(
            g["close"],
            14
        )

        g["nifty_atr_14"] = atr(
            g,
            14
        )

        g["nifty_atr_pct"] = _safe_div(
            g["nifty_atr_14"],
            g["close"]
        )

        ema9 = g["close"].ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = g["close"].ewm(
            span=21,
            adjust=False
        ).mean()

        g["nifty_ema_spread_pct"] = _safe_div(
            ema9 - ema21,
            g["close"]
        )

        g["nifty_trend_up"] = (
            ema9 > ema21
        ).astype(int)

        g["nifty_trend_down"] = (
            ema9 < ema21
        ).astype(int)

        prior_high_15 = (
            g["high"]
            .rolling(15)
            .max()
            .shift(1)
        )

        prior_low_15 = (
            g["low"]
            .rolling(15)
            .min()
            .shift(1)
        )

        g["nifty_breakout_up_15"] = (
            g["close"] > prior_high_15
        ).astype(int)

        g["nifty_breakout_down_15"] = (
            g["close"] < prior_low_15
        ).astype(int)

        # Add structural/context features with a "nifty_" prefix.
        g = _add_candle_structure_features(
            g,
            prefix="nifty_"
        )

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

    # ==================================================
    # PREVIOUS-DAY PIVOT
    # ==================================================

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

    nifty["distance_from_pivot_pct"] = _safe_div(
        nifty["close"] - nifty["daily_pivot"],
        nifty["close"]
    )

    nifty_feature_columns = [
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
        "distance_from_pivot_pct",

        # Structural / chart-pattern features
        "nifty_body_to_range_ratio",
        "nifty_upper_wick_ratio",
        "nifty_lower_wick_ratio",
        "nifty_close_location_in_range",
        "nifty_range_vs_prev",
        "nifty_range_compression_3",
        "nifty_inside_bar",
        "nifty_nr4",
        "nifty_higher_highs_3",
        "nifty_lower_lows_3",
        "nifty_distance_to_resistance_15",
        "nifty_distance_to_support_15",
        "nifty_breakout_strength_up_5",
        "nifty_breakout_strength_down_5",
        "nifty_breakout_confirm_up_5",
        "nifty_breakout_confirm_down_5",
        "nifty_false_breakout_up_5",
        "nifty_false_breakout_down_5"
    ]

    nifty_features = nifty[
        nifty_feature_columns
    ].rename(
        columns={
            "close": "nifty_close"
        }
    )

    # ==================================================
    # OPTION FEATURES
    # ==================================================

    def option_day(g):
        g = g.sort_values("timestamp").copy()

        g["return_1m"] = g["close"].pct_change(1)
        g["return_3m"] = g["close"].pct_change(3)
        g["return_5m"] = g["close"].pct_change(5)

        g["range_pct"] = _safe_div(
            g["high"] - g["low"],
            g["close"]
        )

        g["body_pct"] = _safe_div(
            g["close"] - g["open"],
            g["open"]
        )

        candle_range = (
            g["high"] - g["low"]
        )

        # Signed body / range.
        # Flat candle => neutral strength 0.0 instead of NaN.
        g["candle_strength"] = np.where(
            candle_range > 0,
            (g["close"] - g["open"])
            / candle_range.where(candle_range > 0),
            0.0
        )

        # --------------------------
        # STRUCTURE / PATTERN GEOMETRY
        # --------------------------

        g = _add_candle_structure_features(g)

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

        g["atr_14"] = atr(
            g,
            14
        )

        g["atr_pct"] = _safe_div(
            g["atr_14"],
            g["close"]
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

        g["ema_spread_pct"] = _safe_div(
            ema9 - ema21,
            g["close"]
        )

        g["trend_up"] = (
            ema9 > ema21
        ).astype(int)

        # --------------------------
        # MACD
        # --------------------------

        ema12 = g["close"].ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = g["close"].ewm(
            span=26,
            adjust=False
        ).mean()

        g["macd_line"] = (
            ema12 - ema26
        )

        g["macd_signal"] = (
            g["macd_line"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        g["macd_histogram"] = (
            g["macd_line"]
            - g["macd_signal"]
        )

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

        g["volume_ratio_5"] = np.where(
            volume_avg_5 > 0,
            g["volume"] / volume_avg_5,
            np.where(
                volume_avg_5.isna(),
                np.nan,
                0.0
            )
        )

        g["volume_ratio_20"] = np.where(
            volume_avg_20 > 0,
            g["volume"] / volume_avg_20,
            np.where(
                volume_avg_20.isna(),
                np.nan,
                0.0
            )
        )

        # --------------------------
        # LIQUIDITY / VOLUME GAP (LVG)
        # --------------------------
        # With 1-minute OHLCV data we do not have a true volume-at-price
        # profile. This is therefore an explicit LVG proxy:
        #
        #   upward price gap from previous close
        #   + gap meaningful relative to ATR
        #   + relatively thin volume
        #
        # Positive values indicate an upward option-premium gap.

        previous_close = (
            g["close"].shift(1)
        )

        price_gap = (
            g["open"]
            - previous_close
        )

        g["lvg_score"] = _safe_div(
            price_gap,
            g["atr_14"]
        )

        g["lvg_detected"] = (
            (g["lvg_score"] >= 0.25)
            & (g["volume_ratio_20"] <= 0.85)
        ).astype(int)

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

        g["vwap"] = _safe_div(
            cumulative_value,
            cumulative_volume
        )

        g["distance_from_vwap_pct"] = _safe_div(
            g["close"] - g["vwap"],
            g["close"]
        )

        # --------------------------
        # 5-BAR BREAKOUT
        # Shifted -> no future leakage
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

        g["distance_from_high_5"] = _safe_div(
            g["close"] - prior_high_5,
            g["close"]
        )

        g["distance_from_low_5"] = _safe_div(
            g["close"] - prior_low_5,
            g["close"]
        )

        g["breakout_up_5"] = (
            g["close"] > prior_high_5
        ).astype(int)

        g["breakout_down_5"] = (
            g["close"] < prior_low_5
        ).astype(int)

        # --------------------------
        # FIDELITY CONFIRMATION LAYER
        # --------------------------
        # Uses the chart-structure ideas already represented above:
        # breakout, candle confirmation, resistance, false-breakout
        # rejection, range expansion and VWAP confirmation.

        fidelity_breakout = (
            g["breakout_up_5"] == 1
        )

        fidelity_no_false_breakout = (
            g["false_breakout_up_5"] == 0
        )

        fidelity_close_strength = (
            g["close_location_in_range"] >= 0.60
        )

        fidelity_body_strength = (
            g["body_to_range_ratio"] >= 0.45
        )

        fidelity_range_expansion = (
            g["range_vs_prev"] >= 1.0
        )

        fidelity_above_vwap = (
            g["distance_from_vwap_pct"] > 0
        )

        g["fidelity_score"] = (
            fidelity_breakout.astype(int)
            + fidelity_no_false_breakout.astype(int)
            + fidelity_close_strength.astype(int)
            + fidelity_body_strength.astype(int)
            + fidelity_range_expansion.astype(int)
            + fidelity_above_vwap.astype(int)
        )

        # Hard confirmation deliberately requires an actual breakout
        # and rejects a failed breakout.
        g["fidelity_confirmed"] = (
            fidelity_breakout
            & fidelity_no_false_breakout
            & (g["fidelity_score"] >= 4)
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

    # ==================================================
    # MERGE NIFTY CONTEXT
    # ==================================================

    options = options.merge(
        nifty_features,
        on="timestamp",
        how="left"
    )

    # ==================================================
    # MONEYNESS
    # ==================================================

    options["distance_from_strike"] = (
        options["nifty_close"]
        - options["strike"]
    )

    options["distance_from_strike_pct"] = _safe_div(
        options["distance_from_strike"],
        options["nifty_close"]
    )

    options["premium_pct_of_spot"] = _safe_div(
        options["close"],
        options["nifty_close"]
    )

    options["is_ce"] = (
        options["instrument_type"] == "CE"
    ).astype(int)

    # ==================================================
    # TIME TO EXPIRY / IMPLIED VOLATILITY / THETA
    # ==================================================

    options = _estimate_iv_and_theta(
        options
    )

    # ==================================================
    # CE vs PE — EXPIRY-AWARE, STALENESS-LIMITED ALIGNMENT
    # ==================================================
    #
    # Option candles can be absent for a minute when that contract does not
    # trade. For cross-option context, use the most recent known CE/PE value
    # only when it is no more than 3 minutes old.
    #
    # This does NOT synthesize the option's own OHLC candle. It only fills the
    # paired CE/PE context features used by the model.
    #
    # Pairing includes expiry so contracts from different expiries can never
    # be mixed together.

    PAIR_MAX_STALENESS = pd.Timedelta("3min")

    base = (
        options
        .reset_index()
        .rename(columns={"index": "_original_index"})
        .sort_values("timestamp")
    )

    ce_lookup = (
        options.loc[
            options["instrument_type"] == "CE",
            [
                "timestamp",
                "strike",
                "expiry",
                "close",
                "return_3m"
            ]
        ]
        .rename(
            columns={
                "close": "ce_close",
                "return_3m": "ce_return_3m"
            }
        )
        .sort_values("timestamp")
    )

    pe_lookup = (
        options.loc[
            options["instrument_type"] == "PE",
            [
                "timestamp",
                "strike",
                "expiry",
                "close",
                "return_3m"
            ]
        ]
        .rename(
            columns={
                "close": "pe_close",
                "return_3m": "pe_return_3m"
            }
        )
        .sort_values("timestamp")
    )

    options = pd.merge_asof(
        base,
        ce_lookup,
        on="timestamp",
        by=["strike", "expiry"],
        direction="backward",
        tolerance=PAIR_MAX_STALENESS,
        allow_exact_matches=True
    )

    options = pd.merge_asof(
        options.sort_values("timestamp"),
        pe_lookup,
        on="timestamp",
        by=["strike", "expiry"],
        direction="backward",
        tolerance=PAIR_MAX_STALENESS,
        allow_exact_matches=True
    )

    options = (
        options
        .sort_values("_original_index")
        .drop(columns="_original_index")
        .reset_index(drop=True)
    )

    # ==================================================
    # CE / PE PAIRING DIAGNOSTICS
    # ==================================================

    print("\n----------------------------")
    print("CE / PE PAIRING DIAGNOSTICS")
    print("----------------------------")
    print("Pair max staleness:", PAIR_MAX_STALENESS)
    print("Total option rows:", len(options))

    print("\nMissing paired close after 3-minute alignment:")
    print("ce_close missing:", options["ce_close"].isna().sum())
    print("pe_close missing:", options["pe_close"].isna().sum())

    print("\nMissing paired 3m return after 3-minute alignment:")
    print("ce_return_3m missing:", options["ce_return_3m"].isna().sum())
    print("pe_return_3m missing:", options["pe_return_3m"].isna().sum())

    print("\nPair availability:")
    print(
        pd.DataFrame({
            "has_ce": options["ce_close"].notna(),
            "has_pe": options["pe_close"].notna()
        }).value_counts()
    )

    print("\nMissing CE candles by strike:")
    print(
        options.loc[options["ce_close"].isna()]
        .groupby("strike")
        .size()
        .sort_values(ascending=False)
    )

    print("\nMissing PE candles by strike:")
    print(
        options.loc[options["pe_close"].isna()]
        .groupby("strike")
        .size()
        .sort_values(ascending=False)
    )

    print("----------------------------\n")

    options["ce_pe_ratio"] = _safe_div(
        options["ce_close"],
        options["pe_close"]
    )

    options["ce_minus_pe_return_3m"] = (
        options["ce_return_3m"]
        - options["pe_return_3m"]
    )

    # ==================================================
    # RULE-BASED SETUP SCORE — 10 CONDITIONS
    # ==================================================
    # Original 8 conditions are preserved.
    # 9  = LVG confirmation
    # 10 = MACD confirmation

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
        options["nifty_close"] > options["daily_pivot"],
        options["nifty_close"] < options["daily_pivot"]
    )

    relative_strength = np.where(
        options["is_ce"] == 1,
        options["ce_return_3m"] > options["pe_return_3m"],
        options["pe_return_3m"] > options["ce_return_3m"]
    )

    # Condition 9 — liquidity / volume gap.
    lvg_confirmation = (
        options["lvg_detected"] == 1
    )

    # Condition 10 — MACD momentum confirmation.
    macd_confirmation = (
        (options["macd_line"] > options["macd_signal"])
        & (options["macd_histogram"] > 0)
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
        + lvg_confirmation.astype(int)
        + macd_confirmation.astype(int)
    )

    # Align candidate setup with the new live 6/10 threshold.
    options["candidate_setup"] = (
        options["setup_score"] >= 6
    ).astype(int)

    return options
