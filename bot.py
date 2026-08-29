"""
===============================================================
CRYPTO INTRADAY MA STRATEGY V3
15M ENTRY + 1H TREND FILTER
Multi-Symbol | Backtest | Risk Management | LIVE | NTFY
===============================================================

LOGIC:

1H CLOSED CANDLE
       ↓
EMA 9 / EMA 21 / EMA 50
       ↓
1H TREND FILTER
       ↓
15M CLOSED CANDLE
       ↓
EMA 9 / EMA 21 / EMA 50
       ↓
CROSS OR PULLBACK
       ↓
ATR STOP
       ↓
RR 1:1.8
       ↓
1% RISK
       ↓
NTFY

IMPORTANT:
- Only CLOSED candles are used.
- Current 15M candle is ignored.
- Current 1H candle is ignored.
- LONG requires bullish 1H + bullish 15M setup.
- SHORT requires bearish 1H + bearish 15M setup.
- Backtest and LIVE use the same signal logic.
- Duplicate signals are prevented.
===============================================================
"""

import time
import warnings
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # SYMBOLS
    # --------------------------------------------------------

    "symbols": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "DOT/USDT",
        "CELR/USDT",
        "LINK/USDT",
        "CRV/USDT",
        "NEAR/USDT",
        "ZEC/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "FIL/USDT",
        "JASMY/USDT",
    ],

    # --------------------------------------------------------
    # TIMEFRAMES
    # --------------------------------------------------------

    "entry_timeframe": "15m",
    "trend_timeframe": "1h",

    "limit_15m": 500,
    "limit_1h": 200,

    # Scan every minute.
    # Closed-candle protection prevents duplicate signals.
    "scan_interval_seconds": 60,

    # --------------------------------------------------------
    # CAPITAL / RISK
    # --------------------------------------------------------

    "initial_capital": 10000.0,

    # 1% risk per trade
    "risk_per_trade": 0.01,

    # Risk : Reward
    "rr_ratio": 1.8,

    # Maximum NEW signals per symbol per UTC day
    "max_trades_per_day": 4,

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,

    # Price distance from EMA21
    "pullback_tolerance": 0.004,

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    "atr_period": 14,
    "atr_multiplier": 1.5,

    # --------------------------------------------------------
    # NTFY
    # --------------------------------------------------------

    "ntfy_enabled": True,

    "ntfy_server": "https://ntfy.sh",

    # YOUR NTFY TOPIC
    "ntfy_topic": "btc_ah7K9xQ2_signal",

    # --------------------------------------------------------
    # MODES
    # --------------------------------------------------------

    "run_backtest": True,
    "run_live": True,
}


# ============================================================
# GLOBAL STATE
# ============================================================

last_signal_candle = {}
daily_signal_count = {}
current_day = None


# ============================================================
# LOGGING
# ============================================================

def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[{now}] {message}"
    )


# ============================================================
# NTFY
# ============================================================

def send_ntfy(
    title,
    message,
    priority=4
):

    if not CONFIG["ntfy_enabled"]:
        return

    url = (
        f"{CONFIG['ntfy_server'].rstrip('/')}/"
        f"{CONFIG['ntfy_topic']}"
    )

    try:

        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": str(priority),
                "Tags": "chart_with_upwards_trend,moneybag",
            },
            timeout=10,
        )

        if response.ok:

            log(
                f"NTFY SENT → {title}"
            )

        else:

            log(
                f"NTFY ERROR → HTTP "
                f"{response.status_code}"
            )

    except Exception as e:

        log(
            f"NTFY ERROR → {e}"
        )


# ============================================================
# EMA
# ============================================================

def calculate_ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = (
        df["high"] -
        df["close"].shift(1)
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period
    ).mean()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["ema_9"] = calculate_ema(
        df["close"],
        CONFIG["ema_fast"]
    )

    df["ema_21"] = calculate_ema(
        df["close"],
        CONFIG["ema_slow"]
    )

    df["ema_50"] = calculate_ema(
        df["close"],
        CONFIG["ema_trend"]
    )

    df["atr"] = calculate_atr(
        df,
        CONFIG["atr_period"]
    )

    return df


# ============================================================
# 15M SIGNAL ENGINE
# ============================================================

def generate_15m_signals(df):

    df = df.copy()

    # --------------------------------------------------------
    # 15M TREND
    # --------------------------------------------------------

    uptrend = (
        (df["close"] > df["ema_50"])
        &
        (df["ema_9"] > df["ema_21"])
    )

    downtrend = (
        (df["close"] < df["ema_50"])
        &
        (df["ema_9"] < df["ema_21"])
    )

    # --------------------------------------------------------
    # EMA CROSS
    # --------------------------------------------------------

    bullish_cross = (
        (df["ema_9"] > df["ema_21"])
        &
        (
            df["ema_9"].shift(1)
            <=
            df["ema_21"].shift(1)
        )
    )

    bearish_cross = (
        (df["ema_9"] < df["ema_21"])
        &
        (
            df["ema_9"].shift(1)
            >=
            df["ema_21"].shift(1)
        )
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    distance_from_ema21 = (
        (df["close"] - df["ema_21"]).abs()
        /
        df["close"]
    )

    near_ema21 = (
        distance_from_ema21
        <=
        CONFIG["pullback_tolerance"]
    )

    bullish_candle = (
        df["close"] > df["open"]
    )

    bearish_candle = (
        df["close"] < df["open"]
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_cross = (
        bullish_cross
        &
        (df["close"] > df["ema_50"])
        &
        (df["close"] > df["ema_21"])
    )

    long_pullback = (
        uptrend
        &
        near_ema21
        &
        bullish_candle
        &
        (df["close"] > df["ema_21"])
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_cross = (
        bearish_cross
        &
        (df["close"] < df["ema_50"])
        &
        (df["close"] < df["ema_21"])
    )

    short_pullback = (
        downtrend
        &
        near_ema21
        &
        bearish_candle
        &
        (df["close"] < df["ema_21"])
    )

    df["signal"] = 0

    df.loc[
        long_cross |
        long_pullback,
        "signal"
    ] = 1

    df.loc[
        short_cross |
        short_pullback,
        "signal"
    ] = -1

    # ATR must exist
    df.loc[
        df["atr"].isna(),
        "signal"
    ] = 0

    return df


# ============================================================
# 1H TREND
# ============================================================

def get_1h_trend(df_1h):

    if df_1h is None:
        return 0

    if len(df_1h) < 60:
        return 0

    last = df_1h.iloc[-1]

    # --------------------------------------------------------
    # BULLISH 1H
    # --------------------------------------------------------

    if (
        last["close"] > last["ema_50"]
        and
        last["ema_9"] > last["ema_21"]
    ):

        return 1

    # --------------------------------------------------------
    # BEARISH 1H
    # --------------------------------------------------------

    if (
        last["close"] < last["ema_50"]
        and
        last["ema_9"] < last["ema_21"]
    ):

        return -1

    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------

    return 0


# ============================================================
# FINAL SIGNAL
# ============================================================

def apply_1h_filter(
    signal_15m,
    trend_1h
):

    # LONG only with bullish 1H
    if (
        signal_15m == 1
        and
        trend_1h == 1
    ):

        return 1

    # SHORT only with bearish 1H
    if (
        signal_15m == -1
        and
        trend_1h == -1
    ):

        return -1

    # Everything else = NO TRADE
    return 0


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    entry_price,
    atr,
    signal
):

    if (
        np.isnan(atr)
        or
        atr <= 0
    ):

        atr = (
            entry_price *
            0.01
        )

    stop_distance = (
        atr *
        CONFIG["atr_multiplier"]
    )

    if signal == 1:

        stop_loss = (
            entry_price -
            stop_distance
        )

        take_profit = (
            entry_price +
            stop_distance *
            CONFIG["rr_ratio"]
        )

        side = "LONG"

    else:

        stop_loss = (
            entry_price +
            stop_distance
        )

        take_profit = (
            entry_price -
            stop_distance *
            CONFIG["rr_ratio"]
        )

        side = "SHORT"

    return (
        side,
        stop_loss,
        take_profit,
        stop_distance
    )


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    capital,
    stop_distance
):

    if stop_distance <= 0:
        return 0

    risk_amount = (
        capital *
        CONFIG["risk_per_trade"]
    )

    return (
        risk_amount /
        stop_distance
    )


# ============================================================
# FETCH OHLCV
# ============================================================

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    },
})


def fetch_data(
    symbol,
    timeframe,
    limit
):

    try:

        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )

        df.set_index(
            "timestamp",
            inplace=True,
        )

        # ----------------------------------------------------
        # REMOVE CURRENTLY FORMING CANDLE
        # ----------------------------------------------------

        if len(df) > 1:

            df = df.iloc[:-1].copy()

        return df

    except Exception as e:

        log(
            f"{symbol} {timeframe} DATA ERROR → {e}"
        )

        return None


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df_15m,
    df_1h,
    symbol
):

    capital = (
        CONFIG["initial_capital"]
    )

    position = 0

    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0

    entry_time = None

    trades = []

    daily_trades = {}

    # --------------------------------------------------------
    # 1H DATA IS ALREADY CLOSED.
    #
    # For every 15M candle, use the latest 1H candle
    # that was already CLOSED before that 15M candle.
    # --------------------------------------------------------

    for i in range(
        1,
        len(df_15m)
    ):

        row = df_15m.iloc[i]
        prev = df_15m.iloc[i - 1]

        current_time = (
            df_15m.index[i]
        )

        # ====================================================
        # FIND LAST CLOSED 1H CANDLE
        # ====================================================

        h1_available = df_1h[
            df_1h.index
            <=
            prev.name
        ]

        if len(h1_available) < 60:
            continue

        h1_row = h1_available.iloc[-1]

        # ====================================================
        # 1H TREND
        # ====================================================

        if (
            h1_row["close"] >
            h1_row["ema_50"]
            and
            h1_row["ema_9"] >
            h1_row["ema_21"]
        ):

            trend_1h = 1

        elif (
            h1_row["close"] <
            h1_row["ema_50"]
            and
            h1_row["ema_9"] <
            h1_row["ema_21"]
        ):

            trend_1h = -1

        else:

            trend_1h = 0

        # ====================================================
        # DAILY COUNTER
        # ====================================================

        date_key = (
            current_time.date()
        )

        if date_key not in daily_trades:

            daily_trades[
                date_key
            ] = 0

        # ====================================================
        # MANAGE OPEN POSITION
        # ====================================================

        if position != 0:

            hit_sl = False
            hit_tp = False

            if position == 1:

                if row["low"] <= stop_loss:

                    hit_sl = True
                    exit_price = stop_loss

                elif row["high"] >= take_profit:

                    hit_tp = True
                    exit_price = take_profit

            else:

                if row["high"] >= stop_loss:

                    hit_sl = True
                    exit_price = stop_loss

                elif row["low"] <= take_profit:

                    hit_tp = True
                    exit_price = take_profit

            if hit_sl or hit_tp:

                pnl = (
                    exit_price -
                    entry_price
                ) * position_size * position

                capital += pnl

                trades.append({
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": (
                        "LONG"
                        if position == 1
                        else "SHORT"
                    ),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "pnl": pnl,
                    "result": (
                        "TP"
                        if hit_tp
                        else "SL"
                    ),
                })

                position = 0

        # ====================================================
        # NEW ENTRY
        # ====================================================

        if (
            position == 0
            and
            daily_trades[date_key]
            <
            CONFIG["max_trades_per_day"]
        ):

            signal_15m = int(
                prev["signal"]
            )

            final_signal = (
                apply_1h_filter(
                    signal_15m,
                    trend_1h
                )
            )

            if final_signal == 0:
                continue

            entry_price = float(
                row["open"]
            )

            atr = float(
                prev["atr"]
            )

            (
                side,
                stop_loss,
                take_profit,
                stop_distance
            ) = calculate_trade_levels(
                entry_price,
                atr,
                final_signal
            )

            position_size = (
                calculate_position_size(
                    capital,
                    stop_distance
                )
            )

            if position_size <= 0:
                continue

            position = final_signal

            entry_time = current_time

            daily_trades[
                date_key
            ] += 1

    # ========================================================
    # CLOSE LAST POSITION
    # ========================================================

    if position != 0:

        exit_price = float(
            df_15m.iloc[-1]["close"]
        )

        pnl = (
            exit_price -
            entry_price
        ) * position_size * position

        capital += pnl

        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "exit_time": df_15m.index[-1],
            "side": (
                "LONG"
                if position == 1
                else "SHORT"
            ),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "pnl": pnl,
            "result": "EOD",
        })

    # ========================================================
    # RESULTS
    # ========================================================

    trades_df = pd.DataFrame(
        trades
    )

    if trades_df.empty:

        return {
            "symbol": symbol,
            "error": "No trades"
        }

    total_return = (
        (
            capital -
            CONFIG["initial_capital"]
        )
        /
        CONFIG["initial_capital"]
    ) * 100

    wins = (
        trades_df["pnl"] > 0
    ).sum()

    losses = (
        trades_df["pnl"] < 0
    ).sum()

    winrate = (
        wins /
        len(trades_df)
    ) * 100

    gross_profit = (
        trades_df.loc[
            trades_df["pnl"] > 0,
            "pnl"
        ].sum()
    )

    gross_loss = abs(
        trades_df.loc[
            trades_df["pnl"] < 0,
            "pnl"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = np.inf

    return {
        "symbol": symbol,
        "final_capital": round(
            capital,
            2
        ),
        "total_return_pct": round(
            total_return,
            2
        ),
        "total_trades": len(
            trades_df
        ),
        "wins": int(wins),
        "losses": int(losses),
        "winrate_pct": round(
            winrate,
            2
        ),
        "profit_factor": (
            round(
                profit_factor,
                2
            )
            if np.isfinite(
                profit_factor
            )
            else 999
        ),
        "trades": trades_df,
    }


# ============================================================
# BACKTEST ALL SYMBOLS
# ============================================================

def run_all_backtests():

    results = []

    log(
        "Starting 15M + 1H FILTER backtest..."
    )

    for symbol in CONFIG["symbols"]:

        log(
            f"BACKTEST → {symbol}"
        )

        df_15m = fetch_data(
            symbol,
            CONFIG["entry_timeframe"],
            CONFIG["limit_15m"]
        )

        df_1h = fetch_data(
            symbol,
            CONFIG["trend_timeframe"],
            CONFIG["limit_1h"]
        )

        if (
            df_15m is None
            or
            df_1h is None
        ):

            continue

        if (
            len(df_15m) < 100
            or
            len(df_1h) < 60
        ):

            log(
                f"{symbol} → insufficient data"
            )

            continue

        df_15m = add_indicators(
            df_15m
        )

        df_15m = generate_15m_signals(
            df_15m
        )

        df_1h = add_indicators(
            df_1h
        )

        result = run_backtest(
            df_15m,
            df_1h,
            symbol
        )

        results.append(result)

        if "error" not in result:

            log(
                f"{symbol} → "
                f"Return: "
                f"{result['total_return_pct']}% | "
                f"Trades: "
                f"{result['total_trades']} | "
                f"Winrate: "
                f"{result['winrate_pct']}% | "
                f"PF: "
                f"{result['profit_factor']}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 95)
    print(
        "BACKTEST SUMMARY — 15M + 1H FILTER"
    )
    print("=" * 95)

    print(
        f"{'SYMBOL':<12}"
        f"{'RETURN %':<12}"
        f"{'TRADES':<10}"
        f"{'WINRATE':<12}"
        f"{'PF':<10}"
        f"{'FINAL CAPITAL':<15}"
    )

    print("-" * 95)

    for result in results:

        if "error" in result:

            print(
                f"{result['symbol']:<12}"
                f"NO TRADES"
            )

            continue

        print(
            f"{result['symbol']:<12}"
            f"{result['total_return_pct']:<12}"
            f"{result['total_trades']:<10}"
            f"{result['winrate_pct']:<12}"
            f"{result['profit_factor']:<10}"
            f"{result['final_capital']:<15}"
        )

    print("=" * 95)

    # ========================================================
    # SAVE TRADES
    # ========================================================

    all_trades = []

    for result in results:

        if "trades" in result:

            all_trades.append(
                result["trades"]
            )

    if all_trades:

        combined = pd.concat(
            all_trades,
            ignore_index=True
        )

        combined.to_csv(
            "all_backtest_trades.csv",
            index=False
        )

        log(
            "Saved → all_backtest_trades.csv"
        )


# ============================================================
# LIVE SIGNAL
# ============================================================

def process_live_symbol(
    symbol,
    df_15m,
    df_1h
):

    global current_day

    if (
        len(df_15m) < 60
        or
        len(df_1h) < 60
    ):

        return

    # ========================================================
    # UTC DAY RESET
    # ========================================================

    today = datetime.now(
        timezone.utc
    ).date()

    if current_day != today:

        daily_signal_count.clear()
        last_signal_candle.clear()

        current_day = today

        log(
            "Daily counters reset."
        )

    # ========================================================
    # LAST CLOSED 15M CANDLE
    # ========================================================

    last_15m = df_15m.iloc[-1]

    candle_time = df_15m.index[-1]

    # ========================================================
    # DUPLICATE PROTECTION
    # ========================================================

    if (
        symbol in last_signal_candle
        and
        last_signal_candle[symbol]
        ==
        candle_time
    ):

        return

    # Mark this closed candle as processed
    last_signal_candle[
        symbol
    ] = candle_time

    # ========================================================
    # DAILY LIMIT
    # ========================================================

    count = daily_signal_count.get(
        symbol,
        0
    )

    if (
        count >=
        CONFIG["max_trades_per_day"]
    ):

        return

    # ========================================================
    # 15M SIGNAL
    # ========================================================

    signal_15m = int(
        last_15m["signal"]
    )

    if signal_15m == 0:

        return

    # ========================================================
    # LAST CLOSED 1H CANDLE
    # ========================================================

    h1_available = df_1h[
        df_1h.index
        <=
        candle_time
    ]

    if len(h1_available) < 60:

        return

    last_1h = h1_available.iloc[-1]

    # ========================================================
    # 1H TREND
    # ========================================================

    if (
        last_1h["close"] >
        last_1h["ema_50"]
        and
        last_1h["ema_9"] >
        last_1h["ema_21"]
    ):

        trend_1h = 1

    elif (
        last_1h["close"] <
        last_1h["ema_50"]
        and
        last_1h["ema_9"] <
        last_1h["ema_21"]
    ):

        trend_1h = -1

    else:

        trend_1h = 0

    # ========================================================
    # APPLY 1H FILTER
    # ========================================================

    final_signal = apply_1h_filter(
        signal_15m,
        trend_1h
    )

    if final_signal == 0:

        return

    # ========================================================
    # ENTRY PRICE
    # ========================================================

    entry_price = float(
        last_15m["close"]
    )

    atr = float(
        last_15m["atr"]
    )

    (
        side,
        stop_loss,
        take_profit,
        stop_distance
    ) = calculate_trade_levels(
        entry_price,
        atr,
        final_signal
    )

    # ========================================================
    # RISK
    # ========================================================

    risk_amount = (
        CONFIG["initial_capital"]
        *
        CONFIG["risk_per_trade"]
    )

    position_size = (
        calculate_position_size(
            CONFIG["initial_capital"],
            stop_distance
        )
    )

    # ========================================================
    # PRICE FORMAT
    # ========================================================

    def fmt_price(value):

        if value >= 1000:

            return f"{value:.2f}"

        if value >= 1:

            return f"{value:.4f}"

        return f"{value:.8f}"

    # ========================================================
    # MESSAGE
    # ========================================================

    emoji = (
        "🟢"
        if final_signal == 1
        else
        "🔴"
    )

    title = (
        f"{emoji} "
        f"{side} | {symbol}"
    )

    message = (
        f"━━━━━━━━━━━━━━━━━━\n"
        f"AH1 MA STRATEGY V3\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"

        f"Symbol: {symbol}\n"
        f"Side: {side}\n"
        f"Entry TF: 15M\n"
        f"Trend TF: 1H\n\n"

        f"Entry: {fmt_price(entry_price)}\n"
        f"Stop Loss: {fmt_price(stop_loss)}\n"
        f"Take Profit: {fmt_price(take_profit)}\n\n"

        f"RR: 1:{CONFIG['rr_ratio']}\n"
        f"Risk: {CONFIG['risk_per_trade'] * 100:.1f}%\n"
        f"Risk Amount: ${risk_amount:.2f}\n"
        f"Position Size: {position_size:.6f}\n\n"

        f"── 15M ──\n"
        f"EMA9: {fmt_price(last_15m['ema_9'])}\n"
        f"EMA21: {fmt_price(last_15m['ema_21'])}\n"
        f"EMA50: {fmt_price(last_15m['ema_50'])}\n"
        f"ATR: {fmt_price(last_15m['atr'])}\n\n"

        f"── 1H FILTER ──\n"
        f"EMA9: {fmt_price(last_1h['ema_9'])}\n"
        f"EMA21: {fmt_price(last_1h['ema_21'])}\n"
        f"EMA50: {fmt_price(last_1h['ema_50'])}\n"
        f"Trend: "
        f"{'BULLISH' if trend_1h == 1 else 'BEARISH'}\n\n"

        f"Closed Candle:\n"
        f"{candle_time.strftime('%Y-%m-%d %H:%M UTC')}\n"

        f"\n━━━━━━━━━━━━━━━━━━"
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print("=" * 65)
    print(title)
    print("=" * 65)
    print(message)
    print("=" * 65)

    # ========================================================
    # NTFY
    # ========================================================

    send_ntfy(
        title,
        message,
        priority=4
    )

    daily_signal_count[
        symbol
    ] = count + 1


# ============================================================
# LIVE LOOP
# ============================================================

def live_loop():

    global current_day

    log("=" * 65)
    log(
        "CRYPTO MA STRATEGY V3 STARTED"
    )
    log("=" * 65)

    log(
        "Entry timeframe → 15M"
    )

    log(
        "Trend timeframe → 1H"
    )

    log(
        f"Symbols → "
        f"{len(CONFIG['symbols'])}"
    )

    log(
        f"Risk → "
        f"{CONFIG['risk_per_trade'] * 100}%"
    )

    log(
        f"RR → "
        f"1:{CONFIG['rr_ratio']}"
    )

    log(
        "15M current candle → IGNORED"
    )

    log(
        "1H current candle → IGNORED"
    )

    log(
        "LONG → 15M bullish + 1H bullish"
    )

    log(
        "SHORT → 15M bearish + 1H bearish"
    )

    # ========================================================
    # STARTUP NTFY
    # ========================================================

    if CONFIG["ntfy_enabled"]:

        send_ntfy(
            "AH1 MA STRATEGY V3",
            (
                "Bot started successfully.\n"
                "Entry: 15M\n"
                "Trend Filter: 1H\n"
                f"Symbols: {len(CONFIG['symbols'])}\n"
                "Mode: LIVE"
            ),
            priority=3
        )

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        cycle_start = time.time()

        log(
            "Starting market scan..."
        )

        for symbol in CONFIG["symbols"]:

            try:

                # ------------------------------------------------
                # 15M
                # ------------------------------------------------

                df_15m = fetch_data(
                    symbol,
                    CONFIG["entry_timeframe"],
                    CONFIG["limit_15m"]
                )

                # ------------------------------------------------
                # 1H
                # ------------------------------------------------

                df_1h = fetch_data(
                    symbol,
                    CONFIG["trend_timeframe"],
                    CONFIG["limit_1h"]
                )

                if (
                    df_15m is None
                    or
                    df_1h is None
                ):

                    continue

                if (
                    len(df_15m) < 60
                    or
                    len(df_1h) < 60
                ):

                    continue

                # ------------------------------------------------
                # INDICATORS
                # ------------------------------------------------

                df_15m = add_indicators(
                    df_15m
                )

                df_15m = generate_15m_signals(
                    df_15m
                )

                df_1h = add_indicators(
                    df_1h
                )

                # ------------------------------------------------
                # PROCESS
                # ------------------------------------------------

                process_live_symbol(
                    symbol,
                    df_15m,
                    df_1h
                )

            except Exception as e:

                log(
                    f"{symbol} LIVE ERROR → {e}"
                )

        # ====================================================
        # WAIT
        # ====================================================

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_time = max(
            5,
            CONFIG[
                "scan_interval_seconds"
            ] -
            elapsed
        )

        log(
            f"Scan completed. "
            f"Next scan in "
            f"{sleep_time:.0f}s"
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        # ----------------------------------------------------
        # BACKTEST
        # ----------------------------------------------------

        if CONFIG["run_backtest"]:

            run_all_backtests()

        # ----------------------------------------------------
        # LIVE
        # ----------------------------------------------------

        if CONFIG["run_live"]:

            live_loop()

    except KeyboardInterrupt:

        log(
            "Bot stopped manually."
        )

    except Exception as e:

        log(
            f"FATAL ERROR → {e}"
        )
