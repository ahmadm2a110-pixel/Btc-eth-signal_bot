import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd
import pandas_ta as ta


# ============================================================
# AH1 ELITE v2
# ============================================================
#
# 1H = MAIN DECISION ENGINE
# 4H = CONFIRMATION SCORE ONLY
#
# After signal:
#   -> monitor every closed 1H candle
#   -> maximum initial monitoring: 6 candles
#   -> ACTIVE / STRENGTHENED / WEAKENED
#   -> INVALIDATED / CANCELLED
#   -> TP / SL
#
# 6 candles DOES NOT automatically cancel the setup.
# If still valid after 6 candles -> STILL ACTIVE
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

EXCHANGE_ID = "binance"

MAIN_TIMEFRAME = "1h"
CONFIRM_TIMEFRAME = "4h"

SCAN_INTERVAL = 60

OHLCV_LIMIT = 300

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "SUI/USDT",
]


# ============================================================
# NTFY
# ============================================================

NTFY_SERVER = "https://ntfy.sh"

NTFY_TOPIC = "btc_ah7K9xQ2_signal"


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_LENGTH = 14
ADX_LENGTH = 14
ATR_LENGTH = 14

BREAKOUT_LOOKBACK = 20
VOLUME_LOOKBACK = 20

MIN_SCORE = 70
A_PLUS_SCORE = 90

SHORT_BONUS = 3

FOUR_H_CONFIRMATION = 5

WATCH_MAX_CANDLES = 6


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AH1_ELITE")


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    },
})


# ============================================================
# STATE
# ============================================================

# Last signal generated for each symbol.
last_signal_key = {}


# Active watchers:
#
# symbol -> {
#     signal,
#     quality,
#     score,
#     entry,
#     stop,
#     tp,
#     signal_candle,
#     candles_watched,
#     reasons,
# }
#
active_setups = {}


# Background watcher tasks.
watcher_tasks = {}


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(title, message):

    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"

    headers = {
        "Title": title,
        "Priority": "high",
        "Tags": "chart_with_upwards_trend",
    }

    try:

        async with httpx.AsyncClient(timeout=15) as client:

            response = await client.post(
                url,
                content=message.encode("utf-8"),
                headers=headers,
            )

            if response.status_code >= 400:

                logger.error(
                    "NTFY failed: %s %s",
                    response.status_code,
                    response.text,
                )

            else:

                logger.info(
                    "NTFY sent: %s",
                    title,
                )

    except Exception as e:

        logger.exception(
            "NTFY error: %s",
            e,
        )


# ============================================================
# FETCH DATA
# ============================================================

async def fetch_dataframe(symbol, timeframe):

    try:

        candles = await exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=OHLCV_LIMIT,
        )

        if not candles:
            return None

        df = pd.DataFrame(
            candles,
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

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        return df

    except Exception as e:

        logger.error(
            "Fetch error %s %s: %s",
            symbol,
            timeframe,
            e,
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = ta.ema(
        df["close"],
        length=EMA_FAST,
    )

    df["ema50"] = ta.ema(
        df["close"],
        length=EMA_MID,
    )

    df["ema200"] = ta.ema(
        df["close"],
        length=EMA_SLOW,
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = ta.rsi(
        df["close"],
        length=RSI_LENGTH,
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx = ta.adx(
        df["high"],
        df["low"],
        df["close"],
        length=ADX_LENGTH,
    )

    df["adx"] = adx[
        f"ADX_{ADX_LENGTH}"
    ]

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = ta.macd(
        df["close"],
        fast=12,
        slow=26,
        signal=9,
    )

    df["macd"] = macd[
        "MACD_12_26_9"
    ]

    df["macd_signal"] = macd[
        "MACDs_12_26_9"
    ]

    df["macd_hist"] = macd[
        "MACDh_12_26_9"
    ]

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = ta.atr(
        df["high"],
        df["low"],
        df["close"],
        length=ATR_LENGTH,
    )

    df["atr_pct"] = (
        df["atr"] /
        df["close"]
    ) * 100

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_ma"] = (
        df["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma"]
    )

    # --------------------------------------------------------
    # Previous HIGH / LOW
    #
    # shift(1) is VERY IMPORTANT.
    #
    # The current candle must NOT define its own
    # breakout level.
    # --------------------------------------------------------

    df["previous_high"] = (
        df["high"]
        .shift(1)
        .rolling(BREAKOUT_LOOKBACK)
        .max()
    )

    df["previous_low"] = (
        df["low"]
        .shift(1)
        .rolling(BREAKOUT_LOOKBACK)
        .min()
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["close_position"] = (
        (
            df["close"] -
            df["low"]
        )
        /
        df["range"].replace(0, pd.NA)
    )

    return df


# ============================================================
# 4H CONFIRMATION
# ============================================================

def get_4h_confirmation(df4):

    # Last CLOSED 4H candle.
    row = df4.iloc[-2]

    close = row["close"]
    ema50 = row["ema50"]
    ema200 = row["ema200"]
    adx = row["adx"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema50,
            ema200,
            adx,
        ]
    ):
        return 0, "4H_NEUTRAL"

    if (
        close > ema200
        and ema50 > ema200
        and adx >= 18
    ):

        return (
            FOUR_H_CONFIRMATION,
            "4H_BULLISH",
        )

    if (
        close < ema200
        and ema50 < ema200
        and adx >= 18
    ):

        return (
            -FOUR_H_CONFIRMATION,
            "4H_BEARISH",
        )

    return 0, "4H_NEUTRAL"


# ============================================================
# CANDLE PATTERNS
# ============================================================

def bullish_pinbar(row):

    if row["body"] <= 0:
        return False

    return (
        row["lower_wick"]
        >= row["body"] * 1.5
        and
        row["close_position"]
        >= 0.60
    )


def bearish_pinbar(row):

    if row["body"] <= 0:
        return False

    return (
        row["upper_wick"]
        >= row["body"] * 1.5
        and
        row["close_position"]
        <= 0.40
    )


# ============================================================
# H1 ENGINE
# ============================================================

def analyze_1h(df1, df4):

    # Last CLOSED H1 candle.
    row = df1.iloc[-2]

    prev = df1.iloc[-3]

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # H1 TREND
    # ========================================================

    long_trend = (
        row["close"] > row["ema20"]
        and
        row["ema20"] > row["ema50"]
        and
        row["ema50"] > row["ema200"]
    )

    short_trend = (
        row["close"] < row["ema20"]
        and
        row["ema20"] < row["ema50"]
        and
        row["ema50"] < row["ema200"]
    )

    if long_trend:

        long_score += 20

        long_reasons.append(
            "h1_trend_bullish"
        )

    if short_trend:

        short_score += 20

        short_reasons.append(
            "h1_trend_bearish"
        )

    # ========================================================
    # H1 BREAKOUT
    # ========================================================

    long_breakout = (
        row["close"] >
        row["previous_high"]
        and
        prev["close"] <=
        prev["previous_high"]
    )

    short_breakout = (
        row["close"] <
        row["previous_low"]
        and
        prev["close"] >=
        prev["previous_low"]
    )

    if long_breakout:

        long_score += 20

        long_reasons.append(
            "h1_breakout"
        )

    if short_breakout:

        short_score += 20

        short_reasons.append(
            "h1_breakdown"
        )

    # ========================================================
    # RETEST
    # ========================================================

    recent_long_breakout = (
        df1["close"].iloc[-7:-2]
        >
        df1["previous_high"].iloc[-7:-2]
    ).any()

    recent_short_breakout = (
        df1["close"].iloc[-7:-2]
        <
        df1["previous_low"].iloc[-7:-2]
    ).any()

    long_retest = (
        recent_long_breakout
        and
        row["low"] <=
        row["previous_high"] * 1.008
        and
        row["close"] >
        row["previous_high"]
    )

    short_retest = (
        recent_short_breakout
        and
        row["high"] >=
        row["previous_low"] * 0.992
        and
        row["close"] <
        row["previous_low"]
    )

    if long_retest:

        long_score += 15

        long_reasons.append(
            "successful_retest"
        )

    if short_retest:

        short_score += 15

        short_reasons.append(
            "successful_retest"
        )

    # ========================================================
    # MOMENTUM
    # ========================================================

    bullish_momentum = (
        52 <= row["rsi"] <= 72
        and
        row["macd"] >
        row["macd_signal"]
        and
        row["adx"] >= 18
    )

    bearish_momentum = (
        28 <= row["rsi"] <= 48
        and
        row["macd"] <
        row["macd_signal"]
        and
        row["adx"] >= 18
    )

    if bullish_momentum:

        long_score += 15

        long_reasons.append(
            "momentum_bullish"
        )

    if bearish_momentum:

        short_score += 15

        short_reasons.append(
            "momentum_bearish"
        )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_confirmation = (
        row["volume_ratio"] >= 1.10
    )

    if volume_confirmation:

        if row["close"] > row["open"]:

            long_score += 10

            long_reasons.append(
                "volume_confirmation"
            )

        elif row["close"] < row["open"]:

            short_score += 10

            short_reasons.append(
                "volume_confirmation"
            )

    # ========================================================
    # CANDLE
    # ========================================================

    if bullish_pinbar(row):

        long_score += 5

        long_reasons.append(
            "bullish_pinbar"
        )

    if bearish_pinbar(row):

        short_score += 5

        short_reasons.append(
            "bearish_pinbar"
        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    volatility_ok = (
        0.20 <=
        row["atr_pct"] <=
        4.5
    )

    if not volatility_ok:

        return {
            "signal": "NO_TRADE",
            "quality": "NO_TRADE",
            "score": 0,
            "long_score": 0,
            "short_score": 0,
            "reasons": [],
            "confirmation": "4H_NEUTRAL",
            "row": row,
            "core_long": False,
            "core_short": False,
        }

    # ========================================================
    # 4H = ONLY CONFIRMATION
    # ========================================================

    confirmation_points, confirmation_name = (
        get_4h_confirmation(df4)
    )

    #
    # 4H contributes ONLY 5 points.
    #

    if confirmation_name == "4H_BULLISH":

        long_score += 5
        short_score -= 5

    elif confirmation_name == "4H_BEARISH":

        short_score += 5
        long_score -= 5

    # Neutral = 0
    #
    # 4H NEVER creates a signal.
    # 4H NEVER replaces the H1 setup.
    #

    # ========================================================
    # SHORT PREFERENCE
    # ========================================================

    if short_score >= MIN_SCORE:

        short_score += SHORT_BONUS

    # ========================================================
    # CORE SETUP
    # ========================================================

    core_long = (
        long_trend
        and
        (
            long_breakout
            or
            long_retest
        )
    )

    core_short = (
        short_trend
        and
        (
            short_breakout
            or
            short_retest
        )
    )

    #
    # If the core H1 setup isn't there,
    # score means NOTHING.
    #

    if not core_long:

        long_score = 0

    if not core_short:

        short_score = 0

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if (
        long_score < MIN_SCORE
        and
        short_score < MIN_SCORE
    ):

        return {
            "signal": "NO_TRADE",
            "quality": "NO_TRADE",
            "score": max(
                long_score,
                short_score,
            ),
            "long_score": long_score,
            "short_score": short_score,
            "reasons": [],
            "confirmation": confirmation_name,
            "row": row,
            "core_long": core_long,
            "core_short": core_short,
        }

    if short_score > long_score:

        signal = "SHORT"

        final_score = short_score

        reasons = short_reasons

        core = core_short

    else:

        signal = "LONG"

        final_score = long_score

        reasons = long_reasons

        core = core_long

    if not core:

        signal = "NO_TRADE"

    # ========================================================
    # QUALITY
    # ========================================================

    if final_score >= A_PLUS_SCORE:

        quality = "A+"

    elif final_score >= 82:

        quality = "A"

    elif final_score >= MIN_SCORE:

        quality = "B"

    else:

        quality = "NO_TRADE"

    if signal == "NO_TRADE":

        quality = "NO_TRADE"

    return {
        "signal": signal,
        "quality": quality,
        "score": final_score,
        "long_score": long_score,
        "short_score": short_score,
        "reasons": reasons,
        "confirmation": confirmation_name,
        "row": row,
        "core_long": core_long,
        "core_short": core_short,
    }


# ============================================================
# LEVELS
# ============================================================

def calculate_levels(result):

    row = result["row"]

    entry = float(row["close"])

    atr = float(row["atr"])

    signal = result["signal"]

    if signal == "LONG":

        structure_stop = float(
            row["low"]
        )

        atr_stop = (
            entry -
            atr * 1.8
        )

        stop = min(
            structure_stop,
            atr_stop,
        )

        risk = entry - stop

        tp = (
            entry +
            risk * 2.0
        )

    elif signal == "SHORT":

        structure_stop = float(
            row["high"]
        )

        atr_stop = (
            entry +
            atr * 1.8
        )

        stop = max(
            structure_stop,
            atr_stop,
        )

        risk = stop - entry

        tp = (
            entry -
            risk * 2.0
        )

    else:

        return None

    if risk <= 0:

        return None

    return {
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "risk": risk,
        "rr": 2.0,
    }


# ============================================================
# INITIAL SIGNAL MESSAGE
# ============================================================

def format_signal(
    symbol,
    result,
    levels,
):

    signal = result["signal"]

    if signal == "SHORT":

        emoji = "🔴"

    else:

        emoji = "🟢"

    reasons = "\n".join(
        f"- {x}"
        for x in result["reasons"]
    )

    return f"""
{emoji} {symbol} — {signal}

Quality: {result["quality"]}
Score: {result["score"]}/100

Trigger:
H1_BREAKOUT_OR_RETEST

Entry:
{levels["entry"]:.6f}

Stop Loss:
{levels["stop"]:.6f}

Take Profit:
{levels["tp"]:.6f}

Risk/Reward:
1:{levels["rr"]:.2f}

4H Confirmation:
{result["confirmation"]}

H1 Confluence:
{reasons}

H1 Long Score:
{result["long_score"]}

H1 Short Score:
{result["short_score"]}

Signal Candle:
{result["row"]["timestamp"]}

AH1 ELITE v2
""".strip()


# ============================================================
# WATCHER MESSAGE
# ============================================================

def format_watch_update(
    symbol,
    setup,
    current,
    result,
    status,
    previous_score,
):

    signal = setup["signal"]

    entry = setup["entry"]

    current_price = float(
        current["close"]
    )

    if signal == "LONG":

        pnl_pct = (
            (current_price - entry)
            / entry
        ) * 100

    else:

        pnl_pct = (
            (entry - current_price)
            / entry
        ) * 100

    score = result["score"]

    if score > previous_score:

        score_change = "⬆️ STRENGTHENED"

    elif score < previous_score:

        score_change = "⬇️ WEAKENED"

    else:

        score_change = "➡️ UNCHANGED"

    return f"""
📊 {symbol} — {signal} UPDATE

Status:
{status}

Price:
{current_price:.6f}

Entry:
{entry:.6f}

Move from Entry:
{pnl_pct:+.2f}%

Score:
{previous_score} → {score}/100

{score_change}

H1 Structure:
{"VALID" if result["core_long"] or result["core_short"] else "INVALID"}

4H Confirmation:
{result["confirmation"]}

Watched Candles:
{setup["candles_watched"]}/{WATCH_MAX_CANDLES}

AH1 ELITE WATCHER
""".strip()


# ============================================================
# WATCHER
# ============================================================

async def watch_setup(
    symbol,
    setup,
):

    logger.info(
        "%s | WATCHER STARTED | %s",
        symbol,
        setup["signal"],
    )

    previous_score = setup["score"]

    last_watched_candle = setup[
        "signal_candle"
    ]

    try:

        while True:

            await asyncio.sleep(
                60
            )

            df1 = await fetch_dataframe(
                symbol,
                MAIN_TIMEFRAME,
            )

            df4 = await fetch_dataframe(
                symbol,
                CONFIRM_TIMEFRAME,
            )

            if df1 is None or df4 is None:

                continue

            if len(df1) < 220:

                continue

            df1 = add_indicators(
                df1
            )

            df4 = add_indicators(
                df4
            )

            # ------------------------------------------------
            # Last CLOSED H1 candle
            # ------------------------------------------------

            current = df1.iloc[-2]

            candle_time = current[
                "timestamp"
            ]

            #
            # We only process each H1 candle ONCE.
            #

            if candle_time == last_watched_candle:

                continue

            last_watched_candle = candle_time

            setup["candles_watched"] += 1

            # ------------------------------------------------
            # FIRST: TP / SL
            # ------------------------------------------------

            candle_high = float(
                current["high"]
            )

            candle_low = float(
                current["low"]
            )

            entry = setup["entry"]

            stop = setup["stop"]

            tp = setup["tp"]

            signal = setup["signal"]

            # =================================================
            # LONG
            # =================================================

            if signal == "LONG":

                hit_sl = (
                    candle_low <= stop
                )

                hit_tp = (
                    candle_high >= tp
                )

            # =================================================
            # SHORT
            # =================================================

            else:

                hit_sl = (
                    candle_high >= stop
                )

                hit_tp = (
                    candle_low <= tp
                )

            # ------------------------------------------------
            # Both TP and SL in same candle
            # ------------------------------------------------
            #
            # OHLC data cannot tell us which happened first.
            #
            # Conservative assumption:
            # assume SL happened first.
            #

            if hit_sl and hit_tp:

                status = "SL_FIRST_CONSERVATIVE"

                message = f"""
⚠️ {symbol} — {signal}

Both SL and TP were inside the same H1 candle.

Because OHLC data cannot determine the intrabar order,
AH1 ELITE uses the conservative assumption:

🛑 STOP LOSS FIRST

Entry:
{entry:.6f}

Stop:
{stop:.6f}

TP:
{tp:.6f}

Signal closed.

AH1 ELITE
""".strip()

                await send_ntfy(
                    f"AH1 | {symbol} | SL",
                    message,
                )

                active_setups.pop(
                    symbol,
                    None,
                )

                return

            # ------------------------------------------------
            # SL
            # ------------------------------------------------

            if hit_sl:

                message = f"""
🛑 {symbol} — {signal}

STOP LOSS HIT

Entry:
{entry:.6f}

Stop:
{stop:.6f}

TP:
{tp:.6f}

Watched Candles:
{setup["candles_watched"]}

Signal closed.

AH1 ELITE
""".strip()

                await send_ntfy(
                    f"AH1 | {symbol} | STOP LOSS",
                    message,
                )

                active_setups.pop(
                    symbol,
                    None,
                )

                return

            # ------------------------------------------------
            # TP
            # ------------------------------------------------

            if hit_tp:

                message = f"""
🎯 {symbol} — {signal}

TAKE PROFIT HIT

Entry:
{entry:.6f}

Take Profit:
{tp:.6f}

Risk/Reward:
1:{setup["rr"]:.2f}

Watched Candles:
{setup["candles_watched"]}

Signal closed successfully.

AH1 ELITE
""".strip()

                await send_ntfy(
                    f"AH1 | {symbol} | TAKE PROFIT",
                    message,
                )

                active_setups.pop(
                    symbol,
                    None,
                )

                return

            # ------------------------------------------------
            # RE-CALCULATE H1 SETUP
            # ------------------------------------------------

            result = analyze_1h(
                df1,
                df4,
            )

            new_score = result[
                "score"
            ]

            # ------------------------------------------------
            # INVALIDATION
            # ------------------------------------------------

            #
            # We DO NOT cancel simply because score
            # went down.
            #
            # We cancel only when the original H1 core
            # structure is actually gone.
            #

            if signal == "LONG":

                still_valid = (
                    result["core_long"]
                )

            else:

                still_valid = (
                    result["core_short"]
                )

            if not still_valid:

                message = f"""
🔴 {symbol} — {signal}

SETUP INVALIDATED / CANCELLED

Original Score:
{setup["score"]}/100

Current Score:
{new_score}/100

Reason:
The MAIN H1 setup is no longer valid.

H1 Structure:
INVALIDATED

Watched Candles:
{setup["candles_watched"]}

This signal is now closed and will not be
reported again.

AH1 ELITE
""".strip()

                await send_ntfy(
                    f"AH1 | {symbol} | INVALIDATED",
                    message,
                )

                active_setups.pop(
                    symbol,
                    None,
                )

                return

            # ------------------------------------------------
            # STILL VALID
            # ------------------------------------------------

            if new_score > previous_score:

                status = "🟢 SETUP STRENGTHENED"

            elif new_score < previous_score:

                status = "🟡 SETUP WEAKENED"

            else:

                status = "🟢 SETUP STILL ACTIVE"

            update_message = format_watch_update(
                symbol,
                setup,
                current,
                result,
                status,
                previous_score,
            )

            await send_ntfy(
                f"AH1 | {symbol} | UPDATE",
                update_message,
            )

            previous_score = new_score

            # ------------------------------------------------
            # AFTER 6 CANDLES
            # ------------------------------------------------

            if setup["candles_watched"] >= WATCH_MAX_CANDLES:

                #
                # IMPORTANT:
                #
                # We DO NOT cancel it.
                #
                # We simply stop the initial watcher.
                #

                final_message = f"""
🟢 {symbol} — {signal}

SETUP STILL ACTIVE

Initial monitoring period completed:
6 × H1 candles

Current Score:
{new_score}/100

Original Score:
{setup["score"]}/100

H1 Structure:
VALID

4H Confirmation:
{result["confirmation"]}

The setup was NOT cancelled because
the H1 setup is still valid.

Initial watcher finished.

AH1 ELITE
""".strip()

                await send_ntfy(
                    f"AH1 | {symbol} | STILL ACTIVE",
                    final_message,
                )

                active_setups.pop(
                    symbol,
                    None,
                )

                return

    except asyncio.CancelledError:

        logger.info(
            "%s watcher cancelled.",
            symbol,
        )

        raise

    except Exception as e:

        logger.exception(
            "%s watcher error: %s",
            symbol,
            e,
        )

        active_setups.pop(
            symbol,
            None,
        )


# ============================================================
# START WATCHER
# ============================================================

def start_watcher(
    symbol,
    result,
    levels,
):

    #
    # Only one active setup per symbol.
    #

    if symbol in watcher_tasks:

        task = watcher_tasks[symbol]

        if not task.done():

            logger.info(
                "%s already has an active watcher.",
                symbol,
            )

            return

    setup = {
        "signal": result["signal"],
        "quality": result["quality"],
        "score": result["score"],
        "entry": levels["entry"],
        "stop": levels["stop"],
        "tp": levels["tp"],
        "rr": levels["rr"],
        "signal_candle": result[
            "row"
        ]["timestamp"],
        "candles_watched": 0,
        "reasons": result["reasons"],
    }

    active_setups[symbol] = setup

    task = asyncio.create_task(
        watch_setup(
            symbol,
            setup,
        )
    )

    watcher_tasks[symbol] = task


# ============================================================
# ANALYZE SYMBOL
# ============================================================

async def analyze_symbol(symbol):

    df1 = await fetch_dataframe(
        symbol,
        MAIN_TIMEFRAME,
    )

    df4 = await fetch_dataframe(
        symbol,
        CONFIRM_TIMEFRAME,
    )

    if df1 is None or df4 is None:

        return

    if (
        len(df1) < 220
        or
        len(df4) < 220
    ):

        logger.warning(
            "%s insufficient data.",
            symbol,
        )

        return

    df1 = add_indicators(
        df1
    )

    df4 = add_indicators(
        df4
    )

    result = analyze_1h(
        df1,
        df4,
    )

    # --------------------------------------------------------
    # NO TRADE
    # --------------------------------------------------------

    if result["signal"] == "NO_TRADE":

        logger.info(
            "%s | NO TRADE | L=%s S=%s",
            symbol,
            result["long_score"],
            result["short_score"],
        )

        return

    # --------------------------------------------------------
    # LEVELS
    # --------------------------------------------------------

    levels = calculate_levels(
        result
    )

    if levels is None:

        logger.warning(
            "%s invalid levels.",
            symbol,
        )

        return

    # --------------------------------------------------------
    # SIGNAL KEY
    # --------------------------------------------------------

    candle_time = result[
        "row"
    ]["timestamp"]

    signal_key = (
        result["signal"],
        candle_time,
    )

    #
    # Don't repeat the same signal on the same candle.
    #

    if last_signal_key.get(symbol) == signal_key:

        return

    #
    # If this symbol already has an active setup,
    # don't create another competing setup.
    #

    if symbol in active_setups:

        logger.info(
            "%s already being watched.",
            symbol,
        )

        return

    last_signal_key[symbol] = signal_key

    # --------------------------------------------------------
    # SEND INITIAL SIGNAL
    # --------------------------------------------------------

    message = format_signal(
        symbol,
        result,
        levels,
    )

    logger.info(
        "\n%s\n",
        message,
    )

    await send_ntfy(
        f"AH1 | {symbol} | {result['signal']}",
        message,
    )

    # --------------------------------------------------------
    # START WATCHER
    # --------------------------------------------------------

    start_watcher(
        symbol,
        result,
        levels,
    )


# ============================================================
# CLEAN FINISHED TASKS
# ============================================================

def cleanup_watcher_tasks():

    finished = []

    for symbol, task in watcher_tasks.items():

        if task.done():

            finished.append(
                symbol
            )

    for symbol in finished:

        watcher_tasks.pop(
            symbol,
            None,
        )


# ============================================================
# MAIN LOOP
# ============================================================

async def main():

    logger.info("=" * 65)
    logger.info(
        "AH1 ELITE v2 STARTING"
    )
    logger.info(
        "Main timeframe: 1H"
    )
    logger.info(
        "Confirmation timeframe: 4H"
    )
    logger.info(
        "Long + Short enabled"
    )
    logger.info(
        "4H = confirmation score ONLY"
    )
    logger.info(
        "Watcher = 6 closed H1 candles"
    )
    logger.info(
        "6 candles do NOT automatically cancel setup"
    )
    logger.info("=" * 65)

    try:

        while True:

            cleanup_watcher_tasks()

            for symbol in SYMBOLS:

                try:

                    await analyze_symbol(
                        symbol
                    )

                except Exception as e:

                    logger.exception(
                        "%s analysis failed: %s",
                        symbol,
                        e,
                    )

                await asyncio.sleep(
                    1
                )

            await asyncio.sleep(
                SCAN_INTERVAL
            )

    finally:

        #
        # Cancel active watchers
        #

        for task in watcher_tasks.values():

            if not task.done():

                task.cancel()

        if watcher_tasks:

            await asyncio.gather(
                *watcher_tasks.values(),
                return_exceptions=True,
            )

        await exchange.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "AH1 ELITE stopped."
        )

    except Exception as e:

        logger.exception(
            "Fatal error: %s",
            e,
        )
