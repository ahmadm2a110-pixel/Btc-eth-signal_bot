import asyncio
import logging
import math
import os
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd


# ============================================================
# AH1 ELITE — NEW ENGINE
#
# MARKET DATA
#      │
#      ├── SIGNAL ENGINE
#      │      H1 STRUCTURE
#      │      → SETUP
#      │      → BREAKOUT / BREAKDOWN
#      │      → M15 CONFIRMATION
#      │      → SCORE
#      │      → A+ / A / NO TRADE
#      │
#      └── IMPULSE MONITOR
#             ABNORMAL MOVE
#             → ALERT ONLY
#             → NO ENTRY
#             → NO TP / SL
#             → NO TRADE SIGNAL
#
# DATA: Binance
# ALERTS: ntfy
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

BINANCE_EXCHANGE = "binance"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "LINK/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
]

MAIN_TIMEFRAME = "1h"
CONFIRM_TIMEFRAME = "15m"

CANDLE_LIMIT_H1 = 250
CANDLE_LIMIT_M15 = 250

# ------------------------------------------------------------
# SIGNAL SCORE
# ------------------------------------------------------------

A_PLUS_SCORE = 85
A_SCORE = 75

# ------------------------------------------------------------
# INDICATOR SETTINGS
# ------------------------------------------------------------

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

VOLUME_PERIOD = 20

# ------------------------------------------------------------
# SIGNAL DISTANCE CONTROL
# ------------------------------------------------------------

MAX_EMA_DISTANCE_ATR = 2.0

# ------------------------------------------------------------
# IMPULSE ALERT
# ------------------------------------------------------------

IMPULSE_ATR_MULTIPLIER = 1.8
IMPULSE_BODY_RATIO = 0.65
IMPULSE_VOLUME_MULTIPLIER = 1.8

# Consecutive strong candles
IMPULSE_CONSECUTIVE_CANDLES = 2

# Don't repeatedly alert the same candle
IMPULSE_COOLDOWN_SECONDS = 60 * 60

# ------------------------------------------------------------
# LOOP
# ------------------------------------------------------------

SCAN_INTERVAL_SECONDS = 60

# ------------------------------------------------------------
# NTFY
# ------------------------------------------------------------

NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC = "btc_ah7K9xQ2_signal"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AH1_ELITE")


# ============================================================
# GLOBAL STATE
# ============================================================

last_signal_candle = {}
last_impulse_alert = {}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return 0.0
        value = float(value)

        if not math.isfinite(value):
            return 0.0

        return value
    except Exception:
        return 0.0


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[
        (up_move > down_move) &
        (up_move > 0)
    ] = up_move[
        (up_move > down_move) &
        (up_move > 0)
    ]

    minus_dm[
        (down_move > up_move) &
        (down_move > 0)
    ] = down_move[
        (down_move > up_move) &
        (down_move > 0)
    ]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean() /
        atr.replace(0, float("nan"))
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean() /
        atr.replace(0, float("nan"))
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, float("nan"))
    )

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return (
        plus_di.fillna(0),
        minus_di.fillna(0),
        adx.fillna(0),
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataframe(ohlcv):
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

    if df.empty:
        return df

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["ema20"] = calculate_ema(
        df["close"],
        EMA_FAST,
    )

    df["ema50"] = calculate_ema(
        df["close"],
        EMA_MID,
    )

    df["ema200"] = calculate_ema(
        df["close"],
        EMA_SLOW,
    )

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD,
    )

    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD,
    )

    plus_di, minus_di, adx = calculate_adx(
        df,
        ADX_PERIOD,
    )

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    df["volume_sma"] = (
        df["volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, float("nan"))
    ).fillna(0)

    return df


# ============================================================
# CLOSED CANDLE
# ============================================================

def get_closed_dataframe(df):
    if len(df) < 5:
        return df

    # Last candle may still be forming.
    return df.iloc[:-1].copy()


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure(df):
    if len(df) < 10:
        return "UNKNOWN"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    recent_high = df["high"].iloc[-6:-1].max()
    recent_low = df["low"].iloc[-6:-1].min()

    bullish_break = (
        current["close"] > recent_high
    )

    bearish_break = (
        current["close"] < recent_low
    )

    bullish_trend = (
        current["ema20"] > current["ema50"] and
        current["ema50"] > current["ema200"]
    )

    bearish_trend = (
        current["ema20"] < current["ema50"] and
        current["ema50"] < current["ema200"]
    )

    if bullish_break:
        return "BULL_BREAKOUT"

    if bearish_break:
        return "BEAR_BREAKDOWN"

    if bullish_trend:
        return "BULLISH"

    if bearish_trend:
        return "BEARISH"

    return "RANGE"


# ============================================================
# H1 SIGNAL ENGINE
# ============================================================

def analyze_h1_signal(df):
    if len(df) < 210:
        return None

    structure = detect_structure(df)

    current = df.iloc[-1]

    score_long = 0
    score_short = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # STRUCTURE — 25
    # ========================================================

    if structure == "BULL_BREAKOUT":
        score_long += 25
        long_reasons.append("H1 bullish breakout")

    elif structure == "BULLISH":
        score_long += 18
        long_reasons.append("H1 bullish structure")

    if structure == "BEAR_BREAKDOWN":
        score_short += 25
        short_reasons.append("H1 bearish breakdown")

    elif structure == "BEARISH":
        score_short += 18
        short_reasons.append("H1 bearish structure")

    # ========================================================
    # TREND ALIGNMENT — 10
    # ========================================================

    if (
        current["ema20"] >
        current["ema50"] >
        current["ema200"]
    ):
        score_long += 10
        long_reasons.append("EMA trend aligned")

    if (
        current["ema20"] <
        current["ema50"] <
        current["ema200"]
    ):
        score_short += 10
        short_reasons.append("EMA trend aligned")

    # ========================================================
    # MOMENTUM — 15
    # ========================================================

    if 55 <= current["rsi"] <= 72:
        score_long += 15
        long_reasons.append("Bullish RSI momentum")

    if 28 <= current["rsi"] <= 45:
        score_short += 15
        short_reasons.append("Bearish RSI momentum")

    # ========================================================
    # ADX — 10
    # ========================================================

    if current["adx"] >= 20:
        if current["plus_di"] > current["minus_di"]:
            score_long += 10
            long_reasons.append("ADX bullish strength")

        elif current["minus_di"] > current["plus_di"]:
            score_short += 10
            short_reasons.append("ADX bearish strength")

    # ========================================================
    # VOLUME — 10
    # ========================================================

    if (
        current["volume_sma"] > 0 and
        current["volume"] >
        current["volume_sma"] * 1.2
    ):
        if current["close"] > current["open"]:
            score_long += 10
            long_reasons.append("Volume confirmation")

        elif current["close"] < current["open"]:
            score_short += 10
            short_reasons.append("Volume confirmation")

    # ========================================================
    # CANDLE MOMENTUM — 10
    # ========================================================

    if (
        current["body_ratio"] >= 0.60 and
        current["close"] > current["open"]
    ):
        score_long += 10
        long_reasons.append("Strong bullish candle")

    if (
        current["body_ratio"] >= 0.60 and
        current["close"] < current["open"]
    ):
        score_short += 10
        short_reasons.append("Strong bearish candle")

    # ========================================================
    # ATR / VOLATILITY — 10
    # ========================================================

    if current["atr"] > 0:

        ema_distance = abs(
            current["close"] -
            current["ema20"]
        )

        distance_atr = (
            ema_distance /
            current["atr"]
        )

        if distance_atr <= MAX_EMA_DISTANCE_ATR:

            if current["close"] > current["ema20"]:
                score_long += 10
                long_reasons.append(
                    "Price not overextended"
                )

            if current["close"] < current["ema20"]:
                score_short += 10
                short_reasons.append(
                    "Price not overextended"
                )

    # ========================================================
    # CHOOSE SIDE
    # ========================================================

    if score_long >= A_SCORE and score_long > score_short:
        return {
            "side": "LONG",
            "score": min(score_long, 100),
            "grade": (
                "A+"
                if score_long >= A_PLUS_SCORE
                else "A"
            ),
            "structure": structure,
            "reasons": long_reasons,
            "candle_time": int(current["timestamp"]),
        }

    if score_short >= A_SCORE and score_short > score_long:
        return {
            "side": "SHORT",
            "score": min(score_short, 100),
            "grade": (
                "A+"
                if score_short >= A_PLUS_SCORE
                else "A"
            ),
            "structure": structure,
            "reasons": short_reasons,
            "candle_time": int(current["timestamp"]),
        }

    return None


# ============================================================
# M15 CONFIRMATION
# ============================================================

def confirm_m15(df, side):
    if len(df) < 30:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    if side == "LONG":

        bullish_candle = (
            current["close"] >
            current["open"]
        )

        momentum = (
            current["close"] >
            previous["high"]
        )

        rsi_ok = (
            current["rsi"] >= 50
        )

        return (
            bullish_candle and
            momentum and
            rsi_ok
        )

    if side == "SHORT":

        bearish_candle = (
            current["close"] <
            current["open"]
        )

        momentum = (
            current["close"] <
            previous["low"]
        )

        rsi_ok = (
            current["rsi"] <= 50
        )

        return (
            bearish_candle and
            momentum and
            rsi_ok
        )

    return False


# ============================================================
# IMPULSE DETECTOR
#
# IMPORTANT:
# This is NOT a trading signal.
#
# It only detects an unusually strong market move and
# tells the user to manually inspect the market.
# ============================================================

def detect_impulse(df):
    if len(df) < 30:
        return None

    current = df.iloc[-1]

    if current["atr"] <= 0:
        return None

    candle_range = (
        current["high"] -
        current["low"]
    )

    body = abs(
        current["close"] -
        current["open"]
    )

    body_ratio = (
        body / candle_range
        if candle_range > 0
        else 0
    )

    range_atr = (
        candle_range /
        current["atr"]
    )

    volume_ratio = 0

    if current["volume_sma"] > 0:
        volume_ratio = (
            current["volume"] /
            current["volume_sma"]
        )

    bullish = (
        current["close"] >
        current["open"]
    )

    bearish = (
        current["close"] <
        current["open"]
    )

    # --------------------------------------------------------
    # Strong single candle
    # --------------------------------------------------------

    strong_single = (
        range_atr >= IMPULSE_ATR_MULTIPLIER and
        body_ratio >= IMPULSE_BODY_RATIO and
        volume_ratio >= IMPULSE_VOLUME_MULTIPLIER
    )

    # --------------------------------------------------------
    # Consecutive candles
    # --------------------------------------------------------

    consecutive_direction = None

    if len(df) >= IMPULSE_CONSECUTIVE_CANDLES + 1:

        recent = df.iloc[
            -IMPULSE_CONSECUTIVE_CANDLES:
        ]

        bullish_count = (
            recent["close"] >
            recent["open"]
        ).sum()

        bearish_count = (
            recent["close"] <
            recent["open"]
        ).sum()

        if bullish_count == IMPULSE_CONSECUTIVE_CANDLES:
            consecutive_direction = "BULLISH"

        elif bearish_count == IMPULSE_CONSECUTIVE_CANDLES:
            consecutive_direction = "BEARISH"

    if not strong_single and not consecutive_direction:
        return None

    if bearish or consecutive_direction == "BEARISH":
        direction = "BEARISH"

    elif bullish or consecutive_direction == "BULLISH":
        direction = "BULLISH"

    else:
        return None

    return {
        "direction": direction,
        "range_atr": range_atr,
        "body_ratio": body_ratio,
        "volume_ratio": volume_ratio,
        "candle_time": int(current["timestamp"]),
    }


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(
    client,
    title,
    message,
    priority="default",
    tags=None,
):
    headers = {
        "Title": title,
        "Priority": priority,
    }

    if tags:
        headers["Tags"] = tags

    try:
        response = await client.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            content=message.encode("utf-8"),
            headers=headers,
            timeout=15,
        )

        if response.status_code >= 400:
            logger.error(
                "NTFY error: %s | %s",
                response.status_code,
                response.text,
            )

            return False

        logger.info("NTFY notification sent.")
        return True

    except Exception as exc:
        logger.error(
            "NTFY send error: %s",
            exc,
        )

        return False


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(symbol, signal):
    reasons = "\n".join(
        f"• {reason}"
        for reason in signal["reasons"]
    )

    return (
        f"🎯 {symbol} — {signal['side']}\n\n"
        f"Grade: {signal['grade']}\n"
        f"Score: {signal['score']}/100\n\n"
        f"H1 Structure:\n"
        f"{signal['structure']}\n\n"
        f"Confirmation:\n"
        f"M15 confirmed\n\n"
        f"Reasons:\n"
        f"{reasons}\n\n"
        f"AH1 ELITE"
    )


# ============================================================
# FORMAT IMPULSE ALERT
# ============================================================

def format_impulse_alert(symbol, impulse):
    direction = impulse["direction"]

    emoji = (
        "🔴"
        if direction == "BEARISH"
        else "🟢"
    )

    return (
        f"🚨 {symbol} — STRONG "
        f"{direction} IMPULSE\n\n"

        f"{emoji} Unusually strong market movement "
        f"detected.\n\n"

        f"Range / ATR: "
        f"{impulse['range_atr']:.2f}x\n"

        f"Body Ratio: "
        f"{impulse['body_ratio'] * 100:.0f}%\n"

        f"Volume: "
        f"{impulse['volume_ratio']:.2f}x average\n\n"

        f"⚠️ NO TRADE SIGNAL\n"
        f"⚠️ NO ENTRY\n"
        f"⚠️ NO TP / SL\n\n"

        f"👀 Check the market manually.\n\n"

        f"AH1 ELITE — IMPULSE ALERT"
    )


# ============================================================
# FETCH OHLCV
# ============================================================

async def fetch_ohlcv(
    exchange,
    symbol,
    timeframe,
    limit,
):
    try:
        data = await exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not data:
            return None

        return prepare_dataframe(data)

    except Exception as exc:
        logger.error(
            "%s %s data error: %s",
            symbol,
            timeframe,
            exc,
        )

        return None


# ============================================================
# PROCESS SYMBOL
# ============================================================

async def process_symbol(
    exchange,
    client,
    symbol,
):
    h1_raw = await fetch_ohlcv(
        exchange,
        symbol,
        MAIN_TIMEFRAME,
        CANDLE_LIMIT_H1,
    )

    if h1_raw is None or h1_raw.empty:
        return

    m15_raw = await fetch_ohlcv(
        exchange,
        symbol,
        CONFIRM_TIMEFRAME,
        CANDLE_LIMIT_M15,
    )

    if m15_raw is None or m15_raw.empty:
        return

    h1 = get_closed_dataframe(h1_raw)
    m15 = get_closed_dataframe(m15_raw)

    if len(h1) < 210 or len(m15) < 30:
        return

    # ========================================================
    # 1 — IMPULSE MONITOR
    #
    # This runs independently from the trading engine.
    # ========================================================

    impulse = detect_impulse(h1)

    if impulse:

        candle_time = impulse["candle_time"]

        last_alert = last_impulse_alert.get(
            symbol
        )

        if last_alert != candle_time:

            logger.warning(
                "IMPULSE | %s | %s | %.2fx ATR | %.2fx volume",
                symbol,
                impulse["direction"],
                impulse["range_atr"],
                impulse["volume_ratio"],
            )

            await send_ntfy(
                client,
                f"🚨 {symbol} IMPULSE",
                format_impulse_alert(
                    symbol,
                    impulse,
                ),
                priority="high",
                tags=(
                    "rotating_light,"
                    "chart_with_upwards_trend"
                    if impulse["direction"] == "BULLISH"
                    else
                    "rotating_light,"
                    "chart_with_downwards_trend"
                ),
            )

            last_impulse_alert[symbol] = candle_time

    # ========================================================
    # 2 — NORMAL SIGNAL ENGINE
    # ========================================================

    signal = analyze_h1_signal(h1)

    if signal is None:
        return

    if not confirm_m15(
        m15,
        signal["side"],
    ):
        return

    candle_time = signal["candle_time"]

    if last_signal_candle.get(symbol) == candle_time:
        return

    last_signal_candle[symbol] = candle_time

    logger.info(
        "SIGNAL | %s | %s | %s | %s/100",
        symbol,
        signal["side"],
        signal["grade"],
        signal["score"],
    )

    await send_ntfy(
        client,
        (
            f"🎯 {symbol} "
            f"{signal['side']} "
            f"{signal['grade']}"
        ),
        format_signal(
            symbol,
            signal,
        ),
        priority="high",
        tags=(
            "chart_with_upwards_trend"
            if signal["side"] == "LONG"
            else "chart_with_downwards_trend"
        ),
    )


# ============================================================
# MAIN LOOP
# ============================================================

async def main():
    logger.info("=" * 60)
    logger.info("AH1 ELITE — NEW ENGINE STARTING")
    logger.info("=" * 60)
    logger.info(
        "Data source: Binance"
    )
    logger.info(
        "Main timeframe: %s",
        MAIN_TIMEFRAME,
    )
    logger.info(
        "Confirmation timeframe: %s",
        CONFIRM_TIMEFRAME,
    )
    logger.info(
        "Impulse monitor: ENABLED"
    )
    logger.info(
        "Trading signal engine: ENABLED"
    )
    logger.info(
        "NTFY: ENABLED"
    )
    logger.info("=" * 60)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    })

    async with httpx.AsyncClient() as client:

        try:

            await exchange.load_markets()

            logger.info(
                "Binance connection established."
            )

            while True:

                started = datetime.now(
                    timezone.utc
                )

                logger.info(
                    "Scanning market..."
                )

                tasks = [
                    process_symbol(
                        exchange,
                        client,
                        symbol,
                    )
                    for symbol in SYMBOLS
                    if symbol in exchange.markets
                ]

                if tasks:
                    results = await asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    )

                    for symbol, result in zip(
                        SYMBOLS,
                        results,
                    ):
                        if isinstance(
                            result,
                            Exception,
                        ):
                            logger.error(
                                "%s processing error: %s",
                                symbol,
                                result,
                            )

                elapsed = (
                    datetime.now(
                        timezone.utc
                    ) - started
                ).total_seconds()

                logger.info(
                    "Scan completed in %.2fs",
                    elapsed,
                )

                await asyncio.sleep(
                    SCAN_INTERVAL_SECONDS
                )

        finally:

            await exchange.close()

            logger.info(
                "Binance connection closed."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info(
            "Bot stopped manually."
        )

    except Exception as exc:
        logger.exception(
            "Fatal error: %s",
            exc,
        )
