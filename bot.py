import asyncio
import logging
import math
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
#             LIVE M15 MOVE
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


# ============================================================
# SIGNAL SCORE
# ============================================================

A_PLUS_SCORE = 85
A_SCORE = 75


# ============================================================
# INDICATORS
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VOLUME_PERIOD = 20


# ============================================================
# SIGNAL DISTANCE CONTROL
# ============================================================

MAX_EMA_DISTANCE_ATR = 2.0


# ============================================================
# IMPULSE MONITOR
#
# IMPORTANT:
# Impulse is NOT a trading signal.
#
# It watches the CURRENT M15 candle while it is forming.
# ============================================================

IMPULSE_ATR_MULTIPLIER = 1.50
IMPULSE_BODY_RATIO = 0.60
IMPULSE_VOLUME_MULTIPLIER = 1.50

# Minimum percentage move from the M15 open
IMPULSE_MIN_MOVE_PERCENT = 0.70

# Don't alert the same M15 candle repeatedly
IMPULSE_ALERT_COOLDOWN_SECONDS = 15 * 60


# ============================================================
# LOOP
# ============================================================

SCAN_INTERVAL_SECONDS = 60


# ============================================================
# NTFY
# ============================================================

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

# symbol -> {
#     "candle_time": ...,
#     "direction": ...,
#     "last_alert_time": ...
# }
impulse_state = {}


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
    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


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

    rs = (
        avg_gain /
        avg_loss.replace(0, float("nan"))
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

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

    plus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    minus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    plus_condition = (
        (up_move > down_move) &
        (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move) &
        (down_move > 0)
    )

    plus_dm[plus_condition] = up_move[
        plus_condition
    ]

    minus_dm[minus_condition] = down_move[
        minus_condition
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
        (plus_di + minus_di).replace(
            0,
            float("nan"),
        )
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

    (
        df["plus_di"],
        df["minus_di"],
        df["adx"],
    ) = calculate_adx(
        df,
        ADX_PERIOD,
    )

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
        df["range"].replace(
            0,
            float("nan"),
        )
    ).fillna(0)

    return df


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_dataframe(df):

    if len(df) < 5:
        return df

    return df.iloc[:-1].copy()


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure(df):

    if len(df) < 10:
        return "UNKNOWN"

    current = df.iloc[-1]

    recent_high = (
        df["high"]
        .iloc[-6:-1]
        .max()
    )

    recent_low = (
        df["low"]
        .iloc[-6:-1]
        .min()
    )

    bullish_break = (
        current["close"] >
        recent_high
    )

    bearish_break = (
        current["close"] <
        recent_low
    )

    bullish_trend = (
        current["ema20"] >
        current["ema50"] >
        current["ema200"]
    )

    bearish_trend = (
        current["ema20"] <
        current["ema50"] <
        current["ema200"]
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

        long_reasons.append(
            "H1 bullish breakout"
        )

    elif structure == "BULLISH":

        score_long += 18

        long_reasons.append(
            "H1 bullish structure"
        )

    if structure == "BEAR_BREAKDOWN":

        score_short += 25

        short_reasons.append(
            "H1 bearish breakdown"
        )

    elif structure == "BEARISH":

        score_short += 18

        short_reasons.append(
            "H1 bearish structure"
        )

    # ========================================================
    # TREND ALIGNMENT — 10
    # ========================================================

    if (
        current["ema20"] >
        current["ema50"] >
        current["ema200"]
    ):

        score_long += 10

        long_reasons.append(
            "EMA trend aligned"
        )

    if (
        current["ema20"] <
        current["ema50"] <
        current["ema200"]
    ):

        score_short += 10

        short_reasons.append(
            "EMA trend aligned"
        )

    # ========================================================
    # MOMENTUM — 15
    # ========================================================

    if 55 <= current["rsi"] <= 72:

        score_long += 15

        long_reasons.append(
            "Bullish RSI momentum"
        )

    if 28 <= current["rsi"] <= 45:

        score_short += 15

        short_reasons.append(
            "Bearish RSI momentum"
        )

    # ========================================================
    # ADX — 10
    # ========================================================

    if current["adx"] >= 20:

        if (
            current["plus_di"] >
            current["minus_di"]
        ):

            score_long += 10

            long_reasons.append(
                "ADX bullish strength"
            )

        elif (
            current["minus_di"] >
            current["plus_di"]
        ):

            score_short += 10

            short_reasons.append(
                "ADX bearish strength"
            )

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

            long_reasons.append(
                "Volume confirmation"
            )

        elif current["close"] < current["open"]:

            score_short += 10

            short_reasons.append(
                "Volume confirmation"
            )

    # ========================================================
    # CANDLE MOMENTUM — 10
    # ========================================================

    if (
        current["body_ratio"] >= 0.60 and
        current["close"] > current["open"]
    ):

        score_long += 10

        long_reasons.append(
            "Strong bullish candle"
        )

    if (
        current["body_ratio"] >= 0.60 and
        current["close"] < current["open"]
    ):

        score_short += 10

        short_reasons.append(
            "Strong bearish candle"
        )

    # ========================================================
    # ATR / EXTENSION — 10
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

            elif current["close"] < current["ema20"]:

                score_short += 10

                short_reasons.append(
                    "Price not overextended"
                )

    # ========================================================
    # CHOOSE SIDE
    # ========================================================

    if (
        score_long >= A_SCORE and
        score_long > score_short
    ):

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
            "candle_time": int(
                current["timestamp"]
            ),
        }

    if (
        score_short >= A_SCORE and
        score_short > score_long
    ):

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
            "candle_time": int(
                current["timestamp"]
            ),
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
# LIVE M15 IMPULSE DETECTOR
#
# IMPORTANT:
#
# We intentionally use the LAST candle from m15_raw.
# This candle is still forming.
#
# The ATR / volume baseline is calculated from PREVIOUS
# CLOSED candles so the current candle doesn't distort
# its own benchmark.
# ============================================================

def detect_live_m15_impulse(df):

    if len(df) < 35:
        return None

    current = df.iloc[-1]

    # --------------------------------------------------------
    # Use previous CLOSED candles for baseline
    # --------------------------------------------------------

    closed = df.iloc[:-1].copy()

    if len(closed) < 30:
        return None

    baseline_atr = safe_float(
        closed["atr"].iloc[-1]
    )

    baseline_volume = safe_float(
        closed["volume_sma"].iloc[-1]
    )

    if baseline_atr <= 0:
        return None

    current_open = safe_float(
        current["open"]
    )

    current_close = safe_float(
        current["close"]
    )

    current_high = safe_float(
        current["high"]
    )

    current_low = safe_float(
        current["low"]
    )

    current_volume = safe_float(
        current["volume"]
    )

    if current_open <= 0:
        return None

    # --------------------------------------------------------
    # Current live candle
    # --------------------------------------------------------

    candle_range = (
        current_high -
        current_low
    )

    body = abs(
        current_close -
        current_open
    )

    body_ratio = (
        body / candle_range
        if candle_range > 0
        else 0
    )

    range_atr = (
        candle_range /
        baseline_atr
    )

    volume_ratio = (
        current_volume /
        baseline_volume
        if baseline_volume > 0
        else 0
    )

    move_percent = (
        abs(
            current_close -
            current_open
        ) /
        current_open
    ) * 100

    bullish = (
        current_close >
        current_open
    )

    bearish = (
        current_close <
        current_open
    )

    # --------------------------------------------------------
    # Strong LIVE impulse
    #
    # Need multiple confirmations.
    # --------------------------------------------------------

    range_condition = (
        range_atr >=
        IMPULSE_ATR_MULTIPLIER
    )

    body_condition = (
        body_ratio >=
        IMPULSE_BODY_RATIO
    )

    volume_condition = (
        volume_ratio >=
        IMPULSE_VOLUME_MULTIPLIER
    )

    move_condition = (
        move_percent >=
        IMPULSE_MIN_MOVE_PERCENT
    )

    # Strong impulse requires:
    #
    # 1) abnormal range
    # 2) strong body
    # 3) meaningful move
    #
    # Volume is an additional confirmation.
    #
    strong_price_impulse = (
        range_condition and
        body_condition and
        move_condition
    )

    confirmed_impulse = (
        strong_price_impulse and
        volume_condition
    )

    if not strong_price_impulse:
        return None

    if bullish:
        direction = "BULLISH"

    elif bearish:
        direction = "BEARISH"

    else:
        return None

    return {
        "direction": direction,
        "range_atr": range_atr,
        "body_ratio": body_ratio,
        "volume_ratio": volume_ratio,
        "move_percent": move_percent,
        "confirmed_by_volume": confirmed_impulse,
        "candle_time": int(
            current["timestamp"]
        ),
    }


# ============================================================
# IMPULSE ALERT CONTROL
# ============================================================

def should_send_impulse_alert(
    symbol,
    impulse,
):

    candle_time = impulse["candle_time"]

    now = datetime.now(
        timezone.utc
    ).timestamp()

    state = impulse_state.get(
        symbol
    )

    if state is None:

        impulse_state[symbol] = {
            "candle_time": candle_time,
            "direction": impulse["direction"],
            "last_alert_time": 0,
        }

        return True

    # New M15 candle
    if (
        state["candle_time"] !=
        candle_time
    ):

        impulse_state[symbol] = {
            "candle_time": candle_time,
            "direction": impulse["direction"],
            "last_alert_time": 0,
        }

        return True

    # Direction changed
    if (
        state["direction"] !=
        impulse["direction"]
    ):

        state["direction"] = (
            impulse["direction"]
        )

        state["last_alert_time"] = 0

        return True

    # Cooldown
    elapsed = (
        now -
        state["last_alert_time"]
    )

    if (
        elapsed >=
        IMPULSE_ALERT_COOLDOWN_SECONDS
    ):

        return True

    return False


def mark_impulse_alert_sent(
    symbol,
    impulse,
):

    candle_time = impulse["candle_time"]

    now = datetime.now(
        timezone.utc
    ).timestamp()

    state = impulse_state.setdefault(
        symbol,
        {
            "candle_time": candle_time,
            "direction": impulse["direction"],
            "last_alert_time": 0,
        },
    )

    state["candle_time"] = candle_time
    state["direction"] = impulse["direction"]
    state["last_alert_time"] = now


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

    # IMPORTANT:
    # Ntfy HTTP headers are kept ASCII-only.
    # Emoji stay inside the UTF-8 message body.

    clean_title = (
        str(title)
        .encode(
            "ascii",
            "ignore",
        )
        .decode("ascii")
    )

    headers = {
        "Title": clean_title,
        "Priority": str(priority),
    }

    if tags:

        clean_tags = (
            str(tags)
            .encode(
                "ascii",
                "ignore",
            )
            .decode("ascii")
        )

        headers["Tags"] = clean_tags

    try:

        response = await client.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            content=message.encode(
                "utf-8"
            ),
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

        logger.info(
            "NTFY notification sent."
        )

        return True

    except Exception as exc:

        logger.error(
            "NTFY send error: %s",
            exc,
        )

        return False


# ============================================================
# FORMAT NORMAL SIGNAL
# ============================================================

def format_signal(
    symbol,
    signal,
):

    reasons = "\n".join(
        f"• {reason}"
        for reason in signal["reasons"]
    )

    return (
        f"🎯 {symbol} — "
        f"{signal['side']}\n\n"

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

def format_impulse_alert(
    symbol,
    impulse,
):

    direction = impulse["direction"]

    emoji = (
        "🔴"
        if direction == "BEARISH"
        else "🟢"
    )

    volume_text = (
        "YES"
        if impulse["confirmed_by_volume"]
        else "NO"
    )

    return (
        f"🚨 {symbol} — "
        f"STRONG {direction} IMPULSE\n\n"

        f"{emoji} Strong market movement "
        f"is happening NOW on M15.\n\n"

        f"Current Move: "
        f"{impulse['move_percent']:.2f}%\n"

        f"Range / ATR: "
        f"{impulse['range_atr']:.2f}x\n"

        f"Body Ratio: "
        f"{impulse['body_ratio'] * 100:.0f}%\n"

        f"Volume Confirmation: "
        f"{volume_text}\n"

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

        return prepare_dataframe(
            data
        )

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

    # ========================================================
    # H1 DATA
    # ========================================================

    h1_raw = await fetch_ohlcv(
        exchange,
        symbol,
        MAIN_TIMEFRAME,
        CANDLE_LIMIT_H1,
    )

    if (
        h1_raw is None or
        h1_raw.empty
    ):
        return

    # ========================================================
    # M15 DATA
    # ========================================================

    m15_raw = await fetch_ohlcv(
        exchange,
        symbol,
        CONFIRM_TIMEFRAME,
        CANDLE_LIMIT_M15,
    )

    if (
        m15_raw is None or
        m15_raw.empty
    ):
        return

    if len(h1_raw) < 210:
        return

    if len(m15_raw) < 35:
        return

    # ========================================================
    # CLOSED DATA FOR SIGNAL ENGINE
    # ========================================================

    h1 = get_closed_dataframe(
        h1_raw
    )

    m15 = get_closed_dataframe(
        m15_raw
    )

    # ========================================================
    # 1 — LIVE M15 IMPULSE MONITOR
    #
    # IMPORTANT:
    # Uses m15_raw, NOT closed m15.
    #
    # This means the current M15 candle is monitored live.
    # ========================================================

    impulse = detect_live_m15_impulse(
        m15_raw
    )

    if impulse:

        if should_send_impulse_alert(
            symbol,
            impulse,
        ):

            logger.warning(
                "IMPULSE | %s | %s | "
                "%.2fx ATR | %.2fx volume | "
                "%.2f%% move",
                symbol,
                impulse["direction"],
                impulse["range_atr"],
                impulse["volume_ratio"],
                impulse["move_percent"],
            )

            sent = await send_ntfy(
                client,
                f"{symbol} IMPULSE",
                format_impulse_alert(
                    symbol,
                    impulse,
                ),
                priority="high",
                tags=(
                    "rotating_light,"
                    "chart_with_downwards_trend"
                    if impulse["direction"]
                    == "BEARISH"
                    else
                    "rotating_light,"
                    "chart_with_upwards_trend"
                ),
            )

            if sent:
                mark_impulse_alert_sent(
                    symbol,
                    impulse,
                )

    # ========================================================
    # 2 — NORMAL SIGNAL ENGINE
    #
    # Uses ONLY closed candles.
    # ========================================================

    signal = analyze_h1_signal(
        h1
    )

    if signal is None:
        return

    # M15 confirmation also uses CLOSED candle.
    if not confirm_m15(
        m15,
        signal["side"],
    ):
        return

    candle_time = signal[
        "candle_time"
    ]

    if (
        last_signal_candle.get(
            symbol
        ) == candle_time
    ):
        return

    last_signal_candle[
        symbol
    ] = candle_time

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
            f"{symbol} "
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
            else
            "chart_with_downwards_trend"
        ),
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "=" * 60
    )

    logger.info(
        "AH1 ELITE — NEW ENGINE STARTING"
    )

    logger.info(
        "=" * 60
    )

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
        "Impulse mode: LIVE M15"
    )

    logger.info(
        "Trading signal engine: ENABLED"
    )

    logger.info(
        "NTFY: ENABLED"
    )

    logger.info(
        "=" * 60
    )

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

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped manually."
        )

    except Exception as exc:

        logger.exception(
            "Fatal error: %s",
            exc,
        )
