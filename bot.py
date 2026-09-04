import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd


# ============================================================
# AH1 TRADING ASSISTANT
#
# H1 TREND FILTER
#       ->
# M15 SETUP DETECTION
#       ->
# M15 CONFIRMATION
#       ->
# SCORE
#       ->
# A / A+
#       ->
# SIGNAL
#
# IMPORTANT:
# - H1 ONLY determines market direction.
# - M15 ONLY generates trade setups.
# - Every M15 candle close is a SCAN, not an automatic signal.
# - The same setup cannot generate repeated signals.
# - WATCH alerts are disabled.
# - AUTO TRADING DISABLED.
# - NTFY ENABLED.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "LINK/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "VET/USDT",
    "CELR/USDT",
    "DYDX/USDT",
    "JASMY/USDT",
    "CRV/USDT",
    "DOT/USDT",
    "FIL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "RUNE/USDT",
    "NEAR/USDT",
]


# ------------------------------------------------------------
# TIMEFRAMES
# ------------------------------------------------------------

TREND_TIMEFRAME = "1h"
SIGNAL_TIMEFRAME = "15m"

H1_LIMIT = 250
M15_LIMIT = 250


# ------------------------------------------------------------
# SCANNING
# ------------------------------------------------------------

SCAN_INTERVAL = 60


# ------------------------------------------------------------
# SCORING
# ------------------------------------------------------------

MIN_SCORE = 7
A_SCORE = 8
A_PLUS_SCORE = 9


# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------

ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

VOLUME_PERIOD = 20


# ------------------------------------------------------------
# ENTRY FILTERS
# ------------------------------------------------------------

MAX_ENTRY_DISTANCE_ATR = 1.50

RSI_LONG_MIN = 52
RSI_LONG_MAX = 70

RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 48

MIN_ADX = 20
STRONG_ADX = 25

MIN_VOLUME_RATIO = 1.00
STRONG_VOLUME_RATIO = 1.20


# ------------------------------------------------------------
# M15 STRUCTURE
# ------------------------------------------------------------

STRUCTURE_LOOKBACK = 20

BREAKOUT_BUFFER_ATR = 0.05

MIN_BODY_RATIO = 0.55

MIN_RISK_ATR = 0.75


# ------------------------------------------------------------
# SETUP MANAGEMENT
# ------------------------------------------------------------

SETUP_EXPIRY_HOURS = 6

INVALIDATION_ATR_MULTIPLIER = 1.25

# Prevents the engine from creating a new setup
# immediately after an invalidation on the same structure.
REARM_CANDLES = 2


# ------------------------------------------------------------
# NTFY
# ------------------------------------------------------------

NTFY_TOPIC = "btc_ah7K9xQ2_signal"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AH1_ASSISTANT")


# ============================================================
# STATE
# ============================================================

# Active setup per symbol.
active_setups = {}

# Last processed closed M15 candle per symbol.
last_processed_candle = {}

# Last invalidated setup information.
invalidated_setups = {}


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
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None or pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def fmt_price(value):

    value = safe_float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 100:
        return f"{value:,.3f}"

    if value >= 1:
        return f"{value:,.4f}"

    if value >= 0.01:
        return f"{value:,.5f}"

    return f"{value:,.8f}"


def now_utc():

    return datetime.now(timezone.utc)


# ============================================================
# INDICATORS
# ============================================================

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
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            float("nan"),
        )
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

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
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

    plus_mask = (
        (up_move > down_move)
        & (up_move > 0)
    )

    minus_mask = (
        (down_move > up_move)
        & (down_move > 0)
    )

    plus_dm.loc[plus_mask] = (
        up_move.loc[plus_mask]
    )

    minus_dm.loc[minus_mask] = (
        down_move.loc[minus_mask]
    )

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(
            0,
            float("nan"),
        )
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(
            0,
            float("nan"),
        )
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0,
        float("nan"),
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (
        adx.fillna(0),
        plus_di.fillna(0),
        minus_di.fillna(0),
    )


def add_indicators(df):

    df = df.copy()

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

    adx, plus_di, minus_di = calculate_adx(
        df,
        ADX_PERIOD,
    )

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["volume_sma"] = (
        df["volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    return df


# ============================================================
# MARKET DATA
# ============================================================

async def fetch_ohlcv(
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

        df = pd.DataFrame(
            data,
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

        df = (
            df
            .dropna()
            .drop_duplicates(
                subset=["timestamp"]
            )
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        logger.error(
            "Data error %s %s: %s",
            symbol,
            timeframe,
            exc,
        )

        return None


def remove_open_candle(df):

    if df is None:
        return df

    if len(df) < 3:
        return df

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# H1 TREND FILTER
# ============================================================

def detect_h1_trend(df):

    if df is None or len(df) < 210:
        return "NEUTRAL"

    row = df.iloc[-1]

    close = safe_float(
        row["close"]
    )

    ema20 = safe_float(
        row["ema20"]
    )

    ema50 = safe_float(
        row["ema50"]
    )

    ema200 = safe_float(
        row["ema200"]
    )

    adx = safe_float(
        row["adx"]
    )

    plus_di = safe_float(
        row["plus_di"]
    )

    minus_di = safe_float(
        row["minus_di"]
    )

    # Strong bullish H1 trend.
    if (
        close > ema20
        and ema20 > ema50
        and ema50 > ema200
        and plus_di > minus_di
        and adx >= MIN_ADX
    ):
        return "BULLISH"

    # Strong bearish H1 trend.
    if (
        close < ema20
        and ema20 < ema50
        and ema50 < ema200
        and minus_di > plus_di
        and adx >= MIN_ADX
    ):
        return "BEARISH"

    # Everything else is considered neutral.
    return "NEUTRAL"


# ============================================================
# M15 STRUCTURE
# ============================================================

def get_previous_structure_levels(df):

    if len(df) < STRUCTURE_LOOKBACK + 5:
        return 0.0, 0.0

    # Exclude the current signal candle.
    recent = df.iloc[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    previous_high = safe_float(
        recent["high"].max()
    )

    previous_low = safe_float(
        recent["low"].min()
    )

    return (
        previous_low,
        previous_high,
    )


def detect_m15_breakout(
    df,
    direction,
):

    if len(df) < STRUCTURE_LOOKBACK + 5:
        return None

    current = df.iloc[-1]

    previous_low, previous_high = (
        get_previous_structure_levels(df)
    )

    close = safe_float(
        current["close"]
    )

    high = safe_float(
        current["high"]
    )

    low = safe_float(
        current["low"]
    )

    atr = safe_float(
        current["atr"]
    )

    if atr <= 0:
        return None

    buffer = (
        atr * BREAKOUT_BUFFER_ATR
    )

    if direction == "LONG":

        if close > previous_high + buffer:

            return {
                "type": "BULL_BREAKOUT",
                "level": previous_high,
                "candle_timestamp": current["timestamp"],
                "price": close,
            }

    else:

        if close < previous_low - buffer:

            return {
                "type": "BEAR_BREAKDOWN",
                "level": previous_low,
                "candle_timestamp": current["timestamp"],
                "price": close,
            }

    return None


# ============================================================
# M15 PRICE ACTION
# ============================================================

def candle_quality(
    row,
    direction,
):

    open_price = safe_float(
        row["open"]
    )

    close = safe_float(
        row["close"]
    )

    high = safe_float(
        row["high"]
    )

    low = safe_float(
        row["low"]
    )

    total_range = max(
        high - low,
        1e-12,
    )

    body = abs(
        close - open_price
    )

    body_ratio = (
        body / total_range
    )

    if direction == "LONG":

        if close <= open_price:
            return False

    else:

        if close >= open_price:
            return False

    return (
        body_ratio
        >= MIN_BODY_RATIO
    )


def momentum_confirmation(
    df,
    direction,
):

    if len(df) < 3:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    close = safe_float(
        current["close"]
    )

    previous_high = safe_float(
        previous["high"]
    )

    previous_low = safe_float(
        previous["low"]
    )

    if direction == "LONG":

        return (
            close > previous_high
        )

    return (
        close < previous_low
    )


def price_action_direction(
    df,
    direction,
):

    if len(df) < 6:
        return False

    recent = df.iloc[-6:]

    first_close = safe_float(
        recent["close"].iloc[0]
    )

    last_close = safe_float(
        recent["close"].iloc[-1]
    )

    if direction == "LONG":

        return last_close > first_close

    return last_close < first_close


# ============================================================
# M15 SETUP ANALYSIS
# ============================================================

def analyze_m15_setup(
    df,
    direction,
    breakout,
):

    row = df.iloc[-1]

    price = safe_float(
        row["close"]
    )

    ema20 = safe_float(
        row["ema20"]
    )

    ema50 = safe_float(
        row["ema50"]
    )

    ema200 = safe_float(
        row["ema200"]
    )

    rsi = safe_float(
        row["rsi"]
    )

    atr = safe_float(
        row["atr"]
    )

    adx = safe_float(
        row["adx"]
    )

    plus_di = safe_float(
        row["plus_di"]
    )

    minus_di = safe_float(
        row["minus_di"]
    )

    volume = safe_float(
        row["volume"]
    )

    volume_sma = safe_float(
        row["volume_sma"]
    )

    if atr <= 0:
        return None

    score = 0
    reasons = []
    warnings = []

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    score += 2

    if direction == "LONG":

        reasons.append(
            "M15 bullish breakout confirmed"
        )

    else:

        reasons.append(
            "M15 bearish breakdown confirmed"
        )

    # --------------------------------------------------------
    # M15 EMA STRUCTURE
    # --------------------------------------------------------

    if direction == "LONG":

        if ema20 > ema50:

            score += 1

            reasons.append(
                "M15 EMA20 above EMA50"
            )

        else:

            warnings.append(
                "M15 EMA20 below EMA50"
            )

        if ema50 > ema200:

            score += 1

            reasons.append(
                "M15 EMA50 above EMA200"
            )

    else:

        if ema20 < ema50:

            score += 1

            reasons.append(
                "M15 EMA20 below EMA50"
            )

        else:

            warnings.append(
                "M15 EMA20 above EMA50"
            )

        if ema50 < ema200:

            score += 1

            reasons.append(
                "M15 EMA50 below EMA200"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if direction == "LONG":

        if (
            RSI_LONG_MIN
            <= rsi
            < RSI_LONG_MAX
        ):

            score += 1

            reasons.append(
                "M15 RSI supports LONG"
            )

        else:

            warnings.append(
                "M15 RSI not in ideal LONG zone"
            )

    else:

        if (
            RSI_SHORT_MIN
            < rsi
            <= RSI_SHORT_MAX
        ):

            score += 1

            reasons.append(
                "M15 RSI supports SHORT"
            )

        else:

            warnings.append(
                "M15 RSI not in ideal SHORT zone"
            )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx >= STRONG_ADX:

        score += 1

        reasons.append(
            "M15 ADX strong"
        )

    elif adx >= MIN_ADX:

        score += 1

        reasons.append(
            "M15 ADX acceptable"
        )

    else:

        warnings.append(
            "M15 trend strength weak"
        )

    # --------------------------------------------------------
    # DIRECTIONAL INDEX
    # --------------------------------------------------------

    if direction == "LONG":

        if plus_di > minus_di:

            score += 1

            reasons.append(
                "M15 directional bias bullish"
            )

        else:

            warnings.append(
                "M15 directional bias not bullish"
            )

    else:

        if minus_di > plus_di:

            score += 1

            reasons.append(
                "M15 directional bias bearish"
            )

        else:

            warnings.append(
                "M15 directional bias not bearish"
            )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ratio = 0.0

    if volume_sma > 0:

        volume_ratio = (
            volume / volume_sma
        )

    if volume_ratio >= STRONG_VOLUME_RATIO:

        score += 1

        reasons.append(
            "M15 breakout volume strong"
        )

    elif volume_ratio >= MIN_VOLUME_RATIO:

        score += 1

        reasons.append(
            "M15 volume acceptable"
        )

    else:

        warnings.append(
            "M15 breakout volume weak"
        )

    # --------------------------------------------------------
    # CANDLE QUALITY
    # --------------------------------------------------------

    if candle_quality(
        row,
        direction,
    ):

        score += 1

        reasons.append(
            "M15 confirmation candle strong"
        )

    else:

        warnings.append(
            "M15 confirmation candle weak"
        )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if momentum_confirmation(
        df,
        direction,
    ):

        score += 1

        reasons.append(
            "M15 momentum confirmed"
        )

    else:

        warnings.append(
            "M15 momentum not confirmed"
        )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    if price_action_direction(
        df,
        direction,
    ):

        score += 1

        reasons.append(
            "M15 price action aligned"
        )

    # --------------------------------------------------------
    # EMA20 DISTANCE
    # --------------------------------------------------------

    distance_atr = (
        abs(price - ema20)
        / atr
    )

    extended = (
        distance_atr
        > MAX_ENTRY_DISTANCE_ATR
    )

    if not extended:

        reasons.append(
            "Entry distance acceptable"
        )

    else:

        warnings.append(
            f"Price extended "
            f"{distance_atr:.2f} ATR "
            f"from EMA20"
        )

    # --------------------------------------------------------
    # HARD FILTERS
    # --------------------------------------------------------

    hard_invalid = False

    # Price too extended.
    if extended:

        hard_invalid = True

    # RSI overextended.
    if direction == "LONG":

        if rsi >= RSI_LONG_MAX:
            hard_invalid = True

    else:

        if rsi <= RSI_SHORT_MIN:
            hard_invalid = True

    # Directional index must agree.
    if direction == "LONG":

        if plus_di <= minus_di:
            hard_invalid = True

    else:

        if minus_di <= plus_di:
            hard_invalid = True

    # ADX must show at least some trend strength.
    if adx < MIN_ADX:

        hard_invalid = True

    # Volume must not be extremely weak.
    if volume_ratio < MIN_VOLUME_RATIO:

        hard_invalid = True

    # Confirmation candle must be valid.
    if not candle_quality(
        row,
        direction,
    ):

        hard_invalid = True

    # Momentum must confirm breakout.
    if not momentum_confirmation(
        df,
        direction,
    ):

        hard_invalid = True

    # --------------------------------------------------------
    # GRADE
    # --------------------------------------------------------

    if hard_invalid:

        grade = "NO_TRADE"

    elif score >= A_PLUS_SCORE:

        grade = "A+"

    elif score >= A_SCORE:

        grade = "A"

    else:

        grade = "NO_TRADE"

    # --------------------------------------------------------
    # SETUP ID
    # --------------------------------------------------------

    setup_id = (
        f"{breakout['type']}:"
        f"{int(breakout['level'] * 1000000)}"
    )

    return {
        "direction": direction,
        "score": score,
        "grade": grade,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "volume_ratio": volume_ratio,
        "distance_atr": distance_atr,
        "extended": extended,
        "structure": breakout["type"],
        "breakout_level": breakout["level"],
        "candle_timestamp": breakout["candle_timestamp"],
        "setup_id": setup_id,
        "reasons": reasons,
        "warnings": warnings,
    }


# ============================================================
# TRADE MAP
# ============================================================

def build_trade_map(result):

    price = result["price"]
    atr = result["atr"]

    direction = result["direction"]

    if direction == "LONG":

        entry_low = (
            price - atr * 0.15
        )

        entry_high = (
            price + atr * 0.10
        )

        invalidation = (
            price - atr
            * INVALIDATION_ATR_MULTIPLIER
        )

        tp1 = (
            price + atr * 1.50
        )

        tp2 = (
            price + atr * 2.50
        )

        tp3 = (
            price + atr * 4.00
        )

    else:

        entry_low = (
            price - atr * 0.10
        )

        entry_high = (
            price + atr * 0.15
        )

        invalidation = (
            price + atr
            * INVALIDATION_ATR_MULTIPLIER
        )

        tp1 = (
            price - atr * 1.50
        )

        tp2 = (
            price - atr * 2.50
        )

        tp3 = (
            price - atr * 4.00
        )

    risk = abs(
        price - invalidation
    )

    reward = abs(
        tp1 - price
    )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "invalidation": invalidation,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr,
    }


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(
    title,
    message,
    priority="default",
    tags=None,
):

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
        "Content-Type": (
            "text/plain; charset=utf-8"
        ),
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

        async with httpx.AsyncClient(
            timeout=15
        ) as client:

            response = await client.post(
                NTFY_URL,
                content=(
                    str(message)
                    .encode("utf-8")
                ),
                headers=headers,
            )

            response.raise_for_status()

            logger.info(
                "NTFY sent: %s",
                clean_title,
            )

    except Exception as exc:

        logger.error(
            "NTFY error: %s",
            exc,
        )


# ============================================================
# MESSAGE FORMAT
# ============================================================

def format_signal(
    symbol,
    h1_trend,
    result,
    trade_map,
):

    reasons = "\n".join(
        f"- {item}"
        for item in result["reasons"][:12]
    )

    warnings = "\n".join(
        f"- {item}"
        for item in result["warnings"][:8]
    )

    if not warnings:
        warnings = "- None"

    return (
        f"{symbol} - "
        f"{result['direction']} SIGNAL\n\n"

        f"Grade: {result['grade']}\n"
        f"Score: {result['score']}\n\n"

        f"H1 Trend: {h1_trend}\n"
        f"M15 Structure: "
        f"{result['structure']}\n\n"

        f"Price: "
        f"{fmt_price(result['price'])}\n"

        f"RSI: "
        f"{result['rsi']:.1f}\n"

        f"ADX: "
        f"{result['adx']:.1f}\n"

        f"Volume Ratio: "
        f"{result['volume_ratio']:.2f}x\n"

        f"EMA20 Distance: "
        f"{result['distance_atr']:.2f} ATR\n\n"

        f"CONFIRMATIONS\n"
        f"{reasons}\n\n"

        f"WARNINGS\n"
        f"{warnings}\n\n"

        f"ENTRY ZONE\n"
        f"{fmt_price(trade_map['entry_low'])}"
        f" - "
        f"{fmt_price(trade_map['entry_high'])}\n\n"

        f"INVALIDATION\n"
        f"{fmt_price(trade_map['invalidation'])}\n\n"

        f"TARGETS\n"
        f"TP1: {fmt_price(trade_map['tp1'])}\n"
        f"TP2: {fmt_price(trade_map['tp2'])}\n"
        f"TP3: {fmt_price(trade_map['tp3'])}\n\n"

        f"R:R to TP1: "
        f"{trade_map['rr']:.2f}\n\n"

        f"Action: "
        f"VALID SIGNAL"
    )


def format_invalidation(
    symbol,
    setup,
):

    return (
        f"{symbol} SETUP INVALIDATED\n\n"

        f"Direction: "
        f"{setup['direction']}\n"

        f"Previous Grade: "
        f"{setup['grade']}\n"

        f"Previous Score: "
        f"{setup['score']}\n\n"

        f"M15 Structure: "
        f"{setup['structure']}\n\n"

        f"Action: NO TRADE"
    )


# ============================================================
# SETUP ID
# ============================================================

def build_setup_identity(
    breakout,
):

    level = safe_float(
        breakout["level"]
    )

    direction = (
        "LONG"
        if breakout["type"]
        == "BULL_BREAKOUT"
        else "SHORT"
    )

    return (
        f"{direction}:"
        f"{breakout['type']}:"
        f"{level:.12f}"
    )


# ============================================================
# ACTIVE SETUP VALIDATION
# ============================================================

def setup_is_expired(
    setup,
):

    created_at = setup["created_at"]

    elapsed = (
        now_utc() - created_at
    ).total_seconds() / 3600

    return (
        elapsed
        >= SETUP_EXPIRY_HOURS
    )


def setup_is_invalidated(
    df,
    setup,
):

    row = df.iloc[-1]

    close = safe_float(
        row["close"]
    )

    invalidation = safe_float(
        setup["invalidation"]
    )

    direction = setup["direction"]

    if direction == "LONG":

        return close <= invalidation

    return close >= invalidation


async def monitor_active_setup(
    symbol,
    h1_trend,
    m15,
):

    setup = active_setups.get(
        symbol
    )

    if not setup:
        return False

    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    if setup_is_expired(setup):

        logger.info(
            "%s active setup expired",
            symbol,
        )

        invalidated_setups[
            symbol
        ] = {
            "setup_id": setup["setup_id"],
            "direction": setup["direction"],
            "candle_timestamp": setup[
                "candle_timestamp"
            ],
        }

        active_setups.pop(
            symbol,
            None,
        )

        return False

    # --------------------------------------------------------
    # H1 DIRECTION CHANGED
    # --------------------------------------------------------

    if (
        setup["direction"] == "LONG"
        and h1_trend != "BULLISH"
    ):

        await send_ntfy(
            f"{symbol} SETUP INVALIDATED",
            format_invalidation(
                symbol,
                setup,
            ),
            priority="high",
            tags="warning",
        )

        invalidated_setups[
            symbol
        ] = {
            "setup_id": setup["setup_id"],
            "direction": setup["direction"],
            "candle_timestamp": setup[
                "candle_timestamp"
            ],
        }

        active_setups.pop(
            symbol,
            None,
        )

        return False

    if (
        setup["direction"] == "SHORT"
        and h1_trend != "BEARISH"
    ):

        await send_ntfy(
            f"{symbol} SETUP INVALIDATED",
            format_invalidation(
                symbol,
                setup,
            ),
            priority="high",
            tags="warning",
        )

        invalidated_setups[
            symbol
        ] = {
            "setup_id": setup["setup_id"],
            "direction": setup["direction"],
            "candle_timestamp": setup[
                "candle_timestamp"
            ],
        }

        active_setups.pop(
            symbol,
            None,
        )

        return False

    # --------------------------------------------------------
    # PRICE INVALIDATION
    # --------------------------------------------------------

    if setup_is_invalidated(
        m15,
        setup,
    ):

        await send_ntfy(
            f"{symbol} SETUP INVALIDATED",
            format_invalidation(
                symbol,
                setup,
            ),
            priority="high",
            tags="warning",
        )

        invalidated_setups[
            symbol
        ] = {
            "setup_id": setup["setup_id"],
            "direction": setup["direction"],
            "candle_timestamp": setup[
                "candle_timestamp"
            ],
        }

        active_setups.pop(
            symbol,
            None,
        )

        return False

    # --------------------------------------------------------
    # SAME ACTIVE SETUP
    # --------------------------------------------------------

    return True


# ============================================================
# NEW SETUP CHECK
# ============================================================

def setup_already_seen(
    symbol,
    setup_id,
):

    active = active_setups.get(
        symbol
    )

    if active:

        if (
            active["setup_id"]
            == setup_id
        ):

            return True

    invalidated = invalidated_setups.get(
        symbol
    )

    if invalidated:

        if (
            invalidated["setup_id"]
            == setup_id
        ):

            return True

    return False


# ============================================================
# SIGNAL ENGINE
# ============================================================

async def analyze_symbol(
    symbol
):

    # --------------------------------------------------------
    # FETCH H1
    # --------------------------------------------------------

    h1 = await fetch_ohlcv(
        symbol,
        TREND_TIMEFRAME,
        H1_LIMIT,
    )

    # --------------------------------------------------------
    # FETCH M15
    # --------------------------------------------------------

    m15 = await fetch_ohlcv(
        symbol,
        SIGNAL_TIMEFRAME,
        M15_LIMIT,
    )

    if h1 is None or m15 is None:

        logger.warning(
            "No data: %s",
            symbol,
        )

        return

    # --------------------------------------------------------
    # REMOVE OPEN CANDLES
    # --------------------------------------------------------

    h1 = remove_open_candle(
        h1
    )

    m15 = remove_open_candle(
        m15
    )

    if (
        len(h1) < 210
        or len(m15) < 210
    ):

        logger.warning(
            "Not enough data: %s | H1=%d M15=%d",
            symbol,
            len(h1),
            len(m15),
        )

        return

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    h1 = add_indicators(
        h1
    )

    m15 = add_indicators(
        m15
    )

    # --------------------------------------------------------
    # H1 TREND
    # --------------------------------------------------------

    h1_trend = detect_h1_trend(
        h1
    )

    # --------------------------------------------------------
    # LAST CLOSED M15 CANDLE
    # --------------------------------------------------------

    current_candle = m15.iloc[-1]

    candle_timestamp = (
        current_candle["timestamp"]
    )

    # --------------------------------------------------------
    # ONLY PROCESS A NEW CLOSED M15 CANDLE
    #
    # The bot scans every 60 seconds, but the same
    # closed M15 candle is processed only once.
    # --------------------------------------------------------

    previous_timestamp = (
        last_processed_candle.get(
            symbol
        )
    )

    if (
        previous_timestamp
        == candle_timestamp
    ):

        return

    last_processed_candle[
        symbol
    ] = candle_timestamp

    logger.info(
        "%s | H1=%s | New M15 candle=%s",
        symbol,
        h1_trend,
        candle_timestamp,
    )

    # --------------------------------------------------------
    # MONITOR EXISTING SETUP
    # --------------------------------------------------------

    had_active_setup = (
        symbol in active_setups
    )

    if had_active_setup:

        still_active = (
            await monitor_active_setup(
                symbol,
                h1_trend,
                m15,
            )
        )

        if still_active:

            # Existing setup is still active.
            # DO NOT generate another signal.
            logger.info(
                "%s | Existing setup still active",
                symbol,
            )

            return

    # --------------------------------------------------------
    # H1 MUST HAVE A CLEAR TREND
    # --------------------------------------------------------

    if h1_trend not in {
        "BULLISH",
        "BEARISH",
    }:

        logger.info(
            "%s | H1 neutral | NO TRADE",
            symbol,
        )

        return

    # --------------------------------------------------------
    # DETERMINE ALLOWED DIRECTION
    # --------------------------------------------------------

    direction = (
        "LONG"
        if h1_trend == "BULLISH"
        else "SHORT"
    )

    # --------------------------------------------------------
    # LOOK FOR A BRAND NEW M15 BREAKOUT
    # --------------------------------------------------------

    breakout = detect_m15_breakout(
        m15,
        direction,
    )

    if not breakout:

        logger.info(
            "%s | H1=%s | No new M15 setup",
            symbol,
            h1_trend,
        )

        return

    # --------------------------------------------------------
    # BUILD UNIQUE SETUP ID
    # --------------------------------------------------------

    setup_id = build_setup_identity(
        breakout
    )

    # --------------------------------------------------------
    # NEVER REUSE THE SAME SETUP
    # --------------------------------------------------------

    if setup_already_seen(
        symbol,
        setup_id,
    ):

        logger.info(
            "%s | Same setup already processed",
            symbol,
        )

        return

    # --------------------------------------------------------
    # ANALYZE NEW M15 SETUP
    # --------------------------------------------------------

    result = analyze_m15_setup(
        m15,
        direction,
        breakout,
    )

    if result is None:

        return

    # --------------------------------------------------------
    # ONLY A / A+ ARE REAL SIGNALS
    #
    # No WATCH notification.
    # No weak setup notification.
    # --------------------------------------------------------

    if result["grade"] not in {
        "A",
        "A+",
    }:

        logger.info(
            "%s | New M15 setup rejected | "
            "Direction=%s Score=%d Grade=%s",
            symbol,
            direction,
            result["score"],
            result["grade"],
        )

        return

    # --------------------------------------------------------
    # TRADE MAP
    # --------------------------------------------------------

    trade_map = build_trade_map(
        result
    )

    # --------------------------------------------------------
    # CREATE ACTIVE SETUP
    # --------------------------------------------------------

    active_setups[
        symbol
    ] = {
        "setup_id": result["setup_id"],
        "direction": result["direction"],
        "score": result["score"],
        "grade": result["grade"],
        "structure": result["structure"],
        "price": result["price"],
        "invalidation": trade_map[
            "invalidation"
        ],
        "created_at": now_utc(),
        "candle_timestamp": result[
            "candle_timestamp"
        ],
    }

    # --------------------------------------------------------
    # SEND SIGNAL ONCE
    # --------------------------------------------------------

    title = (
        f"{symbol} "
        f"{result['direction']} "
        f"{result['grade']}"
    )

    message = format_signal(
        symbol,
        h1_trend,
        result,
        trade_map,
    )

    tags = (
        "chart_with_upwards_trend"
        if direction == "LONG"
        else "chart_with_downwards_trend"
    )

    await send_ntfy(
        title,
        message,
        priority="high",
        tags=tags,
    )

    logger.info(
        "%s | NEW SIGNAL | %s | %s | Score=%d",
        symbol,
        direction,
        result["grade"],
        result["score"],
    )


# ============================================================
# MARKET SCAN
# ============================================================

async def scan_market():

    logger.info(
        "Starting market scan..."
    )

    for symbol in SYMBOLS:

        try:

            await analyze_symbol(
                symbol
            )

        except Exception as exc:

            logger.exception(
                "Analysis error %s: %s",
                symbol,
                exc,
            )

        await asyncio.sleep(
            0.25
        )

    logger.info(
        "Market scan completed"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "AH1 TRADING ASSISTANT STARTING"
    )

    logger.info(
        "Trend timeframe: %s",
        TREND_TIMEFRAME,
    )

    logger.info(
        "Signal timeframe: %s",
        SIGNAL_TIMEFRAME,
    )

    logger.info(
        "Symbols: %d",
        len(SYMBOLS),
    )

    logger.info(
        "Auto trading: DISABLED"
    )

    logger.info(
        "NTFY: ENABLED"
    )

    logger.info(
        "Signal mode: NEW SETUP ONLY"
    )

    logger.info(
        "WATCH alerts: DISABLED"
    )

    logger.info(
        "=========================================="
    )

    while True:

        try:

            await scan_market()

        except Exception as exc:

            logger.exception(
                "Scanner error: %s",
                exc,
            )

        await asyncio.sleep(
            SCAN_INTERVAL
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
            "Bot stopped."
        )

    finally:

        try:

            asyncio.run(
                exchange.close()
            )

        except Exception:
            pass
