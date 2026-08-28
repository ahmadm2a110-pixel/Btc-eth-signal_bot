import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd


# ============================================================
# AH1 ELITE v3
# H1 STRUCTURE -> BREAKOUT -> CANDLE 2 CONFIRMATION
# -> M15 PULLBACK / RETEST -> SCORE -> SIGNAL
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

EXCHANGE_ID = "kcex"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "ADA/USDT",
]

H1_TIMEFRAME = "1h"
M15_TIMEFRAME = "15m"

H1_LIMIT = 180
M15_LIMIT = 250

# Range Box
BOX_LOOKBACK = 80
BOX_TOUCH_TOLERANCE = 0.0025       # 0.25%
MIN_BOX_TOUCHES = 2

# Swing detection
SWING_LEFT = 2
SWING_RIGHT = 2

# Breakout / retest
BREAKOUT_MIN_DISTANCE_ATR = 0.0    # Any valid close beyond level is accepted
RETEST_ATR_TOLERANCE = 0.35         # M15 retest zone
MAX_M15_WAIT_CANDLES = 32           # 8 hours

# Strong breakout alert
STRONG_BODY_RATIO = 0.70
STRONG_WICK_MAX_RATIO = 0.15

# Signal score
A_PLUS_SCORE = 85
A_SCORE = 70
MIN_SIGNAL_SCORE = 70

# Risk
RISK_REWARD = 2.0
SL_ATR_BUFFER = 0.30

# Scan interval
SCAN_INTERVAL_SECONDS = 60

# NTFY
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
# STATE
# ============================================================

SETUPS = {}
LAST_SIGNAL_CANDLE = {}


# ============================================================
# HTTP
# ============================================================

http_client = None


async def send_ntfy(title, message, priority="default"):
    try:
        await http_client.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            content=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": "chart_with_upwards_trend",
            },
            timeout=10,
        )
    except Exception as e:
        logger.error(f"NTFY error: {e}")


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def candle_body(c):
    return abs(c["close"] - c["open"])


def candle_range(c):
    return max(c["high"] - c["low"], 1e-12)


def body_ratio(c):
    return candle_body(c) / candle_range(c)


def upper_wick(c):
    return c["high"] - max(c["open"], c["close"])


def lower_wick(c):
    return min(c["open"], c["close"]) - c["low"]


def is_bullish(c):
    return c["close"] > c["open"]


def is_bearish(c):
    return c["close"] < c["open"]


def is_strong_bullish(c):
    r = candle_range(c)

    return (
        is_bullish(c)
        and body_ratio(c) >= STRONG_BODY_RATIO
        and upper_wick(c) / r <= STRONG_WICK_MAX_RATIO
        and lower_wick(c) / r <= STRONG_WICK_MAX_RATIO
    )


def is_strong_bearish(c):
    r = candle_range(c)

    return (
        is_bearish(c)
        and body_ratio(c) >= STRONG_BODY_RATIO
        and upper_wick(c) / r <= STRONG_WICK_MAX_RATIO
        and lower_wick(c) / r <= STRONG_WICK_MAX_RATIO
    )


def pct(value):
    return f"{value:.2f}%"


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI 14
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands 20 / 2
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # ATR 14
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # Volume average
    df["volume_sma"] = volume.rolling(20).mean()

    return df


# ============================================================
# SWING DETECTION
# ============================================================

def find_swing_highs(df):
    highs = []

    for i in range(SWING_LEFT, len(df) - SWING_RIGHT):
        value = df.iloc[i]["high"]

        left = df.iloc[
            i - SWING_LEFT:i
        ]["high"]

        right = df.iloc[
            i + 1:i + 1 + SWING_RIGHT
        ]["high"]

        if value > left.max() and value >= right.max():
            highs.append({
                "index": i,
                "price": value,
                "time": df.iloc[i]["timestamp"],
            })

    return highs


def find_swing_lows(df):
    lows = []

    for i in range(SWING_LEFT, len(df) - SWING_RIGHT):
        value = df.iloc[i]["low"]

        left = df.iloc[
            i - SWING_LEFT:i
        ]["low"]

        right = df.iloc[
            i + 1:i + 1 + SWING_RIGHT
        ]["low"]

        if value < left.min() and value <= right.min():
            lows.append({
                "index": i,
                "price": value,
                "time": df.iloc[i]["timestamp"],
            })

    return lows


# ============================================================
# TRENDLINE DETECTION
# ============================================================

def get_downtrend_line(df):
    """
    LONG setup:
    Need at least two valid Lower Highs.
    The most recent two LHs define the descending trendline.
    """

    highs = find_swing_highs(df)

    if len(highs) < 2:
        return None

    # Work from recent swings
    recent = highs[-8:]

    for i in range(len(recent) - 2, -1, -1):
        p1 = recent[i]
        p2 = recent[i + 1]

        if p2["price"] >= p1["price"]:
            continue

        if p2["index"] <= p1["index"]:
            continue

        # Need some bearish structure between/around points
        lows = find_swing_lows(df)

        if not lows:
            continue

        # Check that market has produced a lower low somewhere
        lower_low_exists = any(
            x["price"] < p1["price"]
            for x in lows
            if x["index"] > p1["index"]
        )

        if not lower_low_exists:
            continue

        slope = (
            p2["price"] - p1["price"]
        ) / (
            p2["index"] - p1["index"]
        )

        if slope >= 0:
            continue

        return {
            "type": "downtrend",
            "p1": p1,
            "p2": p2,
            "slope": slope,
        }

    return None


def get_uptrend_line(df):
    """
    SHORT setup:
    Need at least two valid Higher Lows.
    """

    lows = find_swing_lows(df)

    if len(lows) < 2:
        return None

    recent = lows[-8:]

    for i in range(len(recent) - 2, -1, -1):
        p1 = recent[i]
        p2 = recent[i + 1]

        if p2["price"] <= p1["price"]:
            continue

        if p2["index"] <= p1["index"]:
            continue

        highs = find_swing_highs(df)

        if not highs:
            continue

        higher_high_exists = any(
            x["price"] > p1["price"]
            for x in highs
            if x["index"] > p1["index"]
        )

        if not higher_high_exists:
            continue

        slope = (
            p2["price"] - p1["price"]
        ) / (
            p2["index"] - p1["index"]
        )

        if slope <= 0:
            continue

        return {
            "type": "uptrend",
            "p1": p1,
            "p2": p2,
            "slope": slope,
        }

    return None


def trendline_value(line, index):
    p1 = line["p1"]

    return p1["price"] + line["slope"] * (
        index - p1["index"]
    )


# ============================================================
# RANGE BOX
# ============================================================

def find_range_box(df):
    """
    Valid range:
    At least 2 meaningful touches on resistance
    and 2 meaningful touches on support.
    """

    if len(df) < BOX_LOOKBACK:
        return None

    recent = df.iloc[-BOX_LOOKBACK:].copy()

    highs = find_swing_highs(recent)
    lows = find_swing_lows(recent)

    if len(highs) < 2 or len(lows) < 2:
        return None

    high_prices = [x["price"] for x in highs]
    low_prices = [x["price"] for x in lows]

    resistance = max(
        high_prices[-6:]
    )

    support = min(
        low_prices[-6:]
    )

    if resistance <= support:
        return None

    height = resistance - support

    # Reject tiny boxes
    if height <= 0:
        return None

    resistance_touches = 0
    support_touches = 0

    for x in highs:
        if abs(x["price"] - resistance) / resistance <= BOX_TOUCH_TOLERANCE:
            resistance_touches += 1

    for x in lows:
        if abs(x["price"] - support) / support <= BOX_TOUCH_TOLERANCE:
            support_touches += 1

    if resistance_touches < MIN_BOX_TOUCHES:
        return None

    if support_touches < MIN_BOX_TOUCHES:
        return None

    return {
        "type": "range",
        "resistance": resistance,
        "support": support,
        "resistance_touches": resistance_touches,
        "support_touches": support_touches,
    }


# ============================================================
# H1 BREAKOUT DETECTION
# ============================================================

def detect_h1_breakout(df):
    """
    Returns a breakout only from the most recently CLOSED H1 candle.
    """

    if len(df) < 30:
        return None

    # Last row must be the latest CLOSED candle
    c1 = df.iloc[-1]
    previous = df.iloc[:-1]

    current_index = len(df) - 1

    candidates = []

    # --------------------------------------------------------
    # DOWN TRENDLINE BREAK -> LONG
    # --------------------------------------------------------

    downline = get_downtrend_line(previous)

    if downline:
        level = trendline_value(
            downline,
            current_index,
        )

        if c1["close"] > level:
            candidates.append({
                "direction": "LONG",
                "type": "trendline",
                "level": level,
                "candle": c1,
                "structure": "H1_DOWNTRENDLINE_BROKEN",
            })

    # --------------------------------------------------------
    # UPTRENDLINE BREAK -> SHORT
    # --------------------------------------------------------

    upline = get_uptrend_line(previous)

    if upline:
        level = trendline_value(
            upline,
            current_index,
        )

        if c1["close"] < level:
            candidates.append({
                "direction": "SHORT",
                "type": "trendline",
                "level": level,
                "candle": c1,
                "structure": "H1_UPTRENDLINE_BROKEN",
            })

    # --------------------------------------------------------
    # RANGE BREAK
    # --------------------------------------------------------

    box = find_range_box(previous)

    if box:

        if c1["close"] > box["resistance"]:
            candidates.append({
                "direction": "LONG",
                "type": "range",
                "level": box["resistance"],
                "candle": c1,
                "structure": "H1_RANGE_RESISTANCE_BROKEN",
            })

        elif c1["close"] < box["support"]:
            candidates.append({
                "direction": "SHORT",
                "type": "range",
                "level": box["support"],
                "candle": c1,
                "structure": "H1_RANGE_SUPPORT_BROKEN",
            })

    if not candidates:
        return None

    # Prefer the most direct/latest candidate
    return candidates[-1]


# ============================================================
# CANDLE 2 CONFIRMATION
# ============================================================

def confirm_breakout(h1_df, breakout):
    """
    Candle 1 = breakout candle.
    Candle 2 decides whether breakout is confirmed.

    Rules:
    - Strong same-direction Candle 2 = excellent confirmation.
    - Same-direction close beyond level + body surpasses Candle 1 = confirmed.
    - If Candle 2 tests back but closes back beyond level without enough
      body strength, wait for Candle 3.
    - If Candle 2 closes clearly back inside the old area = fake breakout.
    """

    if len(h1_df) < 2:
        return None

    c1 = breakout["candle"]
    c2 = h1_df.iloc[-1]

    level = breakout["level"]
    direction = breakout["direction"]

    body1_high = max(c1["open"], c1["close"])
    body1_low = min(c1["open"], c1["close"])

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        if c2["close"] < level:
            return {
                "status": "FAKE",
                "candle": c2,
            }

        # Excellent
        if is_strong_bullish(c2):
            return {
                "status": "CONFIRMED",
                "quality": "STRONG",
                "candle": c2,
            }

        # Same direction and body surpasses Candle 1
        if (
            is_bullish(c2)
            and c2["close"] > level
            and c2["close"] > body1_high
        ):
            return {
                "status": "CONFIRMED",
                "quality": "NORMAL",
                "candle": c2,
            }

        # Candle 2 retested but recovered above level.
        # Not strong enough -> wait for Candle 3.
        if c2["close"] >= level:
            return {
                "status": "WAIT",
                "candle": c2,
            }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if direction == "SHORT":

        if c2["close"] > level:
            return {
                "status": "FAKE",
                "candle": c2,
            }

        if is_strong_bearish(c2):
            return {
                "status": "CONFIRMED",
                "quality": "STRONG",
                "candle": c2,
            }

        if (
            is_bearish(c2)
            and c2["close"] < level
            and c2["close"] < body1_low
        ):
            return {
                "status": "CONFIRMED",
                "quality": "NORMAL",
                "candle": c2,
            }

        if c2["close"] <= level:
            return {
                "status": "WAIT",
                "candle": c2,
            }

    return {
        "status": "WAIT",
        "candle": c2,
    }


# ============================================================
# STRONG BREAKOUT ALERT
# ============================================================

async def strong_breakout_alert(symbol, breakout):
    candle = breakout["candle"]

    strength = body_ratio(candle) * 100

    if breakout["direction"] == "LONG":
        strong = is_strong_bullish(candle)
    else:
        strong = is_strong_bearish(candle)

    if not strong:
        return

    if breakout["type"] == "range":
        structure = (
            "Valid H1 Range Box Resistance Broken"
            if breakout["direction"] == "LONG"
            else
            "Valid H1 Range Box Support Broken"
        )
    else:
        structure = (
            "H1 Bearish Trendline Broken"
            if breakout["direction"] == "LONG"
            else
            "H1 Bullish Trendline Broken"
        )

    emoji = "🟢" if breakout["direction"] == "LONG" else "🔴"

    message = f"""
{emoji} {symbol} — STRONG H1 BREAKOUT

{structure}

Candle Body:
{strength:.1f}%

H1 Close:
{candle["close"]:.6f}

⚠️ NO SIGNAL YET

Candle 2 Confirmation Required.

🔎 Setup is under watch.
M15 Pullback / Retest will be monitored.

AH1 ELITE v3
""".strip()

    await send_ntfy(
        f"{symbol} — STRONG H1 BREAKOUT",
        message,
        "high",
    )


# ============================================================
# CREATE SETUP
# ============================================================

def create_setup(symbol, breakout):
    return {
        "symbol": symbol,
        "direction": breakout["direction"],
        "breakout_type": breakout["type"],
        "level": float(breakout["level"]),
        "breakout_time": breakout["candle"]["timestamp"],
        "status": "WAIT_M15",
        "created_at": utc_now().isoformat(),
    }


# ============================================================
# M15 RETEST
# ============================================================

def find_m15_retest(m15_df, setup):
    """
    Find:
    Pullback -> touch nearby breakout level -> rejection candle.

    No HL/LH requirement.
    """

    if len(m15_df) < 20:
        return None

    level = setup["level"]
    direction = setup["direction"]

    breakout_time = setup["breakout_time"]

    data = m15_df[
        m15_df["timestamp"] > breakout_time
    ].copy()

    if data.empty:
        return None

    data = data.tail(MAX_M15_WAIT_CANDLES)

    for i in range(1, len(data)):
        c = data.iloc[i]

        atr = c["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        tolerance = atr * RETEST_ATR_TOLERANCE

        zone_low = level - tolerance
        zone_high = level + tolerance

        touched = (
            c["low"] <= zone_high
            and c["high"] >= zone_low
        )

        if not touched:
            continue

        # ----------------------------------------------------
        # LONG RETEST
        # ----------------------------------------------------

        if direction == "LONG":

            if c["close"] <= level:
                continue

            if not is_bullish(c):
                continue

            # Rejection / recovery candle
            if c["close"] > c["open"]:

                return {
                    "status": "CONFIRMED",
                    "candle": c,
                    "level": level,
                }

        # ----------------------------------------------------
        # SHORT RETEST
        # ----------------------------------------------------

        if direction == "SHORT":

            if c["close"] >= level:
                continue

            if not is_bearish(c):
                continue

            if c["close"] < c["open"]:

                return {
                    "status": "CONFIRMED",
                    "candle": c,
                    "level": level,
                }

    return None


# ============================================================
# SCORE
# ============================================================

def calculate_score(m15_df, retest):
    """
    100 points:

    RSI       25
    Volume    25
    Bollinger 20
    ATR       15
    Candle    15
    """

    c = retest["candle"]
    direction = retest["direction"] if "direction" in retest else None

    score = 0
    details = []

    # If direction isn't attached, infer from candle/level externally
    if direction is None:
        direction = "LONG" if c["close"] > c["open"] else "SHORT"

    # --------------------------------------------------------
    # RSI — 25
    # --------------------------------------------------------

    rsi = c["rsi"]

    if not pd.isna(rsi):

        if direction == "LONG":

            if rsi >= 55:
                score += 25
                details.append("RSI_STRONG")
            elif rsi >= 50:
                score += 18
                details.append("RSI_POSITIVE")
            elif rsi >= 45:
                score += 10
                details.append("RSI_NEUTRAL")

        else:

            if rsi <= 45:
                score += 25
                details.append("RSI_STRONG")
            elif rsi <= 50:
                score += 18
                details.append("RSI_NEGATIVE")
            elif rsi <= 55:
                score += 10
                details.append("RSI_NEUTRAL")

    # --------------------------------------------------------
    # VOLUME — 25
    # --------------------------------------------------------

    if (
        not pd.isna(c["volume_sma"])
        and c["volume_sma"] > 0
    ):

        volume_ratio = c["volume"] / c["volume_sma"]

        if volume_ratio >= 1.50:
            score += 25
            details.append("VOLUME_STRONG")

        elif volume_ratio >= 1.20:
            score += 18
            details.append("VOLUME_GOOD")

        elif volume_ratio >= 1.00:
            score += 10
            details.append("VOLUME_NORMAL")

    # --------------------------------------------------------
    # BOLLINGER — 20
    # --------------------------------------------------------

    if not pd.isna(c["bb_mid"]):

        if direction == "LONG":

            if c["close"] >= c["bb_mid"]:
                score += 20
                details.append("BB_BULLISH")

            elif c["close"] >= c["bb_lower"]:
                score += 10
                details.append("BB_NEUTRAL")

        else:

            if c["close"] <= c["bb_mid"]:
                score += 20
                details.append("BB_BEARISH")

            elif c["close"] <= c["bb_upper"]:
                score += 10
                details.append("BB_NEUTRAL")

    # --------------------------------------------------------
    # ATR — 15
    # --------------------------------------------------------

    if not pd.isna(c["atr"]) and c["close"] > 0:

        atr_percent = (
            c["atr"] / c["close"]
        ) * 100

        if atr_percent >= 0.50:
            score += 15
            details.append("ATR_ACTIVE")

        elif atr_percent >= 0.25:
            score += 10
            details.append("ATR_OK")

        else:
            score += 5
            details.append("ATR_LOW")

    # --------------------------------------------------------
    # M15 CANDLE — 15
    # --------------------------------------------------------

    br = body_ratio(c)

    if direction == "LONG":

        if is_bullish(c) and br >= 0.70:
            score += 15
            details.append("M15_CANDLE_STRONG")

        elif is_bullish(c) and br >= 0.50:
            score += 10
            details.append("M15_CANDLE_GOOD")

        elif is_bullish(c):
            score += 5
            details.append("M15_CANDLE_WEAK")

    else:

        if is_bearish(c) and br >= 0.70:
            score += 15
            details.append("M15_CANDLE_STRONG")

        elif is_bearish(c) and br >= 0.50:
            score += 10
            details.append("M15_CANDLE_GOOD")

        elif is_bearish(c):
            score += 5
            details.append("M15_CANDLE_WEAK")

    return min(score, 100), details


# ============================================================
# SCORE GRADE
# ============================================================

def score_grade(score):

    if score >= A_PLUS_SCORE:
        return "A+"

    if score >= A_SCORE:
        return "A"

    if score >= 60:
        return "B"

    return "NO TRADE"


# ============================================================
# SIGNAL
# ============================================================

async def send_signal(symbol, setup, retest, score, details):

    grade = score_grade(score)

    if score < MIN_SIGNAL_SCORE:
        logger.info(
            f"{symbol} | Retest confirmed but score={score} | NO TRADE"
        )
        return

    direction = setup["direction"]

    entry = float(retest["candle"]["close"])
    atr = float(retest["candle"]["atr"])

    if direction == "LONG":

        stop = float(
            retest["candle"]["low"] - atr * SL_ATR_BUFFER
        )

        risk = entry - stop

        if risk <= 0:
            return

        take_profit = entry + risk * RISK_REWARD

        emoji = "🟢"

    else:

        stop = float(
            retest["candle"]["high"] + atr * SL_ATR_BUFFER
        )

        risk = stop - entry

        if risk <= 0:
            return

        take_profit = entry - risk * RISK_REWARD

        emoji = "🔴"

    signal_key = (
        symbol,
        setup["breakout_time"],
        retest["candle"]["timestamp"],
    )

    if signal_key in LAST_SIGNAL_CANDLE:
        return

    LAST_SIGNAL_CANDLE[signal_key] = True

    message = f"""
{emoji} {symbol} — {direction}

Quality:
{grade}

Score:
{score}/100

Setup:
H1 {setup["breakout_type"].upper()} BREAKOUT

Breakout Level:
{setup["level"]:.6f}

Entry:
{entry:.6f}

Stop Loss:
{stop:.6f}

Take Profit:
{take_profit:.6f}

Risk/Reward:
1:2.00

M15:
PULLBACK + RETEST CONFIRMED

Score Factors:
{", ".join(details)}

Signal Candle:
{retest["candle"]["timestamp"]}

AH1 ELITE v3
""".strip()

    await send_ntfy(
        f"{symbol} — {direction} {grade}",
        message,
        "high",
    )

    logger.info(
        f"{symbol} | {direction} | {grade} | Score={score}"
    )


# ============================================================
# PROCESS SETUP
# ============================================================

async def process_existing_setup(symbol, m15_df):

    setup = SETUPS.get(symbol)

    if not setup:
        return

    if setup["status"] != "WAIT_M15":
        return

    retest = find_m15_retest(
        m15_df,
        setup,
    )

    if not retest:
        return

    retest["direction"] = setup["direction"]

    score, details = calculate_score(
        m15_df,
        retest,
    )

    setup["status"] = "SIGNAL_PROCESSED"

    await send_signal(
        symbol,
        setup,
        retest,
        score,
        details,
    )


# ============================================================
# FETCH DATA
# ============================================================

async def fetch_ohlcv(exchange, symbol, timeframe, limit):

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
                "timestamp_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp_ms"],
            unit="ms",
            utc=True,
        )

        # Remove currently forming candle
        if len(df) > 2:
            df = df.iloc[:-1].copy()

        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:

        logger.error(
            f"{symbol} {timeframe} fetch error: {e}"
        )

        return None


# ============================================================
# MAIN SYMBOL SCAN
# ============================================================

async def scan_symbol(exchange, symbol):

    try:

        h1 = await fetch_ohlcv(
            exchange,
            symbol,
            H1_TIMEFRAME,
            H1_LIMIT,
        )

        m15 = await fetch_ohlcv(
            exchange,
            symbol,
            M15_TIMEFRAME,
            M15_LIMIT,
        )

        if h1 is None or m15 is None:
            return

        if len(h1) < 50 or len(m15) < 50:
            return

        h1 = calculate_indicators(h1)
        m15 = calculate_indicators(m15)

        # ----------------------------------------------------
        # FIRST: EXISTING SETUP
        # ----------------------------------------------------

        await process_existing_setup(
            symbol,
            m15,
        )

        # ----------------------------------------------------
        # DON'T CREATE ANOTHER SETUP WHILE ONE IS ACTIVE
        # ----------------------------------------------------

        existing = SETUPS.get(symbol)

        if existing and existing["status"] == "WAIT_M15":
            return

        # ----------------------------------------------------
        # DETECT NEW H1 BREAKOUT
        # ----------------------------------------------------

        breakout = detect_h1_breakout(h1)

        if not breakout:
            return

        breakout_time = breakout["candle"]["timestamp"]

        # Prevent processing same H1 breakout repeatedly
        previous_breakout = (
            existing["breakout_time"]
            if existing
            else None
        )

        if previous_breakout == breakout_time:
            return

        # ----------------------------------------------------
        # STRONG BREAKOUT ALERT
        # ----------------------------------------------------

        await strong_breakout_alert(
            symbol,
            breakout,
        )

        # ----------------------------------------------------
        # CANDLE 2
        # ----------------------------------------------------

        confirmation = confirm_breakout(
            h1,
            breakout,
        )

        if not confirmation:
            return

        status = confirmation["status"]

        # Fake breakout
        if status == "FAKE":

            logger.info(
                f"{symbol} | Fake breakout detected"
            )

            return

        # Need another candle
        if status == "WAIT":

            logger.info(
                f"{symbol} | Candle 2 not strong enough | waiting"
            )

            return

        # ----------------------------------------------------
        # CONFIRMED
        # ----------------------------------------------------

        if status == "CONFIRMED":

            setup = create_setup(
                symbol,
                breakout,
            )

            SETUPS[symbol] = setup

            direction = setup["direction"]

            emoji = (
                "🟢"
                if direction == "LONG"
                else "🔴"
            )

            message = f"""
{emoji} {symbol} — H1 BREAKOUT CONFIRMED

Direction:
{direction}

Structure:
{breakout["structure"]}

Breakout Level:
{breakout["level"]:.6f}

Candle 2:
CONFIRMED

Confirmation Quality:
{confirmation["quality"]}

⚠️ NO SIGNAL YET

M15 Pullback / Retest Watch Activated.

AH1 ELITE v3
""".strip()

            await send_ntfy(
                f"{symbol} — H1 BREAKOUT CONFIRMED",
                message,
                "default",
            )

            logger.info(
                f"{symbol} | H1 breakout confirmed | "
                f"{direction} | M15 watch activated"
            )

    except Exception as e:

        logger.exception(
            f"{symbol} scan error: {e}"
        )


# ============================================================
# MAIN LOOP
# ============================================================

async def main():

    global http_client

    http_client = httpx.AsyncClient()

    exchange_class = getattr(
        ccxt,
        EXCHANGE_ID,
    )

    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })

    try:

        logger.info("=" * 60)
        logger.info("AH1 ELITE v3 STARTING")
        logger.info("Main timeframe: 1h")
        logger.info("Entry timeframe: 15m")
        logger.info("Strategy: H1 Breakout -> Candle 2 -> M15 Retest")
        logger.info("=" * 60)

        await send_ntfy(
            "AH1 ELITE v3",
            """
AH1 ELITE v3 is ONLINE.

H1 structure detection active.
Candle 2 confirmation active.
M15 Pullback / Retest watcher active.
RSI + Volume + Bollinger + ATR scoring active.

No signal without:
H1 Breakout
+
Candle 2 Confirmation
+
M15 Retest
""".strip(),
            "default",
        )

        while True:

            logger.info(
                f"Scanning {len(SYMBOLS)} symbols..."
            )

            for symbol in SYMBOLS:

                await scan_symbol(
                    exchange,
                    symbol,
                )

                await asyncio.sleep(1)

            await asyncio.sleep(
                SCAN_INTERVAL_SECONDS
            )

    finally:

        await exchange.close()
        await http_client.aclose()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info(
            "AH1 ELITE stopped manually."
        )

    except Exception as e:

        logger.exception(
            f"Fatal error: {e}"
        )
