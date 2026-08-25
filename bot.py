import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import ccxt.async_support as ccxt
import httpx
import numpy as np
import pandas as pd


# ============================================================
# H1 CONFLUENCE BREAKOUT ENGINE
# ============================================================
#
# MAIN TIMEFRAME:
#     1H
#
# HIGHER TIMEFRAME FILTER:
#     4H
#
# TRIGGERS:
#     1. Descending Trendline Break -> LONG
#     2. Ascending Trendline Break  -> SHORT
#     3. Static Resistance Break     -> LONG
#     4. Static Support Break        -> SHORT
#
# CONFIRMATION:
#     Breakout
#     Retest
#     Market Structure
#     Volume
#     ATR / Volatility
#     Pattern
#     4H Regime
#     Risk / Reward
#
# SIGNAL:
#     A+ or A only
#
# NOTRADE:
#     Everything else
#
# NOTIFICATION:
#     ntfy ONLY
#
# ============================================================


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]


# ============================================================
# TIMEFRAMES
# ============================================================

MAIN_TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"


# ============================================================
# DATA
# ============================================================

H1_LIMIT = 500
H4_LIMIT = 250


# ============================================================
# LOOP
# ============================================================

POLL_SECONDS = 20


# ============================================================
# BREAKOUT SETTINGS
# ============================================================

# Minimum distance outside the level relative to ATR.
ATR_BREAKOUT_BUFFER = 0.18

# How close price must come to the broken level during retest.
ATR_RETEST_TOLERANCE = 0.35

# Extra ATR buffer behind structure for stop loss.
ATR_STOP_BUFFER = 0.20


# ============================================================
# STATIC SUPPORT / RESISTANCE
# ============================================================

MIN_LEVEL_TOUCHES = 3
LEVEL_CLUSTER_ATR = 0.35


# ============================================================
# VOLUME
# ============================================================

VOLUME_LOOKBACK = 30
MIN_BREAKOUT_REL_VOLUME = 1.15


# ============================================================
# TRENDLINE
# ============================================================

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

MIN_TRENDLINE_TOUCHES = 3
MAX_TRENDLINE_AGE = 180


# ============================================================
# RISK
# ============================================================

MIN_RR = 2.0
MAX_STOP_ATR = 3.5


# ============================================================
# SCORE
# ============================================================

A_PLUS_SCORE = 85
A_SCORE = 75


# ============================================================
# COOLDOWN
# ============================================================

COOLDOWN_CANDLES = 6


# ============================================================
# STATE FILE
# ============================================================

STATE_FILE = Path(
    os.getenv(
        "STATE_FILE",
        "bot_state.json"
    )
)


# ============================================================
# NTFY
# ============================================================

NTFY_SERVER = "https://ntfy.sh"

NTFY_TOPIC = "btc_ah7K9xQ2_signal"


# ============================================================
# EXCHANGE
# ============================================================

EXCHANGE_ID = os.getenv(
    "EXCHANGE_ID",
    "binance"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(
    "H1-BREAKOUT"
)


# ============================================================
# STATE
# ============================================================

@dataclass
class SetupState:

    status: str = "NO_SETUP"

    direction: Optional[str] = None

    trigger_type: Optional[str] = None

    level: Optional[float] = None

    breakout_price: Optional[float] = None

    breakout_candle: Optional[int] = None

    retest_seen: bool = False

    retest_candle: Optional[int] = None

    pattern: Optional[str] = None

    score: float = 0.0

    last_signal_candle: Optional[int] = None

    cooldown_until: Optional[int] = None

    created_at: Optional[int] = None


# ============================================================
# UTILITIES
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:

        return default


def fmt_price(
    price: float
):

    if price >= 1000:
        decimals = 2

    elif price >= 100:
        decimals = 3

    elif price >= 1:
        decimals = 4

    else:
        decimals = 6

    return f"{price:.{decimals}f}"


# ============================================================
# STATE LOAD
# ============================================================

def load_state() -> Dict[str, SetupState]:

    default = {
        symbol: SetupState()
        for symbol in SYMBOLS
    }

    if not STATE_FILE.exists():
        return default

    try:

        raw = json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

        result = {}

        for symbol in SYMBOLS:

            data = raw.get(
                symbol,
                {}
            )

            allowed = {
                key: value
                for key, value in data.items()
                if key in SetupState.__dataclass_fields__
            }

            result[symbol] = SetupState(
                **allowed
            )

        return result

    except Exception as e:

        logger.exception(
            "Could not load state: %s",
            e
        )

        return default


# ============================================================
# STATE SAVE
# ============================================================

def save_state(
    states: Dict[str, SetupState]
):

    try:

        payload = {
            symbol: asdict(state)
            for symbol, state in states.items()
        }

        temp_file = STATE_FILE.with_suffix(
            ".tmp"
        )

        temp_file.write_text(
            json.dumps(
                payload,
                indent=2
            ),
            encoding="utf-8"
        )

        temp_file.replace(
            STATE_FILE
        )

    except Exception as e:

        logger.exception(
            "Could not save state: %s",
            e
        )


# ============================================================
# OHLCV -> DATAFRAME
# ============================================================

def candles_to_df(
    ohlcv
):

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df["timestamp"] = (
        pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )
        .astype("Int64")
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    return df.reset_index(
        drop=True
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df
):

    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["tr"] = tr

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = (
        tr.rolling(
            14
        ).mean()
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        close.ewm(
            span=20,
            adjust=False
        ).mean()
    )

    df["ema50"] = (
        close.ewm(
            span=50,
            adjust=False
        ).mean()
    )

    df["ema200"] = (
        close.ewm(
            span=200,
            adjust=False
        ).mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain.rolling(
            14
        ).mean()
    )

    avg_loss = (
        loss.rolling(
            14
        ).mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        close.ewm(
            span=12,
            adjust=False
        ).mean()
    )

    ema26 = (
        close.ewm(
            span=26,
            adjust=False
        ).mean()
    )

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    df["volume_ma"] = (
        volume.rolling(
            VOLUME_LOOKBACK
        ).mean()
    )

    df["rel_volume"] = (
        volume
        / df["volume_ma"]
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["upper_wick"] = (
        df["high"]
        - df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["lower_wick"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        - df["low"]
    )

    return df


# ============================================================
# PIVOTS
# ============================================================

def find_pivots(
    df,
    left=PIVOT_LEFT,
    right=PIVOT_RIGHT
):

    highs = []
    lows = []

    if len(df) < (
        left
        + right
        + 5
    ):
        return highs, lows

    for i in range(
        left,
        len(df) - right
    ):

        current_high = safe_float(
            df.iloc[i]["high"]
        )

        current_low = safe_float(
            df.iloc[i]["low"]
        )

        left_highs = df.iloc[
            i - left:i
        ]["high"]

        right_highs = df.iloc[
            i + 1:i + right + 1
        ]["high"]

        left_lows = df.iloc[
            i - left:i
        ]["low"]

        right_lows = df.iloc[
            i + 1:i + right + 1
        ]["low"]

        if (
            current_high >= left_highs.max()
            and current_high >= right_highs.max()
        ):
            highs.append(i)

        if (
            current_low <= left_lows.min()
            and current_low <= right_lows.min()
        ):
            lows.append(i)

    return highs, lows


# ============================================================
# STATIC LEVEL CLUSTERING
# ============================================================

def cluster_levels(
    prices: List[float],
    atr: float
):

    if not prices:
        return []

    prices = sorted(prices)

    tolerance = max(
        atr * LEVEL_CLUSTER_ATR,
        np.mean(prices) * 0.001
    )

    clusters = []

    current = [
        prices[0]
    ]

    for price in prices[1:]:

        center = np.mean(
            current
        )

        if (
            abs(
                price - center
            )
            <= tolerance
        ):

            current.append(
                price
            )

        else:

            if len(current) >= (
                MIN_LEVEL_TOUCHES
            ):

                clusters.append(
                    float(
                        np.mean(
                            current
                        )
                    )
                )

            current = [
                price
            ]

    if len(current) >= (
        MIN_LEVEL_TOUCHES
    ):

        clusters.append(
            float(
                np.mean(
                    current
                )
            )
        )

    return clusters


# ============================================================
# STATIC SUPPORT / RESISTANCE
# ============================================================

def detect_static_levels(
    df
):

    if len(df) < 80:
        return [], []

    atr = safe_float(
        df.iloc[-1]["atr"]
    )

    highs, lows = find_pivots(
        df
    )

    high_prices = [
        safe_float(
            df.iloc[i]["high"]
        )
        for i in highs
    ]

    low_prices = [
        safe_float(
            df.iloc[i]["low"]
        )
        for i in lows
    ]

    resistance = cluster_levels(
        high_prices,
        atr
    )

    support = cluster_levels(
        low_prices,
        atr
    )

    return support, resistance


# ============================================================
# TRENDLINE CALCULATION
# ============================================================

def line_value(
    x1,
    y1,
    x2,
    y2,
    x
):

    if x2 == x1:
        return None

    slope = (
        y2 - y1
    ) / (
        x2 - x1
    )

    return (
        y1
        + slope * (
            x - x1
        )
    )


# ============================================================
# TRENDLINE BUILDER
# ============================================================

def build_trendline(
    df,
    pivot_indices,
    mode
):

    if len(pivot_indices) < 2:
        return None

    recent = [
        i
        for i in pivot_indices
        if i >= (
            len(df)
            - MAX_TRENDLINE_AGE
        )
    ]

    if len(recent) < 2:
        return None

    candidates = []

    start = max(
        0,
        len(recent) - 8
    )

    for a in range(
        start,
        len(recent)
    ):

        for b in range(
            a + 1,
            len(recent)
        ):

            i1 = recent[a]
            i2 = recent[b]

            if i2 <= i1:
                continue

            if mode == "resistance":

                y1 = safe_float(
                    df.iloc[i1]["high"]
                )

                y2 = safe_float(
                    df.iloc[i2]["high"]
                )

            else:

                y1 = safe_float(
                    df.iloc[i1]["low"]
                )

                y2 = safe_float(
                    df.iloc[i2]["low"]
                )

            slope = (
                y2 - y1
            ) / (
                i2 - i1
            )

            # Descending resistance.
            if (
                mode == "resistance"
                and slope >= 0
            ):
                continue

            # Ascending support.
            if (
                mode == "support"
                and slope <= 0
            ):
                continue

            touches = 0

            for idx in recent:

                predicted = line_value(
                    i1,
                    y1,
                    i2,
                    y2,
                    idx
                )

                if predicted is None:
                    continue

                actual = (
                    safe_float(
                        df.iloc[idx]["high"]
                    )
                    if mode == "resistance"
                    else safe_float(
                        df.iloc[idx]["low"]
                    )
                )

                local_atr = safe_float(
                    df.iloc[idx]["atr"]
                )

                tolerance = (
                    local_atr * 0.35
                )

                if (
                    tolerance > 0
                    and abs(
                        actual
                        - predicted
                    )
                    <= tolerance
                ):
                    touches += 1

            if touches >= (
                MIN_TRENDLINE_TOUCHES
            ):

                candidates.append(
                    (
                        touches,
                        i1,
                        i2,
                        y1,
                        y2
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[2]
        ),
        reverse=True
    )

    touches, i1, i2, y1, y2 = (
        candidates[0]
    )

    return {
        "x1": i1,
        "x2": i2,
        "y1": y1,
        "y2": y2,
        "touches": touches,
    }


# ============================================================
# BREAKOUT AGAINST LEVEL
# ============================================================

def is_level_breakout(
    previous_close,
    current_close,
    level,
    atr,
    direction
):

    buffer = max(
        atr * ATR_BREAKOUT_BUFFER,
        abs(level) * 0.0005
    )

    if direction == "LONG":

        return (
            previous_close <= level
            and current_close
            > level + buffer
        )

    return (
        previous_close >= level
        and current_close
        < level - buffer
    )


# ============================================================
# STATIC BREAKOUT
# ============================================================

def find_static_breakout(
    df
):

    if len(df) < 100:
        return None

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_close = safe_float(
        previous["close"]
    )

    current_close = safe_float(
        current["close"]
    )

    atr = safe_float(
        current["atr"]
    )

    support, resistance = (
        detect_static_levels(
            df.iloc[:-1]
        )
    )

    candidates = []

    for level in resistance:

        if is_level_breakout(
            previous_close,
            current_close,
            level,
            atr,
            "LONG"
        ):

            candidates.append(
                {
                    "direction": "LONG",
                    "type": "STATIC_RESISTANCE",
                    "level": level,
                }
            )

    for level in support:

        if is_level_breakout(
            previous_close,
            current_close,
            level,
            atr,
            "SHORT"
        ):

            candidates.append(
                {
                    "direction": "SHORT",
                    "type": "STATIC_SUPPORT",
                    "level": level,
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: abs(
            current_close
            - item["level"]
        )
    )

    return candidates[0]


# ============================================================
# TRENDLINE BREAKOUT
# ============================================================

def find_trendline_breakout(
    df
):

    if len(df) < 100:
        return None

    # Exclude current candle when constructing line.
    history = df.iloc[:-1].copy()

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_close = safe_float(
        previous["close"]
    )

    current_close = safe_float(
        current["close"]
    )

    atr = safe_float(
        current["atr"]
    )

    highs, lows = find_pivots(
        history
    )

    descending = build_trendline(
        history,
        highs,
        "resistance"
    )

    ascending = build_trendline(
        history,
        lows,
        "support"
    )

    candidates = []

    # --------------------------------------------------------
    # Descending trendline -> LONG
    # --------------------------------------------------------

    if descending:

        line_previous = line_value(
            descending["x1"],
            descending["y1"],
            descending["x2"],
            descending["y2"],
            len(history) - 1
        )

        if line_previous is not None:

            if (
                previous_close
                <= line_previous
                and current_close
                > (
                    line_previous
                    + atr
                    * ATR_BREAKOUT_BUFFER
                )
            ):

                candidates.append(
                    {
                        "direction": "LONG",
                        "type": "DESCENDING_TRENDLINE",
                        "level": float(
                            line_previous
                        ),
                    }
                )

    # --------------------------------------------------------
    # Ascending trendline -> SHORT
    # --------------------------------------------------------

    if ascending:

        line_previous = line_value(
            ascending["x1"],
            ascending["y1"],
            ascending["x2"],
            ascending["y2"],
            len(history) - 1
        )

        if line_previous is not None:

            if (
                previous_close
                >= line_previous
                and current_close
                < (
                    line_previous
                    - atr
                    * ATR_BREAKOUT_BUFFER
                )
            ):

                candidates.append(
                    {
                        "direction": "SHORT",
                        "type": "ASCENDING_TRENDLINE",
                        "level": float(
                            line_previous
                        ),
                    }
                )

    if not candidates:
        return None

    return candidates[0]


# ============================================================
# PATTERNS
# ============================================================

def detect_double_bottom(
    df
):

    highs, lows = find_pivots(
        df
    )

    if len(lows) < 2:
        return False

    i1, i2 = lows[-2:]

    low1 = safe_float(
        df.iloc[i1]["low"]
    )

    low2 = safe_float(
        df.iloc[i2]["low"]
    )

    atr = safe_float(
        df.iloc[-1]["atr"]
    )

    if atr <= 0:
        return False

    if abs(
        low1 - low2
    ) > atr * 1.2:
        return False

    if (
        i2 - i1
        < 5
    ):
        return False

    neckline = safe_float(
        df.iloc[
            i1:i2 + 1
        ]["high"].max()
    )

    current_close = safe_float(
        df.iloc[-1]["close"]
    )

    return (
        current_close
        > neckline
        + atr * 0.15
    )


def detect_double_top(
    df
):

    highs, lows = find_pivots(
        df
    )

    if len(highs) < 2:
        return False

    i1, i2 = highs[-2:]

    high1 = safe_float(
        df.iloc[i1]["high"]
    )

    high2 = safe_float(
        df.iloc[i2]["high"]
    )

    atr = safe_float(
        df.iloc[-1]["atr"]
    )

    if atr <= 0:
        return False

    if abs(
        high1 - high2
    ) > atr * 1.2:
        return False

    if (
        i2 - i1
        < 5
    ):
        return False

    neckline = safe_float(
        df.iloc[
            i1:i2 + 1
        ]["low"].min()
    )

    current_close = safe_float(
        df.iloc[-1]["close"]
    )

    return (
        current_close
        < neckline
        - atr * 0.15
    )


def detect_head_shoulders(
    df
):

    highs, lows = find_pivots(
        df
    )

    if len(highs) < 3:
        return False

    left_i, head_i, right_i = (
        highs[-3:]
    )

    left = safe_float(
        df.iloc[left_i]["high"]
    )

    head = safe_float(
        df.iloc[head_i]["high"]
    )

    right = safe_float(
        df.iloc[right_i]["high"]
    )

    atr = safe_float(
        df.iloc[-1]["atr"]
    )

    if atr <= 0:
        return False

    if not (
        head > left
        and head > right
    ):
        return False

    if abs(
        left - right
    ) > atr * 1.8:
        return False

    neckline = safe_float(
        df.iloc[
            left_i:right_i + 1
        ]["low"].max()
    )

    return (
        safe_float(
            df.iloc[-1]["close"]
        )
        < neckline
        - atr * 0.15
    )


def detect_inverse_head_shoulders(
    df
):

    highs, lows = find_pivots(
        df
    )

    if len(lows) < 3:
        return False

    left_i, head_i, right_i = (
        lows[-3:]
    )

    left = safe_float(
        df.iloc[left_i]["low"]
    )

    head = safe_float(
        df.iloc[head_i]["low"]
    )

    right = safe_float(
        df.iloc[right_i]["low"]
    )

    atr = safe_float(
        df.iloc[-1]["atr"]
    )

    if atr <= 0:
        return False

    if not (
        head < left
        and head < right
    ):
        return False

    if abs(
        left - right
    ) > atr * 1.8:
        return False

    neckline = safe_float(
        df.iloc[
            left_i:right_i + 1
        ]["high"].min()
    )

    return (
        safe_float(
            df.iloc[-1]["close"]
        )
        > neckline
        + atr * 0.15
    )


def detect_engulfing(
    df
):

    if len(df) < 3:
        return None

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_open = safe_float(
        previous["open"]
    )

    previous_close = safe_float(
        previous["close"]
    )

    current_open = safe_float(
        current["open"]
    )

    current_close = safe_float(
        current["close"]
    )

    bullish = (
        previous_close < previous_open
        and current_close > current_open
        and current_open <= previous_close
        and current_close >= previous_open
    )

    bearish = (
        previous_close > previous_open
        and current_close < current_open
        and current_open >= previous_close
        and current_close <= previous_open
    )

    if bullish:
        return "BULLISH_ENGULFING"

    if bearish:
        return "BEARISH_ENGULFING"

    return None


def detect_pinbar(
    df
):

    if len(df) < 2:
        return None

    candle = df.iloc[-1]

    body = safe_float(
        candle["body"]
    )

    upper_wick = safe_float(
        candle["upper_wick"]
    )

    lower_wick = safe_float(
        candle["lower_wick"]
    )

    candle_range = safe_float(
        candle["range"]
    )

    if candle_range <= 0:
        return None

    if (
        lower_wick >= body * 2
        and lower_wick >= candle_range * 0.5
    ):
        return "BULLISH_PINBAR"

    if (
        upper_wick >= body * 2
        and upper_wick >= candle_range * 0.5
    ):
        return "BEARISH_PINBAR"

    return None


def detect_patterns(
    df
):

    patterns = []

    if detect_double_bottom(
        df
    ):
        patterns.append(
            "DOUBLE_BOTTOM"
        )

    if detect_double_top(
        df
    ):
        patterns.append(
            "DOUBLE_TOP"
        )

    if detect_head_shoulders(
        df
    ):
        patterns.append(
            "HEAD_AND_SHOULDERS"
        )

    if detect_inverse_head_shoulders(
        df
    ):
        patterns.append(
            "INVERSE_HEAD_AND_SHOULDERS"
        )

    engulfing = detect_engulfing(
        df
    )

    if engulfing:
        patterns.append(
            engulfing
        )

    pinbar = detect_pinbar(
        df
    )

    if pinbar:
        patterns.append(
            pinbar
        )

    return patterns


# ============================================================
# PATTERN FILTER BY DIRECTION
# ============================================================

def best_pattern_for_direction(
    patterns,
    direction
):

    if direction == "LONG":

        preferred = [
            "DOUBLE_BOTTOM",
            "INVERSE_HEAD_AND_SHOULDERS",
            "BULLISH_ENGULFING",
            "BULLISH_PINBAR",
        ]

    else:

        preferred = [
            "DOUBLE_TOP",
            "HEAD_AND_SHOULDERS",
            "BEARISH_ENGULFING",
            "BEARISH_PINBAR",
        ]

    for pattern in preferred:

        if pattern in patterns:
            return pattern

    return None


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure_score(
    df,
    direction
):

    highs, lows = find_pivots(
        df
    )

    if (
        len(highs) < 2
        or len(lows) < 2
    ):
        return 0

    previous_high = safe_float(
        df.iloc[
            highs[-2]
        ]["high"]
    )

    latest_high = safe_float(
        df.iloc[
            highs[-1]
        ]["high"]
    )

    previous_low = safe_float(
        df.iloc[
            lows[-2]
        ]["low"]
    )

    latest_low = safe_float(
        df.iloc[
            lows[-1]
        ]["low"]
    )

    if direction == "LONG":

        if (
            latest_high > previous_high
            and latest_low > previous_low
        ):
            return 15

        if (
            latest_high > previous_high
            or latest_low > previous_low
        ):
            return 8

    else:

        if (
            latest_high < previous_high
            and latest_low < previous_low
        ):
            return 15

        if (
            latest_high < previous_high
            or latest_low < previous_low
        ):
            return 8

    return 0


# ============================================================
# 4H MARKET REGIME
# ============================================================

def htf_regime(
    df4,
    direction
):

    if len(df4) < 60:
        return 0, "UNKNOWN"

    candle = df4.iloc[-1]

    close = safe_float(
        candle["close"]
    )

    ema20 = safe_float(
        candle["ema20"]
    )

    ema50 = safe_float(
        candle["ema50"]
    )

    ema200 = safe_float(
        candle["ema200"]
    )

    if direction == "LONG":

        if (
            close > ema20 > ema50
            and ema50 > ema200
        ):
            return 10, "BULLISH"

        if close > ema50:
            return 5, "BULLISH_WEAK"

        if close < ema50:
            return -5, "BEARISH"

    else:

        if (
            close < ema20 < ema50
            and ema50 < ema200
        ):
            return 10, "BEARISH"

        if close < ema50:
            return 5, "BEARISH_WEAK"

        if close > ema50:
            return -5, "BULLISH"

    return 0, "NEUTRAL"


# ============================================================
# RETEST DETECTION
# ============================================================

def detect_retest(
    df,
    state: SetupState
):

    if state.status not in (
        "WAITING_RETEST",
        "RETEST"
    ):
        return False

    if state.level is None:
        return False

    candle = df.iloc[-1]

    level = safe_float(
        state.level
    )

    atr = safe_float(
        candle["atr"]
    )

    if atr <= 0:
        return False

    tolerance = max(
        atr * ATR_RETEST_TOLERANCE,
        abs(level) * 0.001
    )

    high = safe_float(
        candle["high"]
    )

    low = safe_float(
        candle["low"]
    )

    close = safe_float(
        candle["close"]
    )

    touched = (
        low <= level + tolerance
        and high >= level - tolerance
    )

    if not touched:
        return False

    if state.direction == "LONG":

        return close >= level

    return close <= level


# ============================================================
# CONFIRMATION SCORE
# ============================================================

def confirmation_score(
    df,
    direction
):

    score = 0
    reasons = []

    candle = df.iloc[-1]

    close = safe_float(
        candle["close"]
    )

    open_price = safe_float(
        candle["open"]
    )

    rsi = safe_float(
        candle["rsi"]
    )

    macd = safe_float(
        candle["macd"]
    )

    macd_signal = safe_float(
        candle["macd_signal"]
    )

    relative_volume = safe_float(
        candle["rel_volume"],
        1
    )

    atr = safe_float(
        candle["atr"]
    )

    candle_range = safe_float(
        candle["range"]
    )

    # --------------------------------------------------------
    # Directional candle
    # --------------------------------------------------------

    if direction == "LONG":

        if close > open_price:

            score += 8

            reasons.append(
                "bullish_confirmation_candle"
            )

    else:

        if close < open_price:

            score += 8

            reasons.append(
                "bearish_confirmation_candle"
            )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        relative_volume
        >= MIN_BREAKOUT_REL_VOLUME
    ):

        score += 10

        reasons.append(
            f"relative_volume={relative_volume:.2f}"
        )

    elif relative_volume >= 1.0:

        score += 5

    # --------------------------------------------------------
    # ATR expansion
    # --------------------------------------------------------

    if len(df) >= 25:

        atr_average = safe_float(
            df["atr"]
            .iloc[-21:-1]
            .mean()
        )

        if atr_average > 0:

            atr_ratio = (
                atr
                / atr_average
            )

            if atr_ratio >= 1.10:

                score += 10

                reasons.append(
                    "atr_expansion"
                )

            elif atr_ratio >= 0.95:

                score += 5

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if direction == "LONG":

        if 52 <= rsi <= 72:

            score += 7

            reasons.append(
                "rsi_bullish"
            )

        elif 45 <= rsi < 52:

            score += 3

    else:

        if 28 <= rsi <= 48:

            score += 7

            reasons.append(
                "rsi_bearish"
            )

        elif 48 < rsi <= 55:

            score += 3

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if direction == "LONG":

        if macd > macd_signal:

            score += 5

            reasons.append(
                "macd_bullish"
            )

    else:

        if macd < macd_signal:

            score += 5

            reasons.append(
                "macd_bearish"
            )

    # --------------------------------------------------------
    # Candle range
    # --------------------------------------------------------

    if (
        atr > 0
        and candle_range >= atr * 0.70
    ):

        score += 5

        reasons.append(
            "strong_candle_range"
        )

    return score, reasons


# ============================================================
# TRADE PLAN
# ============================================================

def calculate_trade_plan(
    df,
    direction
):

    candle = df.iloc[-1]

    entry = safe_float(
        candle["close"]
    )

    atr = safe_float(
        candle["atr"]
    )

    if atr <= 0:
        return None

    highs, lows = find_pivots(
        df
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        recent_lows = [
            safe_float(
                df.iloc[i]["low"]
            )
            for i in lows[-5:]
        ]

        if recent_lows:

            structure_low = min(
                recent_lows
            )

        else:

            structure_low = (
                entry - atr
            )

        stop = (
            structure_low
            - atr * ATR_STOP_BUFFER
        )

        risk = (
            entry
            - stop
        )

        if risk <= 0:
            return None

        if risk > (
            atr * MAX_STOP_ATR
        ):
            return None

        target = (
            entry
            + risk * MIN_RR
        )

        rr = (
            target - entry
        ) / risk

        return {
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk": risk,
            "rr": rr,
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    recent_highs = [
        safe_float(
            df.iloc[i]["high"]
        )
        for i in highs[-5:]
    ]

    if recent_highs:

        structure_high = max(
            recent_highs
        )

    else:

        structure_high = (
            entry + atr
        )

    stop = (
        structure_high
        + atr * ATR_STOP_BUFFER
    )

    risk = (
        stop
        - entry
    )

    if risk <= 0:
        return None

    if risk > (
        atr * MAX_STOP_ATR
    ):
        return None

    target = (
        entry
        - risk * MIN_RR
    )

    rr = (
        entry - target
    ) / risk

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "rr": rr,
    }


# ============================================================
# SCORE ENGINE
# ============================================================

def calculate_score(
    df,
    df4,
    direction,
    trigger_type,
    pattern,
    retest
):

    score = 0

    reasons = []

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    score += 20

    reasons.append(
        f"breakout={trigger_type}"
    )

    # --------------------------------------------------------
    # Retest
    # --------------------------------------------------------

    if retest:

        score += 20

        reasons.append(
            "successful_retest"
        )

    # --------------------------------------------------------
    # Market structure
    # --------------------------------------------------------

    structure_score = (
        market_structure_score(
            df,
            direction
        )
    )

    score += structure_score

    if structure_score >= 15:

        reasons.append(
            "market_structure_confirmed"
        )

    elif structure_score >= 8:

        reasons.append(
            "market_structure_partial"
        )

    # --------------------------------------------------------
    # Pattern
    # --------------------------------------------------------

    if pattern:

        score += 10

        reasons.append(
            f"pattern={pattern}"
        )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    indicator_score, indicator_reasons = (
        confirmation_score(
            df,
            direction
        )
    )

    score += indicator_score

    reasons.extend(
        indicator_reasons
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    htf_score, regime = htf_regime(
        df4,
        direction
    )

    score += htf_score

    reasons.append(
        f"4h_regime={regime}"
    )

    return (
        score,
        reasons,
        regime
    )


# ============================================================
# QUALITY
# ============================================================

def get_quality(
    score
):

    if score >= A_PLUS_SCORE:
        return "A+"

    if score >= A_SCORE:
        return "A"

    return "NO_TRADE"


# ============================================================
# DISCOVER BREAKOUT
# ============================================================

def discover_breakout(
    df
):

    candidates = []

    static_breakout = (
        find_static_breakout(
            df
        )
    )

    if static_breakout:
        candidates.append(
            static_breakout
        )

    trendline_breakout = (
        find_trendline_breakout(
            df
        )
    )

    if trendline_breakout:
        candidates.append(
            trendline_breakout
        )

    if not candidates:
        return None

    candle = df.iloc[-1]

    close = safe_float(
        candle["close"]
    )

    atr = safe_float(
        candle["atr"]
    )

    relative_volume = safe_float(
        candle["rel_volume"],
        1
    )

    def rank(candidate):

        distance = abs(
            close
            - candidate["level"]
        )

        strength = (
            distance / atr
            if atr > 0
            else 0
        )

        return (
            strength * 2
            + relative_volume
        )

    candidates.sort(
        key=rank,
        reverse=True
    )

    return candidates[0]


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    df,
    df4,
    state: SetupState
):

    if len(df) < 220:
        return state, None

    current = df.iloc[-1]

    candle_timestamp = int(
        current["timestamp"]
    )

    # ========================================================
    # COOLDOWN
    # ========================================================

    if (
        state.cooldown_until
        and candle_timestamp
        < state.cooldown_until
    ):

        return state, None

    # ========================================================
    # ACTIVE SETUP
    # ========================================================

    if state.status in (
        "WAITING_RETEST",
        "RETEST"
    ):

        # ----------------------------------------------------
        # Check retest
        # ----------------------------------------------------

        retest = detect_retest(
            df,
            state
        )

        if retest:

            state.retest_seen = True

            state.retest_candle = (
                candle_timestamp
            )

            state.status = "RETEST"

            patterns = detect_patterns(
                df
            )

            pattern = (
                best_pattern_for_direction(
                    patterns,
                    state.direction
                )
            )

            score, reasons, regime = (
                calculate_score(
                    df,
                    df4,
                    state.direction,
                    state.trigger_type,
                    pattern,
                    True
                )
            )

            trade_plan = (
                calculate_trade_plan(
                    df,
                    state.direction
                )
            )

            if trade_plan is None:

                return state, None

            # ------------------------------------------------
            # Hard 4H rejection
            # ------------------------------------------------

            if (
                state.direction == "LONG"
                and regime == "BEARISH"
            ):

                logger.info(
                    "%s | LONG rejected by 4H",
                    symbol
                )

                return (
                    SetupState(),
                    None
                )

            if (
                state.direction == "SHORT"
                and regime == "BULLISH"
            ):

                logger.info(
                    "%s | SHORT rejected by 4H",
                    symbol
                )

                return (
                    SetupState(),
                    None
                )

            # ------------------------------------------------
            # Minimum R:R
            # ------------------------------------------------

            if (
                trade_plan["rr"]
                < MIN_RR
            ):

                logger.info(
                    "%s | R:R too low",
                    symbol
                )

                return state, None

            # ------------------------------------------------
            # Quality
            # ------------------------------------------------

            quality = get_quality(
                score
            )

            if quality not in (
                "A",
                "A+"
            ):

                logger.info(
                    "%s | Retest found but score "
                    "too low: %.1f",
                    symbol,
                    score
                )

                return state, None

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            state.status = "SIGNAL"

            state.score = score

            state.pattern = pattern

            state.last_signal_candle = (
                candle_timestamp
            )

            signal = {
                "symbol": symbol,
                "direction": state.direction,
                "quality": quality,
                "score": round(
                    score,
                    1
                ),
                "trigger": state.trigger_type,
                "pattern": pattern,
                "level": state.level,
                "entry": trade_plan["entry"],
                "stop": trade_plan["stop"],
                "target": trade_plan["target"],
                "rr": trade_plan["rr"],
                "regime_4h": regime,
                "candle": candle_timestamp,
                "reasons": reasons,
            }

            # ------------------------------------------------
            # Cooldown
            # ------------------------------------------------

            indices = df.index[
                df["timestamp"]
                == candle_timestamp
            ]

            if len(indices):

                current_index = int(
                    indices[-1]
                )

                cooldown_index = min(
                    current_index
                    + COOLDOWN_CANDLES,
                    len(df) - 1
                )

                state.cooldown_until = int(
                    df.iloc[
                        cooldown_index
                    ]["timestamp"]
                )

            return state, signal

        # ----------------------------------------------------
        # Breakout failure
        # ----------------------------------------------------

        level = safe_float(
            state.level
        )

        close = safe_float(
            current["close"]
        )

        atr = safe_float(
            current["atr"]
        )

        if state.direction == "LONG":

            if (
                close
                < level
                - atr * 0.50
            ):

                logger.info(
                    "%s | LONG breakout invalidated",
                    symbol
                )

                return (
                    SetupState(),
                    None
                )

        else:

            if (
                close
                > level
                + atr * 0.50
            ):

                logger.info(
                    "%s | SHORT breakout invalidated",
                    symbol
                )

                return (
                    SetupState(),
                    None
                )

        # Still waiting.
        return state, None

    # ========================================================
    # NO ACTIVE SETUP
    # ========================================================

    breakout = discover_breakout(
        df
    )

    if not breakout:

        return state, None

    # --------------------------------------------------------
    # Register breakout
    # --------------------------------------------------------

    state.status = (
        "WAITING_RETEST"
    )

    state.direction = (
        breakout["direction"]
    )

    state.trigger_type = (
        breakout["type"]
    )

    state.level = safe_float(
        breakout["level"]
    )

    state.breakout_price = safe_float(
        current["close"]
    )

    state.breakout_candle = (
        candle_timestamp
    )

    state.created_at = (
        candle_timestamp
    )

    patterns = detect_patterns(
        df
    )

    state.pattern = (
        best_pattern_for_direction(
            patterns,
            state.direction
        )
    )

    logger.info(
        "%s | BREAKOUT | %s | %s | level=%s | pattern=%s",
        symbol,
        state.direction,
        state.trigger_type,
        fmt_price(
            state.level
        ),
        state.pattern
    )

    # IMPORTANT:
    #
    # NO SIGNAL HERE.
    #
    # We wait for the retest.
    #

    return state, None


# ============================================================
# NTFY SEND
# ============================================================

async def send_ntfy(
    client: httpx.AsyncClient,
    signal
):

    direction = signal[
        "direction"
    ]

    quality = signal[
        "quality"
    ]

    symbol = signal[
        "symbol"
    ]

    title = (
        f"{symbol} "
        f"{direction} "
        f"{quality}"
    )

    body = (
        f"{symbol} — {direction}\n\n"

        f"Quality: {quality}\n"
        f"Score: {signal['score']}/100\n\n"

        f"Trigger:\n"
        f"{signal['trigger']}\n\n"

        f"Pattern:\n"
        f"{signal['pattern'] or 'None'}\n\n"

        f"Breakout Level:\n"
        f"{fmt_price(signal['level'])}\n\n"

        f"Entry:\n"
        f"{fmt_price(signal['entry'])}\n\n"

        f"Stop Loss:\n"
        f"{fmt_price(signal['stop'])}\n\n"

        f"Take Profit:\n"
        f"{fmt_price(signal['target'])}\n\n"

        f"Risk/Reward:\n"
        f"1:{signal['rr']:.2f}\n\n"

        f"4H Regime:\n"
        f"{signal['regime_4h']}\n\n"

        f"Confluence:\n"
        + "\n".join(
            f"- {reason}"
            for reason in signal["reasons"][:10]
        )
        + "\n\n"

        f"H1 Confluence Breakout Engine"
    )

    url = (
        f"{NTFY_SERVER.rstrip('/')}/"
        f"{NTFY_TOPIC}"
    )

    headers = {
        "Title": title,
        "Priority": "high",
        "Tags": (
            "chart_with_upwards_trend"
            if direction == "LONG"
            else "chart_with_downwards_trend"
        ),
    }

    try:

        response = await client.post(
            url,
            content=body.encode(
                "utf-8"
            ),
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        logger.info(
            "%s | ntfy signal sent successfully",
            symbol
        )

        return True

    except Exception as e:

        logger.error(
            "%s | ntfy send failed: %s",
            symbol,
            e
        )

        # IMPORTANT:
        # ntfy failure must NOT crash the bot.
        return False


# ============================================================
# FETCH DATA
# ============================================================

async def fetch_symbol_data(
    exchange,
    symbol
):

    h1_raw = await exchange.fetch_ohlcv(
        symbol,
        MAIN_TIMEFRAME,
        limit=H1_LIMIT
    )

    h4_raw = await exchange.fetch_ohlcv(
        symbol,
        HIGHER_TIMEFRAME,
        limit=H4_LIMIT
    )

    h1 = candles_to_df(
        h1_raw
    )

    h4 = candles_to_df(
        h4_raw
    )

    if len(h1) < 20:
        return None, None

    if len(h4) < 20:
        return None, None

    # ========================================================
    # CRITICAL:
    #
    # Remove the currently forming candle.
    #
    # The bot ONLY analyzes completed candles.
    # ========================================================

    h1 = h1.iloc[:-1].copy()

    h4 = h4.iloc[:-1].copy()

    h1 = add_indicators(
        h1
    )

    h4 = add_indicators(
        h4
    )

    return h1, h4


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "=============================================="
    )

    logger.info(
        "H1 CONFLUENCE BREAKOUT ENGINE STARTING"
    )

    logger.info(
        "Main timeframe: %s",
        MAIN_TIMEFRAME
    )

    logger.info(
        "Higher timeframe: %s",
        HIGHER_TIMEFRAME
    )

    logger.info(
        "Notification: ntfy"
    )

    logger.info(
        "ntfy topic: %s",
        NTFY_TOPIC
    )

    logger.info(
        "=============================================="
    )

    states = load_state()

    exchange_class = getattr(
        ccxt,
        EXCHANGE_ID
    )

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot"
            },
        }
    )

    last_processed = {
        symbol: None
        for symbol in SYMBOLS
    }

    async with httpx.AsyncClient() as client:

        try:

            await exchange.load_markets()

            logger.info(
                "Exchange loaded: %s",
                exchange.id
            )

            while True:

                cycle_start = (
                    time.time()
                )

                for symbol in SYMBOLS:

                    try:

                        # ------------------------------------------------
                        # MARKET CHECK
                        # ------------------------------------------------

                        if symbol not in exchange.markets:

                            logger.warning(
                                "%s | market not available",
                                symbol
                            )

                            continue

                        # ------------------------------------------------
                        # FETCH
                        # ------------------------------------------------

                        h1, h4 = (
                            await fetch_symbol_data(
                                exchange,
                                symbol
                            )
                        )

                        if (
                            h1 is None
                            or h4 is None
                        ):
                            continue

                        # ------------------------------------------------
                        # CLOSED H1 CANDLE
                        # ------------------------------------------------

                        candle_timestamp = int(
                            h1.iloc[-1][
                                "timestamp"
                            ]
                        )

                        # ------------------------------------------------
                        # PROCESS EACH CANDLE ONLY ONCE
                        # ------------------------------------------------

                        if (
                            last_processed[
                                symbol
                            ]
                            == candle_timestamp
                        ):

                            continue

                        last_processed[
                            symbol
                        ] = candle_timestamp

                        logger.info(
                            "%s | new closed H1 candle",
                            symbol
                        )

                        # ------------------------------------------------
                        # ENGINE
                        # ------------------------------------------------

                        new_state, signal = (
                            process_symbol(
                                symbol,
                                h1,
                                h4,
                                states[symbol]
                            )
                        )

                        states[
                            symbol
                        ] = new_state

                        save_state(
                            states
                        )

                        # ------------------------------------------------
                        # SIGNAL
                        # ------------------------------------------------

                        if signal:

                            logger.info(
                                "%s | %s SIGNAL | score=%s",
                                symbol,
                                signal["direction"],
                                signal["score"]
                            )

                            await send_ntfy(
                                client,
                                signal
                            )

                            # Save state again after
                            # notification attempt.
                            save_state(
                                states
                            )

                    except (
                        ccxt.NetworkError,
                        ccxt.ExchangeNotAvailable,
                        ccxt.RequestTimeout,
                    ) as e:

                        logger.warning(
                            "%s | exchange/network error: %s",
                            symbol,
                            e
                        )

                    except Exception as e:

                        logger.exception(
                            "%s | unexpected error: %s",
                            symbol,
                            e
                        )

                # --------------------------------------------------------
                # LOOP TIMING
                # --------------------------------------------------------

                elapsed = (
                    time.time()
                    - cycle_start
                )

                sleep_time = max(
                    5,
                    POLL_SECONDS
                    - elapsed
                )

                await asyncio.sleep(
                    sleep_time
                )

        finally:

            await exchange.close()

            logger.info(
                "Exchange connection closed."
            )


# ============================================================
# START
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

    except Exception as e:

        logger.exception(
            "Fatal error: %s",
            e
        )
